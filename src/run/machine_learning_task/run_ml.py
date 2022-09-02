from sys import path
from typing import Callable

path.append("./")

from numpy.random import seed as set_np_seed, choice as random_choice
from random import seed as set_seed
from os.path import basename, join as join_paths
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from pandas import (
    DataFrame,
    read_parquet,
    read_excel,
    IndexSlice,
    Timestamp,
    Timedelta,
    Series,
    MultiIndex,
)
from numpy import datetime64, array, ndarray, concatenate
from pandarallel import pandarallel
from numpy import isnan
from sklearn.model_selection import LeaveOneGroupOut
from numpy import sqrt
from scipy.stats import ttest_ind_from_stats
from tqdm import tqdm

from affect_size.cliff_delta import cliff_delta

from src.utils.io import load_config
from src.utils import INTENSITIES_MAPPING, SESSIONS_GROUPINGS
from src.utils.experiment_info import add_laughter_to_experiment_info
from src.feature_extraction.eda import get_eda_features
from src.feature_extraction.ppg import get_ppg_features
from src.feature_extraction.acc import get_acc_features

from xgboost import XGBClassifier
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


pandarallel.initialize(progress_bar=True)
_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/statistical_analysis/{_filename}.log", level=INFO)
logger = getLogger(_filename)

N_JOBS: int = -1
GAUSSIAN_PROCESS_KERNEL: str | None = "matern"


def perform_data_segmentation(
    all_data: DataFrame,
    experimento_info_w_laugh: DataFrame,
    event_of_interest: str = "laughter",
    baseline_event: str = "baseline_2",
) -> tuple[ndarray, ndarray, ndarray, ndarray]:

    if "laugh" in event_of_interest:
        logger.debug(f"Segmenting data for laughter -> changin to include all sessions")
        event_of_interest = "f0t"

    # TODO: make more efficient
    left_data_list = list()
    right_data_list = list()
    labels_list = list()
    groups_list = list()
    for user in experimento_info_w_laugh.index.get_level_values(0).unique():
        repeat_time: int = 0
        for event in (
            experimento_info_w_laugh.loc[IndexSlice[user, :], :]
            .index.get_level_values(1)
            .unique()
        ):
            if event[:3] == event_of_interest:
                start: Timestamp | datetime64 = experimento_info_w_laugh.loc[
                    IndexSlice[user, event], "start"
                ].unique()[0]
                end = start + Timedelta(seconds=2)
                left_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "left",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                right_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "right",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                left_data_list.append(left_data.values)
                right_data_list.append(right_data.values)
                labels_list.append(1)
                groups_list.append(user)

                start: Timestamp | datetime64 = experimento_info_w_laugh.loc[
                    IndexSlice[user, baseline_event], "start"
                ].unique()[0]
                start = start + Timedelta(seconds=2 * repeat_time)
                end = start + Timedelta(seconds=2)
                left_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "left",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                right_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "right",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                left_data_list.append(left_data.values)
                right_data_list.append(right_data.values)
                labels_list.append(0)
                groups_list.append(user)

                repeat_time += 1
            else:
                continue

    return (
        array(left_data_list),
        array(right_data_list),
        array(labels_list),
        array(groups_list),
    )


def get_random_side(right_side_data: ndarray, left_side_data: ndarray) -> ndarray:
    # NOTE: written by copilot 🤖
    if random_choice([0, 1]):
        return right_side_data
    else:
        return left_side_data


def perform_feature_extraction_over_segments(
    left_data_list: ndarray, right_data_list: ndarray
) -> tuple[ndarray, ndarray, ndarray]:
    hand_crafted_features_lists = list()

    for data_list in [left_data_list, right_data_list]:
        eda_phasic_feats: ndarray = array(
            [
                get_eda_features(data=label_data[:, 2], sampling_rate=4)
                for label_data in data_list
            ]
        )
        eda_mixed_feats: ndarray = array(
            [
                get_eda_features(data=label_data[:, 3], sampling_rate=4)
                for label_data in data_list
            ]
        )

        ppg_feats: ndarray = array(
            [
                get_ppg_features(data=label_data[:, 1], sampling_rate=64)
                for label_data in data_list
            ]
        )
        acc_feats: ndarray = array(
            [get_acc_features(data=label_data[:, 0]) for label_data in data_list]
        )

        hand_crafted_features: ndarray = concatenate(
            [acc_feats, ppg_feats, eda_phasic_feats, eda_mixed_feats], axis=1
        )
        hand_crafted_features_lists.append(hand_crafted_features)

    hand_crafted_features_left: ndarray = hand_crafted_features_lists[0]
    hand_crafted_features_right: ndarray = hand_crafted_features_lists[1]

    hand_crafted_features_random: ndarray = array(
        [
            get_random_side(el_right, el_left)
            for el_right, el_left in zip(
                hand_crafted_features_right, hand_crafted_features_left
            )
        ]
    )

    return (
        hand_crafted_features_left,
        hand_crafted_features_right,
        hand_crafted_features_random,
    )


def train_cv_models(
    ml_models: list[ClassifierMixin],
    hand_crafted_features_left: ndarray,
    hand_crafted_features_right: ndarray,
    hand_crafted_features_random: ndarray,
    labels_list: ndarray,
    groups_list: ndarray,
):
    results = DataFrame()

    for combination, hand_crafted_features_combination in zip(
        [
            "train left - test left",
            "train right - test right",
            "train random - test random",
            "train left - test right",
            "train right - test left",
            "train random - test left",
            "train random - test right",
            "train left - test random",
            "train right - test random",
        ],
        [
            [hand_crafted_features_left, hand_crafted_features_left],
            [hand_crafted_features_right, hand_crafted_features_right],
            [hand_crafted_features_random, hand_crafted_features_random],
            [hand_crafted_features_left, hand_crafted_features_right],
            [hand_crafted_features_right, hand_crafted_features_left],
            [hand_crafted_features_left, hand_crafted_features_random],
            [hand_crafted_features_right, hand_crafted_features_random],
            [hand_crafted_features_random, hand_crafted_features_left],
            [hand_crafted_features_random, hand_crafted_features_right],
        ],
    ):
        (
            hand_crafted_features_train,
            hand_crafted_features_test,
        ) = hand_crafted_features_combination

        hand_crafted_features_train[isnan(hand_crafted_features_train)] = 0
        hand_crafted_features_test[isnan(hand_crafted_features_test)] = 0

        for ml_model in tqdm(ml_models):
            scores = list()
            # NOTE: this is LOSO! Just our own implementation, where the test set, for
            # the user left out, is from a different side
            for user in set(groups_list):
                train_data_mask: ndarray = groups_list != user
                train_data: ndarray = hand_crafted_features_train[train_data_mask]
                test_data_mark: ndarray = groups_list == user
                test_data: ndarray = hand_crafted_features_test[test_data_mark]

                ml_model.fit(train_data, labels_list[train_data_mask])
                scores.append(ml_model.score(test_data, labels_list[test_data_mark]))

            results[f"{combination}_{ml_model.__class__.__name__}"] = scores
    return results


def perform_test_over_results(
    mean_res_clean: DataFrame, groups_list: ndarray, test_type: str = "ttest"
) -> DataFrame:

    if test_type == "ttest":
        test: Callable = ttest_ind_from_stats
    else:
        raise NotImplementedError(f"Test type {test_type} not implemented. Use ttest")

    cv = len(groups_list)

    mean_res_clean["two-sided p value"] = mean_res_clean.apply(
        lambda data: test(
            mean1=data[("left", "mean")],
            std1=data[("left", "se")] * (sqrt(cv) - 1),
            nobs1=cv,
            mean2=data[("right", "mean")],
            std2=data[("right", "se")] * (sqrt(cv) - 1),
            nobs2=cv,
            equal_var=False,
            alternative="two-sided",
        ).pvalue,
        axis=1,
    )

    mean_res_clean["lx > rx p value"] = mean_res_clean.apply(
        lambda data: test(
            mean1=data[("left", "mean")],
            std1=data[("left", "se")] * (sqrt(cv) - 1),
            nobs1=cv,
            mean2=data[("right", "mean")],
            std2=data[("right", "se")] * (sqrt(cv) - 1),
            nobs2=cv,
            equal_var=False,
            alternative="greater",
        ).pvalue,
        axis=1,
    )

    mean_res_clean["lx < rx p value"] = mean_res_clean.apply(
        lambda data: test(
            mean1=data[("left", "mean")],
            std1=data[("left", "se")] * (sqrt(cv) - 1),
            nobs1=cv,
            mean2=data[("right", "mean")],
            std2=data[("right", "se")] * (sqrt(cv) - 1),
            nobs2=cv,
            equal_var=False,
            alternative="less",
        ).pvalue,
        axis=1,
    )
    return mean_res_clean


def aggregate_results(results: DataFrame, groups_list: ndarray) -> DataFrame:
    mean_res: Series = results.mean()
    ses_res: Series = results.std() / (sqrt(len(groups_list)) - 1)
    mean_res_clean = DataFrame(
        columns=MultiIndex.from_tuples(
            [(side, val) for side in ["left", "right"] for val in ["mean", "se"]]
        )
    )
    for col in mean_res.index:
        side, ml_model_name = col.split("_")
        mean_res_clean.loc[ml_model_name, (side, "mean")] = mean_res[col]
        mean_res_clean.loc[ml_model_name, (side, "se")] = ses_res[col]

    mean_res_clean = perform_test_over_results(
        mean_res_clean=mean_res_clean, groups_list=groups_list
    )
    return mean_res_clean


def calculate_cliff_delta(data: DataFrame, sides: list[str]):
    res = cliff_delta(
        s1=data.loc[:, IndexSlice[sides[0], :]].values,
        s2=data.loc[:, IndexSlice[sides[1], :]].values,
        alpha=0.05,
        accurate_ci=True,
    )[0]
    return res


def evaluate_effect_size(results: DataFrame, mean_res_clean: DataFrame) -> DataFrame:
    results.columns = MultiIndex.from_tuples([col.split("_") for col in results])

    # FIXME: not working
    cliff_delta_results = dict()
    cliff_delta_results["cliff delta (lx vs rx)"] = calculate_cliff_delta(
        data=mean_res_clean, sides=["left", "right"]
    )

    cliff_delta_results["cliff delta (lx vs rnd)"] = calculate_cliff_delta(
        data=mean_res_clean, sides=["left", "random"]
    )

    cliff_delta_results["cliff delta (rx vs rnd)"] = calculate_cliff_delta(
        data=mean_res_clean, sides=["right", "random"]
    )

    for key, val in cliff_delta_results.items():
        logger.info(f"Cliff delta {key}: {val}")

    return DataFrame(cliff_delta_results)


def perform_ml(
    hand_crafted_features_left: ndarray,
    hand_crafted_features_right: ndarray,
    hand_crafted_features_random: ndarray,
    labels_list: ndarray,
    groups_list: ndarray,
    seed: int,
    subset_name: str,
    save_results: bool = True,
    **kwargs,
):
    path_to_save: str = kwargs.get("path_to_save", "./results")
    ml_models: list[ClassifierMixin] = [
        KNeighborsClassifier(n_jobs=N_JOBS),
        SVC(random_state=seed, kernel="linear", C=1, probability=False),
        GaussianProcessClassifier(
            n_jobs=N_JOBS,
            copy_X_train=False,
            random_state=seed,
            kernel=Matern(length_scale=1.0, nu=0.1)
            if GAUSSIAN_PROCESS_KERNEL == "matern"
            else None,
        ),  # this is O(m^3) in memory!!!
        DecisionTreeClassifier(random_state=seed),
        AdaBoostClassifier(random_state=seed),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(n_jobs=N_JOBS, random_state=seed),
        XGBClassifier(n_jobs=N_JOBS, random_state=seed),
    ]

    if subset_name == "all":
        pass
    elif "EDA" in subset_name:
        hand_crafted_features_left = hand_crafted_features_left[:, 28:]
        hand_crafted_features_right = hand_crafted_features_right[:, 28:]
        hand_crafted_features_random = hand_crafted_features_random[:, 28:]
    elif "BVP" in subset_name or "PPG" in subset_name:
        hand_crafted_features_left = hand_crafted_features_left[:, 11:28]
        hand_crafted_features_right = hand_crafted_features_right[:, 11:28]
        hand_crafted_features_random = hand_crafted_features_random[:, 11:28]
    elif "ACC" in subset_name:
        hand_crafted_features_left = hand_crafted_features_left[:, :11]
        hand_crafted_features_right = hand_crafted_features_right[:, :11]
        hand_crafted_features_random = hand_crafted_features_random[:, :11]

    results = train_cv_models(
        ml_models=ml_models,
        hand_crafted_features_left=hand_crafted_features_left,
        hand_crafted_features_right=hand_crafted_features_right,
        hand_crafted_features_random=hand_crafted_features_random,
        labels_list=labels_list,
        groups_list=groups_list,
    )
    if save_results:
        results.to_csv(join_paths(path_to_save, f"ml_raw_results_{subset_name}.csv"))

    mean_res_clean = aggregate_results(results=results, groups_list=groups_list)
    if save_results:
        mean_res_clean.to_csv(
            join_paths(path_to_save, f"ml_aggregated_results_{subset_name}.csv")
        )

    # TODO: add cliff delta for new paper. Not implemented in UBICOMP workshop
    # cliff_delta_results = evaluate_effect_size(
    #     results=results, mean_res_clean=mean_res_clean
    # )
    # if save_results:
    #     cliff_delta_results.to_csv(
    #         join_paths(path_to_save, f"ml_cliff_delta_results_{subset_name}.csv")
    #     )


def main(seed: int):
    set_np_seed(seed)
    set_seed(seed)

    path_to_config: str = f"src/run/machine_learning_task/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_experiment_info: str = configs["path_to_experiment_info"]
    path_to_laughter_info: str = configs["path_to_laughter_info"]
    path_to_preprocessed_data: str = configs["path_to_preprocessed_data"]
    event_of_interest: str = configs["event_of_interest"]
    baseline_event: str = configs["baseline_event"]

    experiment_info: DataFrame = read_parquet(path_to_experiment_info)
    laughter_info_data: DataFrame = read_excel(
        path_to_laughter_info, header=0, index_col=0
    )
    all_data: DataFrame = read_parquet(path_to_preprocessed_data)

    (experimento_info_w_laugh, _,) = add_laughter_to_experiment_info(
        laughter_info_data=laughter_info_data, experiment_info=experiment_info
    )

    (
        left_data_list,
        right_data_list,
        labels_list,
        groups_list,
    ) = perform_data_segmentation(
        all_data=all_data,
        experimento_info_w_laugh=experimento_info_w_laugh,
        event_of_interest=event_of_interest,
        baseline_event=baseline_event,
    )

    (
        hand_crafted_features_left,
        hand_crafted_features_right,
        hand_crafted_features_random,
    ) = perform_feature_extraction_over_segments(
        left_data_list=left_data_list, right_data_list=right_data_list
    )

    # ml_subset_names = ['all', 'EDA', 'BVP', 'ACC']
    ml_subset_names = ["EDA"]
    for name in ml_subset_names:
        logger.info(f"Starting ml task for for {name}")
        perform_ml(
            hand_crafted_features_left=hand_crafted_features_left,
            hand_crafted_features_right=hand_crafted_features_right,
            hand_crafted_features_random=hand_crafted_features_random,
            labels_list=labels_list,
            groups_list=groups_list,
            seed=seed,
            subset_name=name,
        )


if __name__ == "__main__":
    seed: int = 42
    main(seed=seed)
