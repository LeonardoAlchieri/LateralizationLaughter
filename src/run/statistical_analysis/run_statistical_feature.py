from sys import path

path.append("./")

from pandas import (
    DataFrame,
    read_parquet,
    read_excel,
    IndexSlice,
)
from pandarallel import pandarallel
from numpy.random import seed as set_np_seed
from random import seed as set_seed
from os.path import basename
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from src.utils.io import load_config
from src.utils import get_cliff_bin
from src.utils.correlation import calculate_statistical_test
from src.utils.plots import statistical_test_plot, cliff_delta_plot
from src.utils.feature_extraction import get_signal_feature_extraction, extract_features
from src.utils.correlation import calculate_cliff_delta
from src.utils.experiment_info import (
    add_laughter_to_experiment_info,
    add_events_to_signal_data,
)

# NOTE: fixed seed
pandarallel.initialize(progress_bar=True)

_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/statistical_analysis/{_filename}.log", level=INFO)
logger = getLogger(_filename)


def perform_cliff_delta(
    signal_features: DataFrame,
    signal_name: str,
    make_plot: bool = True,
    subset_features_plot: dict[str, list[str]] | dict[None] = dict(),
) -> DataFrame:
    """Method to perform cliff delta effect sixze between the left and right side
    feature extracted for a specific signal

    Parameters
    ----------
    signal_features : DataFrame
        dataframe with extracted features
    signal_name : str
        name of the signal, used for plotting purposes
    make_plot : bool, optional
        if True, a heatmap with the results will be made, by default True
    subset_features_plot : dict[str, list[str]], optional
        if not empty, a dictionary with the subset of columns to be plotted,
        by default dict()

    Returns
    -------
    DataFrame
        returns the dataframe with the calculated cliff delta values
    """
    cliff_delta_results = signal_features.groupby(level=[0, 3], axis=0).apply(
        calculate_cliff_delta
    )
    cliff_delta_results_vals = cliff_delta_results.iloc[:, 0].unstack(level=1)
    cliff_delta_bins = cliff_delta_results_vals.applymap(get_cliff_bin)
    signal_name_short = signal_name[:3]
    if subset_features_plot.get(signal_name_short, None) is not None:
        logger.info(
            f"Subsetting features for plotting: {subset_features_plot[signal_name_short]}"
        )
        cliff_delta_bins = cliff_delta_bins[subset_features_plot[signal_name_short]]
        cliff_delta_results_vals = cliff_delta_results_vals[
            subset_features_plot[signal_name_short]
        ]
    if make_plot:
        cliff_delta_plot(
            cliff_delta_bins=cliff_delta_bins,
            cliff_delta_results_vals=cliff_delta_results_vals,
            signal_name=signal_name,
        )
    return cliff_delta_results_vals


def perform_statistical_test(
    signal_features: DataFrame,
    signal_name: str,
    p_val_threshold: float = 0.05,
    make_plot: bool = True,
) -> DataFrame:
    """Method to perform statistical test between the left and right side
    feature extracted for a specific signal

    Parameters
    ----------
    signal_features : DataFrame
        dataframe with extracted features1
    signal_name : str
        name of the signal, used for plotting purposes
    p_val_threshold : float, optional
        threshold for the p value, by default 0.05
    make_plot : bool, optional
        if True, a heatmap with the results will be made, by default True

    Returns
    -------
    DataFrame
        the method returns the dataframe with the results of the statistical test
    """
    test_results = signal_features.groupby(level=[0, 3], axis=0).apply(
        calculate_statistical_test
    )
    test_results["Significant?"] = test_results.iloc[:, 1] < p_val_threshold
    if make_plot:
        statistical_test_plot(
            test_results=test_results,
            signal_name=signal_name,
            threshold=p_val_threshold,
        )
    return test_results


def main(seed: int = 0):
    set_np_seed(seed)
    set_seed(seed)

    path_to_config: str = f"src/run/statistical_analysis/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_experiment_info: str = configs["path_to_experiment_info"]
    path_to_laughter_info: str = configs["path_to_laughter_info"]
    path_to_preprocessed_data: str = configs["path_to_preprocessed_data"]
    drop_nan_feature: bool = configs["drop_nan_feature"]
    subset_features_plot: dict[str, list[str]] = configs["subset_features_plot"]

    experiment_info: DataFrame = read_parquet(path_to_experiment_info)
    laughter_info_data: DataFrame = read_excel(
        path_to_laughter_info, header=0, index_col=0
    )
    all_data: DataFrame = read_parquet(path_to_preprocessed_data)

    (
        experimento_info_w_laugh,
        sessions_groupings_w_laugh,
    ) = add_laughter_to_experiment_info(
        laughter_info_data=laughter_info_data, experiment_info=experiment_info
    )

    # TODO: find a better way to handle this
    signals_of_interest: list[str] = {
        "EDA_filt_phasic": 4,
        "EDA_filt_stand": 4,
        "BVP_filt": 64,
        "ACC_filt": 32,
    }
    for signal_to_analyse, sampling_rate in signals_of_interest.items():
        logger.info(f"Current signal: {signal_to_analyse}")
        # NOTE: we drop the nans since the data is recorded in the all_data
        # dataframe with the larget (64Hz) sampling rate, and some signals
        # may have smaller sampling rates
        current_data = all_data.loc[:, IndexSlice[:, signal_to_analyse]].dropna(
            how="any"
        )
        different_groupings_signal_data_w_laugh = add_events_to_signal_data(
            signal_data=current_data,
            experiment_info=experiment_info,
            experimento_info_w_laugh=experimento_info_w_laugh,
            sessions_groupings_w_laugh=sessions_groupings_w_laugh,
        )
        feature_extraction_method, feature_names = get_signal_feature_extraction(
            colname=signal_to_analyse
        )
        signal_features = different_groupings_signal_data_w_laugh.groupby(
            level=[0, 1, 2], axis=0
        ).parallel_apply(
            extract_features,
            feature_extraction_method=feature_extraction_method,
            sampling_rate=sampling_rate,
            feature_names=feature_names,
        )
        if drop_nan_feature:
            logger.info("Dropping nan features")
            signal_features = signal_features.dropna(how="any", inplace=False)
        
        perform_statistical_test(
            signal_features=signal_features, signal_name=signal_to_analyse
        )

        perform_cliff_delta(
            signal_features=signal_features,
            signal_name=signal_to_analyse,
            subset_features_plot=subset_features_plot,
        )


if __name__ == "__main__":
    # NOTE: suggested to fix seed
    seed: int = 11
    main(seed=seed)
