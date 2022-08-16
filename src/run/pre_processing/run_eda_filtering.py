from sys import path
from os import remove as remove_file
from glob import glob
from numpy import ndarray
from pandas import (
    IndexSlice,
    MultiIndex,
    Series,
    concat,
    read_parquet,
)
from time import time
from random import choice as choose_randomly
from logging import basicConfig, getLogger, INFO, DEBUG

path.append(".")
from src.utils.io import load_config, save_data
from src.utils.filters import butter_lowpass_filter_filtfilt, apply_filtering
from src.utils.eda import decomposition
from src.utils.plots import make_lineplot

basicConfig(filename="logs/run_eda_filtering.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/config_eda_filtering.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]
    cutoff_frequency: float = configs["cutoff_frequency"]
    butterworth_order: int = configs["butterworth_order"]
    plots: bool = configs["plots"]
    clean_plots: bool = configs["clean_plots"]

    if clean_plots:
        files_to_remove = glob("./visualizations/EDA/*.pdf")
        for f in files_to_remove:
            remove_file(f)
        del files_to_remove

    all_data = read_parquet(path_to_data)
    # NOTE: the data here is order this way: {side: {user: Series}}, where each pandas
    # Series contains also the `attr` field with the metadata relative to the specific user

    eda_cols: list[tuple[str, str]] = [col for col in all_data.columns if "EDA" in col]
    logger.info("Data loaded correctly.")
    if plots:
        random_side: str = choose_randomly(list(all_data.keys()))
        random_user: str = choose_randomly(list(all_data[random_side].keys()))
        logger.info(f"Making plots for side {random_side} and user {random_user}")

    if plots:
        make_lineplot(
            data=all_data.loc[IndexSlice[random_user, :], IndexSlice[random_side, :]],
            which="EDA",
            savename=f"eda_{random_side[-1]}_{random_user[0]}",
            title="Example EDA",
        )

    eda_data_filtered = (
        all_data[eda_cols]
        .dropna(how="all")
        .groupby(level=0, axis=0, group_keys=True)
        .apply(
            lambda user_data: (
                user_data.groupby(level=0, axis=1, group_keys=True).apply(
                    lambda user_data_side: apply_filtering(
                        data=user_data_side,
                        filter=butter_lowpass_filter_filtfilt,
                        filter_args=dict(
                            cutoff=cutoff_frequency, fs=4, order=butterworth_order
                        ),
                        col_appendix="filt",
                    )
                )
            )
        )
    )

    if plots:
        make_lineplot(
            data=eda_data_filtered.loc[
                IndexSlice[random_user, :], IndexSlice[random_side, :]
            ],
            which="EDA",
            savename=f"eda_filtered_{random_side[-1]}_{random_user[0]}",
            title="Example EDA after filter",
        )
    start = time()

    def get_phasic_component(data_clean: Series | ndarray, frequency: int):
        return decomposition(data_clean, frequency=4)["phasic component"]

    eda_data_phasic = (
        eda_data_filtered.dropna(how="all")
        .groupby(level=0, axis=0, group_keys=True)
        .apply(
            lambda user_data: (
                user_data.groupby(level=0, axis=1, group_keys=True).apply(
                    lambda x: apply_filtering(
                        data=x,
                        filter=get_phasic_component,
                        filter_args=dict(frequency=4),
                        col_appendix="phasic",
                    )
                )
            )
        )
    )

    print("Total phasic component calculation: %.2f s" % (time() - start))
    if plots:
        make_lineplot(
            data=eda_data_phasic.loc[
                IndexSlice[random_user, :], IndexSlice[random_side, :]
            ],
            which="EDA",
            savename=f"eda_phasic_{random_side[-1]}_{random_user[0]}",
            title="Example EDA filtered, phasic component",
        )

    eda_data_standardized = (
        eda_data_filtered.dropna()
        .groupby(level=0, axis=0, group_keys=False)
        .apply(lambda x: (x - x.mean()) / x.std())
    )
    eda_data_standardized.columns = MultiIndex.from_tuples(
        [(col[0], f"{col[1]}_stand") for col in eda_data_standardized.columns]
    )

    if plots:
        make_lineplot(
            data=eda_data_standardized.loc[
                IndexSlice[random_user, :], IndexSlice[random_side, :]
            ],
            which="EDA",
            savename=f"eda_standardized_{random_side[-1]}_{random_user[0]}",
            title="Example EDA filtered & standardized",
        )

    logger.info(
        "Concatenating standardized and phasic components of EDA to all other data"
    )
    data_to_save = concat(
        [all_data, eda_data_standardized, eda_data_phasic], join="outer", axis=1
    ).sort_index()

    # NOTE: this allows to have the columns in order nicely
    data_to_save = data_to_save.reindex(sorted(data_to_save.columns), axis=1)
    save_data(
        data_to_save=data_to_save,
        filepath=f"{path_to_save_folder}/all_data_eda-filt",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
