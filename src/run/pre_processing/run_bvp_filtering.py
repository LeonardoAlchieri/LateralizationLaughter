from sys import path
from os import remove as remove_file
from glob import glob
from pandas import DataFrame, IndexSlice, read_parquet, concat, MultiIndex
from random import choice as choose_randomly
from logging import basicConfig, getLogger, INFO, DEBUG

path.append(".")
from src.utils.io import load_config, save_data
from src.utils.filters import butter_lowpass_filter_lfilter
from src.utils.plots import make_lineplot

basicConfig(filename="logs/run_bvp_filtering.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/config_bvp_filtering.yml"

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
        files_to_remove = glob("./visualizations/BVP/*.pdf")
        for f in files_to_remove:
            remove_file(f)
        del files_to_remove

    all_data = read_parquet(path_to_data)
    # NOTE: the data here is order this way: {side: {user: Series}}, where each pandas
    # Series contains also the `attr` field with the metadata relative to the specific user

    bvp_cols = [col for col in all_data.columns if "BVP" in col]

    logger.info("Data loaded correctly.")
    if plots:
        random_side: str = choose_randomly(list(all_data.keys()))
        random_user: str = choose_randomly(list(all_data[random_side].keys()))
        logger.info(f"Making plots for side {random_side} and user {random_user}")

    if plots:
        make_lineplot(
            data=all_data.loc[IndexSlice[random_user, :], IndexSlice[random_side, :]],
            which="BVP",
            savename=f"bvp_{random_side[-1]}_{random_user[0]}",
            title="Example BVP",
        )

    # TODO: implement attrs!
    # FIXME: I do not like the fact that the frequency is hard-coded!
    # FIXME: add groupby over user! This is wrong!
    logger.info("Applying Butterworth lowpass filter.")
    logger.debug(f"Cutoff frequency: {cutoff_frequency}")
    # bvp_data_filtered = all_data[bvp_cols].copy()
    bvp_data_filtered = (
        all_data[bvp_cols]
        .dropna(how="all")
        .groupby(level=0, axis=0, group_keys=True)
        .apply(
            lambda user_data: DataFrame(
                butter_lowpass_filter_lfilter(
                    data=user_data,
                    cutoff=cutoff_frequency,
                    fs=64,
                    order=butterworth_order,
                ),
                index=user_data.index.get_level_values(level=1),
                columns=MultiIndex.from_tuples(
                    [(col[0], f"{col[1]}_filt") for col in bvp_cols]
                ),
            )
        )
    )

    logger.info("Filter applied correctly.")

    if plots:
        make_lineplot(
            data=bvp_data_filtered.loc[
                IndexSlice[random_user, :], IndexSlice[random_side, :]
            ],
            which="BVP",
            savename=f"bvp_filtered_{random_side[-1]}_{random_user[0]}",
            title="Example BVP after filter",
        )

    # NOTE: this allows to have the columns in order nicely
    logger.info("Saving filtered data...")

    data_to_save = concat(
        [all_data, bvp_data_filtered], join="outer", axis=1
    ).sort_index()
    data_to_save = all_data.copy()
    for col in bvp_data_filtered.columns:
        data_to_save[col] = bvp_data_filtered[col]

    data_to_save = data_to_save.reindex(sorted(data_to_save.columns), axis=1)
    save_data(
        data_to_save=data_to_save,
        filepath=f"{path_to_save_folder}/all_data_bvp-filt",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
