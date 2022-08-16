from sys import path
from os import remove as remove_file
from glob import glob
from pandas import IndexSlice, concat, read_parquet
from random import choice as choose_randomly
from logging import basicConfig, getLogger, INFO, DEBUG

path.append(".")
from src.utils.io import load_config, save_data
from src.utils.filters import movin_avg_acc
from src.utils.plots import make_lineplot

basicConfig(filename="logs/run_acc_filtering.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/config_acc_filtering.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]
    window_size: int = configs["window_size"]
    plots: bool = configs["plots"]
    clean_plots: bool = configs["clean_plots"]

    if clean_plots:
        files_to_remove = glob("./visualizations/ACC/*.pdf")
        for f in files_to_remove:
            remove_file(f)
        del files_to_remove

    all_data = read_parquet(path_to_data)
    # NOTE: the data here is order this way: {side: {user: Series}}, where each pandas
    # Series contains also the `attr` field with the metadata relative to the specific user

    acc_cols = [col for col in all_data.columns if "ACC" in col[-1]]
    logger.info("Data loaded correctly.")
    if plots:
        random_side: str = choose_randomly(list(all_data.keys()))
        random_user: str = choose_randomly(list(all_data[random_side].keys()))
        logger.info(f"Making plots for side {random_side} and user {random_user}")

    if plots:
        make_lineplot(
            data=all_data.loc[IndexSlice[random_user, :], IndexSlice[random_side, :]],
            which="ACC",
            savename=f"acc_{random_side[-1]}_{random_user[0]}",
            title="Example ACC",
        )

    # TODO: implement attrs!
    # FIXME: add groupby over user! This is wrong!
    acc_data_filtered = (
        all_data[acc_cols]
        .dropna(how="all")
        .groupby(level=0, axis=0, group_keys=True)
        .apply(
            lambda user_data: (
                user_data.groupby(level=0, axis=1, group_keys=True).apply(
                    lambda user_data_side: movin_avg_acc(
                        user_data_side,
                        window_size,
                        index=None,
                    )
                )
            )
        )
    )

    if plots:
        make_lineplot(
            data=acc_data_filtered.loc[
                IndexSlice[random_user, :], IndexSlice[random_side, :]
            ],
            which="ACC",
            savename=f"acc_filtered_{random_side[-1]}_{random_user[0]}",
            title="Example ACC after filter",
        )

    logger.info(
        "Concatenating standardized and phasic components of ACC to all other data"
    )

    logger.debug(
        f'ACC data filtered w/o NaN (how=any): {acc_data_filtered.dropna(how="any")}'
    )
    logger.debug(
        f'ACC data filtered w/o NaN (how=all): {acc_data_filtered.dropna(how="all")}'
    )
    data_to_save = concat(
        [all_data, acc_data_filtered], join="outer", axis=1
    ).sort_index()

    # NOTE: this allows to have the columns in order nicely
    data_to_save = data_to_save.reindex(sorted(data_to_save.columns), axis=1)
    save_data(
        data_to_save=data_to_save,
        filepath=f"{path_to_save_folder}/all_data_acc-filt",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
