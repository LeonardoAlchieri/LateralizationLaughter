# imports
from sys import path
from random import choice as choose_randomly
from logging import basicConfig, getLogger, INFO, DEBUG
from matplotlib.pyplot import title, savefig, subplots, plot
from pandas import DataFrame, IndexSlice

path.append(".")
from src.utils.io import load_config, load_and_process_csv_data, save_data

basicConfig(filename="logs/run_preloading.log", level=DEBUG)

logger = getLogger("main")


def make_biometrics_plots_together(user_data: DataFrame, user_id: str) -> None:
    fig, axs = subplots(
        len(user_data.index.get_level_values(level=0)), 1, figsize=(10, 10)
    )
    # TODO: copy here the rest of the method in the General Testing notebook
    for n, (data_type, specific_data) in enumerate(user_data.groupby(level=0, axis=0)):
        for side, specific_side_data in specific_data.groupby(level=0, axis=1):
            plot(
                specific_side_data.index.get_level_values(level=1),
                specific_side_data.values,
                ax=axs[n],
            )
        title(f"Data for left and right side, {data_type}")
    savefig(f"./visualizations/all_data/all_data_{user_id}.pdf")


def main():
    path_to_config: str = "src/run/config_preloading.yml"

    logger.info("Starting")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_main_folder: str = configs["path_to_main_folder"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    n_jobs: int = configs["n_jobs"]
    save_format: str = configs["save_format"]
    plots: bool = configs["plots"]

    df = load_and_process_csv_data(
        path_to_main_folder=path_to_main_folder,
        side=None,
        n_jobs=n_jobs,
        save_format=save_format,
    )
    if plots:
        random_user: str = choose_randomly(list(df.index.get_level_values(level=0)))
        logger.info(f"Making plots for user {random_user}")

        make_biometrics_plots_together(
            user_data=df.loc[IndexSlice[random_user, :], :], user_id=random_user
        )

    save_data(
        data_to_save=df,
        filepath=f"{path_to_save_folder}/all_data",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
