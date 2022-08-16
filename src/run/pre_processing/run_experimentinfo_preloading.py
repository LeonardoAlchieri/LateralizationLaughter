# This script just takes the different csv files for the experiment info and puts them together
from glob import glob
from sys import path
from logging import basicConfig, getLogger, INFO, DEBUG

from pandas import DataFrame, concat, to_datetime

path.append(".")
from src.utils.io import load_config, read_experimentinfo, save_data


basicConfig(filename="logs/run_eda_filtering.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/config_experimentinfo_preloading.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]

    all_experimentinfo_paths: list[str] = glob(f"{path_to_data}/*/experiment_info.csv")
    logger.debug(f"All paths identified: {all_experimentinfo_paths}")
    logger.info("Loading and joining experiment info")
    all_experimentinfo_joined: DataFrame = concat(
        [
            read_experimentinfo(path=path, user=path.split("/")[4])
            for path in all_experimentinfo_paths
        ],
        axis=0,
    )

    all_experimentinfo_joined.loc[:, ["start", "end"]] = all_experimentinfo_joined.loc[
        :, ["start", "end"]
    ].apply(to_datetime)

    save_data(
        data_to_save=all_experimentinfo_joined,
        filepath=f"{path_to_save_folder}/all_experimento_info",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
