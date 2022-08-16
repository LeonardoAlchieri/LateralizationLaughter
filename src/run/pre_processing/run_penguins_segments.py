from sys import path
from os.path import join as join_paths

# The penguin segments, in the experiment info, have wrong ending points. The script here
# attempts to perform this correction
from logging import basicConfig, getLogger, INFO, DEBUG
from pandas import read_parquet

path.append(".")
from src.utils.io import load_config, read_experimentinfo, save_data


basicConfig(filename="logs/run_eda_filtering.log", level=DEBUG)

logger = getLogger("main")

from pandas import DataFrame, Series
from numpy import datetime64


def correct_penguins(user_data: DataFrame) -> DataFrame:
    # penguins_times: Series = user_data.loc['penguins', :]
    penguins_idx: int = user_data.index.get_level_values(1).get_loc("penguins")
    real_penguins_end: datetime64 = user_data.iloc[penguins_idx + 1, 0]
    user_data.iloc[penguins_idx, 1] = real_penguins_end
    user_data.iloc[penguins_idx, 2] = (
        user_data.iloc[penguins_idx, 1] - user_data.iloc[penguins_idx, 0]
    ).total_seconds()
    return user_data


def main():
    path_to_config: str = "src/run/config_penguins_segments.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]

    experimento_info = read_parquet(path_to_data)
    experimento_info_corrected = experimento_info.groupby(
        level=0, axis=0, group_keys=False
    ).apply(correct_penguins)

    save_data(
        data_to_save=experimento_info_corrected,
        filepath=join_paths(path_to_save_folder, "all_experimento_info_corrected"),
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
