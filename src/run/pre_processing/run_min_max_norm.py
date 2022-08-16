from sys import path
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, read_parquet
from logging import basicConfig, getLogger, INFO, DEBUG

path.append(".")
from src.utils.io import load_config, save_data

basicConfig(filename="logs/run_min_max_norm.log", level=DEBUG)

logger = getLogger("main")


def user_wise_min_max(data: DataFrame) -> DataFrame:
    """Simple method to apply to a user-wise DataFrame the min max normalization.

    Parameters
    ----------
    data : DataFrame
        dataframe to be column-wise normalized=

    Returns
    -------
    DataFrame
        returns the dataframe for a user where each column has been normalized.
    """
    # NOTE: the scaler is applied column-wise by default
    scaler = MinMaxScaler()
    # NOTE: MinMaxScaler ignores NaN values (which are expected to be a lot of!)
    data[data.columns] = scaler.fit_transform(data[data.columns].values)
    return data


def main():
    path_to_config: str = "src/run/config_min_max_norm.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]

    logger.info("Data loading.")
    all_data = read_parquet(path_to_data)

    logger.info("Data loaded correctly.")

    logger.info("Applying user-wise min max.")

    logger.debug(f"Data before normalization: {all_data.head()}")
    data_to_save: DataFrame = all_data.groupby(level=0, axis=0, group_keys=False).apply(
        user_wise_min_max
    )
    logger.debug(f"Data after normalization: {data_to_save.head()}")
    save_data(
        data_to_save=data_to_save,
        filepath=f"{path_to_save_folder}/all_data_minmax_norm",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
