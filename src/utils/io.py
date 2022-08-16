from typing import Any
from glob import glob
from logging import getLogger
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
from yaml import safe_load as load_yaml
from pandas import DataFrame, MultiIndex, Series, concat, read_csv, read_parquet

from src.utils import get_execution_time, make_timestamp_idx

logger = getLogger("io")


@get_execution_time
def load_config(path: str) -> dict[str, Any]:
    """Simple method to load yaml configuration for a given script.

    Args:
        path (str): path to the yaml file

    Returns:
        Dict[str, Any]: the method returns a dictionary with the loaded configurations
    """
    with open(path, "r") as file:
        config_params = load_yaml(file)
    return config_params


def load_nested_parquet_data(
    path_to_main_folder: str, side: str | None = None, data_type: str | None = None
) -> defaultdict[str, defaultdict[str, dict[str, Series]]] | defaultdict[
    str, dict[str, Series | DataFrame]
]:
    """Simple method to load multiple parquet files in a folder > subfolder structure.
    The main folder should be given, and then the method will crawl and load all of the
    parquet files inside the folder structure, which is expected as:
    ```
    <main_folder>/<side>/<data_type>/files.parquet
    ```
    where `<side>` is expected to be either 'left' or 'right'

    Parameters
    ----------
    path_to_main_folder : str
        path to the folder
    side : str | None, optional
        side for the folder structure; it must thus match it (usually 'left' or 'right'
        expected); by default None. If None, it will be assumed to get both the 'left'
        and 'right' side. Defaults to None.
    data_type : str | None, optional
        data type for the folder structure; it must thus match it (e.g. 'EDA'); by default None.

    Returns
    -------
    defaultdict[str, defaultdict[str, Series | DataFrame]]
        the method returns a loaded default dictionary, with a triple structure:
        ```
        {side: {data_type: {user: Series or DataFrame}}}
        ```
        where `Series` (or `DataFrame`) is a Series (DataFrame) associated with the user
    """
    # FIXME: change from Series to series, or remove one level of dictionary
    if side is None:
        logger.debug("No side provided. Assuming both sides")
        sides: list[str] = ["right", "left"]
    else:
        sides: list[str] = [side]

    all_data_as_dict: defaultdict[str, defaultdict[str, Series]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for chosen_side in sides:
        logger.info(f"Loading for side {chosen_side}")
        all_cleaned_paths: list[str] = glob(
            f"{path_to_main_folder}/{chosen_side}/*/*.parquet"
        )
        if data_type is not None:
            all_cleaned_paths = [
                path for path in all_cleaned_paths if data_type in path
            ]
        logger.debug(f"All files to be loaded: {all_cleaned_paths}")
        tricky_tags: list[str] = ["tags", "IBI", "Table"]

        for path in tqdm(all_cleaned_paths):
            # NOTE: this condition can be tested without casting to list, but it will be
            # slower, for some reason
            if any([tag not in path for tag in tricky_tags]):
                data_loaded: DataFrame = read_parquet(path)
                # NOTE: if only one column is present, return as Series, otherwise as DataFrame
                if len(data_loaded.columns) == 1:
                    all_data_as_dict[chosen_side][path.split("/")[-2]][
                        path.split("/")[-1].split(".")[0]
                    ] = data_loaded.iloc[:, 0]
                else:
                    all_data_as_dict[chosen_side][path.split("/")[-2]][
                        path.split("/")[-1].split(".")[0]
                    ] = data_loaded
                del data_loaded
            else:
                RuntimeWarning(f"Tags {tricky_tags} loading is not implemented yet.")

    if data_type is None:
        return all_data_as_dict
    else:
        return {
            side: inner_dict[data_type] for side, inner_dict in all_data_as_dict.items()
        }


def load_and_process_csv_data(
    path_to_main_folder: str,
    side: str | None = None,
    n_jobs: int = -1,
    save_format: str = "parquet",
) -> DataFrame:
    """Load the data as given by Elena, i.e. in a tested structure of type
    ```
    <user>/<side>/<data_type>.csv
    ```
    The data is loaded and then concatenated into a single DataFrame, to be saved as a
    file, either parquet or csv.

    Parameters
    ----------
    path_to_main_folder : str
        the path to the root of the folder where the nested strucurre is expected
    side : str | None, optional
        side to be extracted, if None both 'left' and 'right', by default None
    n_jobs : int, optional
        number of jobs to parallelize loading (see `joblib`), by default -1
    save_format : str, optional
        format to save data in, currently accepted 'csv' or 'parquet', by default 'parquet'

    Raises
    ------
    ValueError
        if a `save_format` differnet from 'csv' or 'parquet' is given, an error is raised
    """

    if side is None:
        logger.debug("No side provided. Assuming both sides")
        sides: list[str] = ["right", "left"]
    else:
        sides: list[str] = [side]
    del side  # NOTE: do not want to risk having stuff around not needed

    logger.info(f"Loading csv data from {path_to_main_folder}")
    all_data_as_dict: defaultdict[str, defaultdict[str, DataFrame]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for chosen_side in sides:
        test_all_paths_left: list[str] = glob(
            f"{path_to_main_folder}/*/{chosen_side}/*.csv"
        )
        logger.debug(f"All files to be loaded: {test_all_paths_left}")

        for path in tqdm(test_all_paths_left):
            if "tags" not in path and "IBI" not in path and "Table" not in path:
                all_data_as_dict[chosen_side][path.split("/")[-3]][
                    path.split("/")[-1].split(".")[0]
                ] = read_csv(path, header=[0, 1])
            else:
                continue

    del chosen_side

    def make_inner_concat(individual_name: str, inner_dict: dict, side: str):
        return concat(
            [
                make_timestamp_idx(
                    dataframe=dataframe.copy(),
                    side=side,
                    data_name=data_name,
                    individual_name=individual_name,
                )
                for data_name, dataframe in inner_dict.items()
            ],
            axis=1,
            join="outer",
        ).sort_index()

    logger.info("Starting indexing and concatenations")

    df = concat(
        [
            concat(
                Parallel(n_jobs=n_jobs)(
                    delayed(make_inner_concat)(individual_name, inner_dict, chosen_side)
                    for individual_name, inner_dict in all_data_as_dict[
                        chosen_side
                    ].items()
                ),
                axis=0,
                join="outer",
            ).sort_index()
            for chosen_side in all_data_as_dict.keys()
        ],
        axis=1,
        join="outer",
    ).sort_index()
    # TODO: I should probably change from the loading paradigm and the indexing part
    # right above here, but I could not find an interesting way to do it
    return df


def read_experimentinfo(path: str, user: str) -> DataFrame:
    """Simple method to read the experiment info file for a given user.

    Parameters
    ----------
    path : str
        path to the csv file, usually of the type <user>/experiment_info.csv
    user : str
        user name, e.g. s099

    Returns
    -------
    DataFrame
        returns the dataframe loaded and with the correct multi-level index
    """
    df: DataFrame = read_csv(path, index_col=0)
    df.index = MultiIndex.from_tuples([(user, idx) for idx in df.index])
    return df


def save_data(
    data_to_save: DataFrame, filepath: str, save_format: str = "parquet"
) -> None:
    """Simple auxiliary method to save dataframe to a file, but can handle both parquet
    and csv using just a simple string call.

    Parameters
    ----------
    data_to_save : DataFrame
        dataframe to save
    filepath : str
        file to the dataframe, without extention, which will be added depending on the
        `save_format` given
    save_format : str, optional
        save format for the data, either 'csv' or 'parquet' at the moment, by default 'parquet'

    Raises
    ------
    ValueError
        if a non-supported save format is asked, the method will raise an error
    """
    logger.info("Saving data")
    logger.debug(f"Save format selected: {save_format}")
    if save_format == "parquet":
        data_to_save.to_parquet(f"{filepath}.parquet")
    elif save_format == "csv":
        data_to_save.to_csv(f"{filepath}.csv")
    else:
        raise ValueError(
            f'{save_format} is not a valid format. Accepted: "parquet" or "csv"'
        )
    logger.info("Data saved successfully")
