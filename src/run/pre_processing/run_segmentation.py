from logging import basicConfig, getLogger, INFO, DEBUG
from pandas import (
    read_parquet,
    read_excel,
    MultiIndex,
    to_datetime,
    DataFrame,
    IndexSlice,
    Timestamp,
    concat,
)
from numpy import nan as NaN
from sys import path
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

path.append(".")
from src.utils.io import load_config, save_data

basicConfig(filename="logs/run_segmentation.log", level=DEBUG)

logger = getLogger("main")

INTENSITIES_MAPPING: dict[str, float] = {"low": 0, "medium": 1, "high": 2}


def make_cut(
    user_data: DataFrame, event_id: str, start: Timestamp, end: Timestamp
) -> DataFrame:
    """Simple method to actually apply the cut.

    Parameters
    ----------
    user_data : DataFrame
        user dataframe to cut
    event_id : str
        event id for the laughter episode
    start : Timestamp
        start of the laughter episode
    end : Timestamp
        end of the laughter episode

    Returns
    -------
    DataFrame
        dataframe cut around the stand and end of the laughter episode
    """
    start: Timestamp = start.tz_localize("Europe/Rome")
    end: Timestamp = end.tz_localize("Europe/Rome")
    single_event = user_data.loc[IndexSlice[:, start:end], :]
    single_event.index = MultiIndex.from_tuples(
        [(idx[0], event_id, idx[1]) for idx in single_event.index]
    )
    return single_event


def make_user_segmentations(
    user_data: DataFrame, laughter_info_data: DataFrame
) -> DataFrame:
    """Method to apply segmentation a single user timeseries data (for all variables).
    This is intended to be run with a pandas apply method.

    Parameters
    ----------
    user_data : DataFrame
        dataframe containing as rows the timeseries of a single user, and columns the
        different variables, divided by side.
    laughter_info_data : DataFrame
        dataframe contaning for rows the user id and laughter episode, and for columns
        start and end of the laughter episode.

    Returns
    -------
    DataFrame
        returns a dataframe with a 3-level index: user, event_id (for laughter),
        and timestamp.
    """
    user: str = user_data.index.unique(level=0)[0]
    if user in laughter_info_data.index.get_level_values(0):
        segments = laughter_info_data.loc[IndexSlice[user, :], ["start", "end"]]
    else:
        return None

    return concat(
        [
            make_cut(user_data=user_data, start=start, end=end, event_id=row_idx[-1])
            for row_idx, (start, end) in segments.iterrows()
        ],
        axis=0,
    )


def main():
    path_to_config: str = "src/run/config_segmentation.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_laughter_info: str = configs["path_to_laughter_info"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]

    # TODO: imlpement method to read using save_format paradigm -> EVERYWHERE

    all_data = read_parquet(path_to_data)
    logger.info("Loaded data")
    laughter_info_data = read_excel(path_to_laughter_info, header=0, index_col=0)
    logger.info("Loaded laughter episode info. Cleaning a bit.")
    laughter_info_data.index = MultiIndex.from_tuples(
        [tuple(idx.split("_")) for idx in laughter_info_data.index]
    )
    laughter_info_data = laughter_info_data.sort_index()

    laughter_info_data["intensity"] = (
        laughter_info_data["intensity"]
        .apply(lambda x: INTENSITIES_MAPPING.get(x, NaN))
        .astype(float)
    )
    # NOTE: NaN > 0 => False, so I filter all expect medium & high intensities
    laughter_info_data = laughter_info_data[laughter_info_data["intensity"] > 0]

    laughter_info_data[["start", "end"]] = laughter_info_data[["start", "end"]].apply(
        to_datetime
    )
    logger.info("Finished laughter info cleaning.")

    logger.info("Applying segmentation.")
    res: DataFrame = all_data.groupby(level=0, axis=0).parallel_apply(
        make_user_segmentations, laughter_info_data=laughter_info_data
    )
    logger.info("Segmentation completed. Saving...")

    save_data(
        data_to_save=res,
        filepath=f"{path_to_save_folder}/all_data_segmented_laughter",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
