from pandas import DataFrame, concat, to_datetime, MultiIndex
from copy import deepcopy
from numpy import nan

from src.utils import (
    slice_user_over_experiment_time,
    INTENSITIES_MAPPING,
    SESSIONS_GROUPINGS,
)


def add_events_to_signal_data(
    signal_data: DataFrame,
    experiment_info: DataFrame,
    experimento_info_w_laugh: DataFrame,
    sessions_groupings_w_laugh: list[str],
) -> DataFrame:
    """This method adds the events to the signal dataframe, using a multi-index
    structure

    Parameters
    ----------
    signal_data : DataFrame
        raw data with the signal values
    experiment_info : DataFrame
        dataframe with information regarding the experiment
    experimento_info_w_laugh : DataFrame
        dataframe with the experiment info and the laughter info
    sessions_groupings_w_laugh : list[str]
        list of ways to group the different events

    Returns
    -------
    DataFrame
        dataframe with a 3-level multi index structure, w/ event, user, timeframe
    """
    signal_data = signal_data.groupby(level=0, axis=0, group_keys=False).apply(
        slice_user_over_experiment_time,
        experimento_info=experiment_info,
        slicing_col="experiment",
    )
    signal_data.columns = signal_data.columns.droplevel(1)
    # TODO: add parallel computation here
    different_groupings_signal_data_w_laugh: dict[str, DataFrame] = concat(
        [
            signal_data.groupby(level=0, axis=0, group_keys=False).apply(
                slice_user_over_experiment_time,
                experimento_info=experimento_info_w_laugh,
                slicing_col=session,
            )
            for session_group in sessions_groupings_w_laugh.values()
            for session in session_group
        ],
        keys=[
            f"{group_name}%{event}"
            for group_name, group_item in sessions_groupings_w_laugh.items()
            for event in group_item
        ],
        names=["grouping"],
    )
    different_groupings_signal_data_w_laugh.index = MultiIndex.from_tuples(
        [
            (el[0].split("%")[0], el[0].split("%")[1], el[1], el[2])
            for el in different_groupings_signal_data_w_laugh.index
        ],
        names=["group", "event", "user", "timestamp"],
    )

    return different_groupings_signal_data_w_laugh


def add_laughter_to_experiment_info(
    laughter_info_data: DataFrame, experiment_info: DataFrame
) -> tuple[DataFrame, list[str]]:
    """Simple method to add laughter episodess to the experiment info dataframe.
    Will also perform some simple cleaning.

    Parameters
    ----------
    laughter_info_data : DataFrame
        dataframe with laughter data
    experiment_info : DataFrame
        dataframe with experiment info, i.e., informations regarding the
        events performed

    Returns
    -------
    DataFrame
        returns a dataframe with all of the experiment info and the laughter info,
        with a multiindex structure; and a list of the experiment events
        grouping with the laughter as well
    """
    laughter_info_data.index = MultiIndex.from_tuples(
        [tuple(idx.split("_")) for idx in laughter_info_data.index]
    )
    laughter_info_data = laughter_info_data.sort_index()
    laughter_info_data["intensity"] = (
        laughter_info_data["intensity"]
        .apply(lambda x: INTENSITIES_MAPPING.get(x, nan))
        .astype(float)
    )

    laughter_info_data = laughter_info_data[laughter_info_data["intensity"] > 0]
    laughter_info_data[["start", "end"]] = laughter_info_data[["start", "end"]].apply(
        to_datetime
    )

    sessions_groupings_w_laugh = deepcopy(SESSIONS_GROUPINGS)
    sessions_groupings_w_laugh["laughter_episodes"] = list(
        laughter_info_data.index.get_level_values(1).unique()
    )

    # Join the laughter info with the experimento info
    experimento_info_w_laugh = concat(
        [laughter_info_data.iloc[:, :3], experiment_info], axis=0
    ).sort_index()

    return experimento_info_w_laugh, sessions_groupings_w_laugh
