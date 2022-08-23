from typing import Callable
from src.feature_extraction.eda import get_eda_features
from src.feature_extraction.acc import get_acc_features
from src.feature_extraction.ppg import get_ppg_features
from pandas import DataFrame
from logging import getLogger

logger = getLogger(__name__)


def get_signal_feature_extraction(colname: str) -> tuple[Callable, list[str]]:
    """Method to return the feature extraction parameter for a signal.

    Parameters
    ----------
    colname : str
        name of the column over which to perform the feature extraction

    Returns
    -------
    tuple of Callable and list[str]
        method to perform feature extraction and the corresponding names for
        the features extracted

    Raises
    ------
    ValueError
        if the column name is not recognized, a ValueError is raised
    """
    signals_of_interest: list[str] = ["EDA", "BVP", "ACC"]
    if "EDA" in colname:
        return get_eda_features, [
            "min",
            "max",
            "mean",
            "std",
            "dynamic range",
            "slope",
            "absolute slope",
            "mean first derivative",
            "std first derivative",
            "number of peaks",
            "peaks amplitude",
        ]
    elif "ACC" in colname:
        return get_acc_features, [
            "min",
            "max",
            "mean",
            "std",
            "dynamic range",
            "slope",
            "absolute slope",
            "mean first derivetive",
            "std first derivative",
            "mean second derivative",
            "std second derivative",
        ]
    elif "BVP" in colname or "PPG" in colname:
        return get_ppg_features, [
            "min",
            "max",
            "mean",
            "std",
            "dynamic range",
            "slope",
            "auc",
            "num pgg peaks",
            "ratio peaks segment length",
            "ratio highest smallest peak",
            "mean first derivative",
            "std first derivative",
            "hr mean",
            "hrv mean NN",
            "hrv SDNN",
            "hrv RMSSD",
            "hrv SDSD",
        ]
    else:
        raise ValueError(
            f"{colname} is not a valid signal. Please choose from {signals_of_interest}"
        )


def extract_features(
    x: DataFrame,
    feature_extraction_method: Callable,
    feature_names: list[str],
    sampling_rate: float | int,
) -> DataFrame | None:
    """Method to apply feature extraction to a signal.

    Parameters
    ----------
    x : DataFrame
        dataframe with data for a single event and individual
    feature_extraction_method: Callable
        method to perform feature extraction. Should be one of the methods
        in `src/feature_extraction/eda.py` or `src/feature_extraction/acc.py`
        or `src/feature_extraction/ppg.py`
    feature_names : list[str]
        names of the features to extract
    sampling_rate : float | int
        sampling rate of the signal

    Returns
    -------
    DataFrame or None
        the method returns a dataframe with the features extracted or None if
        not enough data is present (less than 5)
    """
    if len(x) > 5:
        return DataFrame(
            [
                feature_extraction_method(data=x["left"], sampling_rate=sampling_rate),
                feature_extraction_method(data=x["right"], sampling_rate=sampling_rate),
            ],
            columns=feature_names,
            index=["left", "right"],
        ).T
    else:
        logger.warn("Not enough data to extract features")
        return None
