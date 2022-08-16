# NOTE: this file contains some scripts useful for cleaning, filter and
# similar time serie biometrics data
from distutils.log import warn
from logging import getLogger
from typing import Callable
from numpy import amax, nan, ndarray, pad
from pandas import DataFrame, IndexSlice, Series, array, Index
from scipy.signal import butter, lfilter, filtfilt

logger = getLogger("filters")


def apply_filtering(
    data: DataFrame | Series | list | ndarray,
    filter: Callable,
    filter_args: dict,
    col_appendix: str | None = "filt",
) -> DataFrame:
    """Method to be used to apply a filtering function to a timeseries data, e.g. EDA,
    in this project. At the moment, the method can be used for EDA, BVP and ACC.

    Parameters
    ----------
    data : DataFrame | Series | list | ndarray
        data to be filtered; it will be converted into a Pandas Series, so that is the
        preferred way
    filter : Callable
        filter to be applied to "data"
    filter_args : dict
        additional arguments to the filter
    col_appendix : str
        appendix to put over the column names of the `data` dataframe when returned
        filtered; defaults to 'filt'
        If `None`, no change shall be applied to the column.

    Returns
    -------
    DataFrame
        a DataFrame with the filtered data, the same index as the input `data` and the
        same columns, plus `col_appendix`.
    """
    if isinstance(data, ndarray) or isinstance(data, list):
        # NOTE: this falicitates the dropna
        data: Series = Series(data)
    elif isinstance(data, DataFrame):
        # NOTE: consider only the first
        data: Series = data.iloc[:, 0]
    # NOTE: essential, otherwise the filter will be empty
    data_clean = data.dropna()
    if len(data_clean) == 0:
        warn("The current user does not have an EDA signal. Returning all NaNs")
        y = data.values
        idx: Index = data.index.get_level_values(1)
        cols: list[str] = [f"{data.name[-1]}_{col_appendix}"]
    else:
        y = filter(data_clean, **filter_args)
        idx: Index = data_clean.index.get_level_values(1)
        cols: list[str] = [f"{data_clean.name[-1]}_{col_appendix}"]
    return DataFrame(y, index=idx, columns=cols)


def butter_lowpass(cutoff: float, fs: int, order: int) -> tuple[float, float]:
    """Figure out the numerator (`b`) and denominator (`a`) coefficients for a linear digital
    filter

    Args:
        cutoff (float): cutoff value
        fs (int): frequency of the signal
        order (int): order for the Butterworth filter

    Returns:
        (tuple): numerator and denominator coefficients for the lowpass filter
    """
    # NOTE: Nyquist frequeny is half the sampling frequency
    nyq = 0.5 * fs
    # NOTE: Normalization of the cutoff signal
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter_filtfilt(
    data: Series, cutoff: float, fs: int, order: int
) -> ndarray:
    """Method to create and then apply a digital filter forward and backward to a signal.

    Args:
        data (ndarray): timeseries to be filtered, e.g. EDA data signal
        cutoff (float): cutoff point for the lowpass, e.g. `5`
        fs (int): frequency of the singal, e.g. `64`
        order (int): order for the filter, e.g. `2`

    Returns:
        ndarray: returns the data filtered
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y: ndarray = filtfilt(b=b, a=a, x=data.values)
    return y


def butter_lowpass_filter_lfilter(
    data: ndarray, cutoff: float, fs: int, order: int
) -> ndarray:
    """Method to create and then apply an IIR or FIR filter.

    Args:
        data (ndarray): timeseries to be filtered, e.g. EDA data signal
        cutoff (float): cutoff point for the lowpass, e.g. `5`
        fs (int): frequency of the singal, e.g. `64`
        order (int): order for the filter, e.g. `2`

    Returns:
        ndarray: returns the data filtered
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y: ndarray = lfilter(b=b, a=a, x=data)
    return y


def movin_avg_acc(
    acc_df: ndarray | DataFrame, window_size: int, index: None | ndarray | Index = None
) -> ndarray:
    """Simple method to evaluate the moving average, as performed by Empatica, over the
    ACC signal. Keep in mind that this is mot trully a real moving average, but slightly
    differnet.
    Indeed, the description is shown in Notes.

    Parameters
    ----------
    acc_df : ndarray | DataFrame
        _description_
    window_size : int
        _description_
    index : None | ndarray | Index, optional
        _description_, by default None

    Returns
    -------
    ndarray
        _description_

    Raises
    ------
    TypeError
        _description_

    Notes
    -----
    Formula for acc filter: this is basically an exponential moving average, but where the maximum for a given direction is considered.
    .. math::
        X_{\mathrm{filt}, t} = (1-\alpha)\cdot X_{\mathrm{filt}, t-1} + \alpha \cdot f(X_{x,t},X_{y,t},X_{z,t})


    where:
    .. math::
        f(X_{x,t},X_{y,t},X_{z,t}) = f(x_t, y_t, z_t)=  \sum_{j=1}^{M-1} \max \left( |x_{t, j} - x_{t, j-1}|, |y_{t, j} - y_{t, j-1}|, |z_{t, j} - z_{t, j-1}| \right)

    with :math:`M` being the window size, to be considered of 32 steps (1 second of data),
    as per [empatica](https://support.empatica.com/hc/en-us/articles/202028739-How-is-the-acceleration-data-formatted-in-E4-connect-).
    Indeed, the arrays :math:`x_t`, :math:`y_t` and :math:`z_t` are of length :math:`M`.
    """
    if isinstance(acc_df, DataFrame):
        # warn(f'acc_data is a DataFrame. Converting to ndarray.')
        acc_df: ndarray = acc_df.dropna(how="all")
        acc_data: ndarray = acc_df.values
        if index is None:
            index: Index = acc_df.index.get_level_values(level=1)
    elif isinstance(acc_df, ndarray):
        acc_data: ndarray = acc_df
    else:
        raise TypeError(f"acc_data is not a ndarray or DataFrame.")

    acc_data = acc_data / 64
    logger.debug(f"Acc data: {acc_data} as received (dropped nan).")
    # NOTE: this is called assignment expression! -> faster and more elegant
    avg: float = 0
    # NOTE: average over the 3 acc axis
    differences = amax(abs(acc_data[1:] - acc_data[:-1]), axis=1)
    avgs: ndarray = array(
        [
            avg := 0.9 * avg + 0.1 * sum(differences[n : n + window_size]) / 32
            for n in range(0, len(acc_data) - window_size, 1)
        ]
    )
    logger.debug(
        f"Lenght of avg filteres: {len(avgs)}; lengths of inout data: {len(acc_data)}"
    )
    # NOTE: I pad to nans in order to have an array of the same length as the input one
    avgs = pad(avgs, (-len(avgs) + len(acc_data), 0), "constant", constant_values=nan)
    logger.debug(f"Average evaluated: {avgs}")
    logger.debug(
        f"Lenght of avg filteres after padding: {len(avgs)}; lengths of inout data: {len(acc_data)}"
    )
    logger.debug(f"Length of index: {len(index)}")

    return DataFrame(avgs, index=index, columns=["ACC_filt"])
