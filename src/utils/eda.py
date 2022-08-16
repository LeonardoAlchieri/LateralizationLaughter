from numpy import array, ndarray
from pandas import Series
from cvxEDA import cvxEDA

from src.utils import blockPrinting

# See https://github.com/lciti/cvxEDA for more EDA analysis methdos


@blockPrinting
def decomposition(
    eda_signal: Series | ndarray | list, frequency: int = 4
) -> dict[str, ndarray]:
    """This method will apply the cvxEDA decomposition to an EDA signal. The cvxEDA
    implementation is the one from Greco et al.

    Parameters
    ----------
    eda_signal : Series | ndarray | list
        eda signal to be decomposed
    Fs : int, optional
        frequency of the input signal, e.g. 64Hz

    Returns
    -------
    dict[str, ndarray]
        the method returns a dictionary with the decomposed signals
        (see cvxEDA for more details)
    """
    # TODO: see if the standardization is actually something we want!
    yn = standardize(eda_signal=eda_signal)
    return cvxEDA(yn, 1.0 / frequency)


# TODO: probably remove and use some third party library
def standardize(eda_signal: Series | ndarray | list) -> ndarray:
    """Simple method to standardize an EDA signal.

    Parameters
    ----------
    eda_signal : Series | ndarray | list
        eda signal to standardize

    Returns
    -------
    ndarray
        returns an array standardized
    """
    y: ndarray = array((eda_signal))
    yn: ndarray = (y - y.mean()) / y.std()
    return yn
