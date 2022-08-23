from logging import getLogger
from typing import Any
from numpy import ndarray, mean, std, gradient, isnan, array, zeros
from scipy.stats import linregress

logger = getLogger(__name__)


def get_acc_features(data: ndarray, **kwargs) -> ndarray:
    """This method performs the feature extraction for an ACC signal.
    The features extracted are: minimum, maximum, mean, standard deviation, dynamic change,
    slope, absolute value of the slope, mean and standard deviation of the first and
    second derivative.
    The features extracted follow what done by Di Lascio et al. (2019).

    Parameters
    ----------
    data : ndarray
        acc data to extract features from.

    Returns
    -------
    ndarray
        the method returns an array of extracted features, in the order given in the
        description
    """

    data: ndarray = data[~isnan(data)]
    logger.debug(f"Len of eda data after removal of NaN: {len(data)}")
    if len(data) == 0:
        return zeros(9)
    else:
        min_feat: float = min(data)
        max_feat: float = max(data)
        mean_feat: float = mean(data)
        std_feat: float = std(data)
        dynamic_range_feat: float = max_feat - min_feat
        slope_feat, _, _, _, _ = linregress(range(len(data)), data)
        absolute_slope_feat: float = abs(slope_feat)
        first_derivative_data: ndarray = gradient(data)
        first_derivetive_mean_feat: float = mean(first_derivative_data)
        first_derivative_std_feat: float = std(first_derivative_data)
        second_derivative_data: ndarray = gradient(first_derivative_data)
        second_derivative_mean_feat: float = mean(second_derivative_data)
        second_derivative_std_feat: float = std(second_derivative_data)

        return array(
            [
                min_feat,
                max_feat,
                mean_feat,
                std_feat,
                dynamic_range_feat,
                slope_feat,
                absolute_slope_feat,
                first_derivetive_mean_feat,
                first_derivative_std_feat,
                second_derivative_mean_feat,
                second_derivative_std_feat,
            ]
        )
