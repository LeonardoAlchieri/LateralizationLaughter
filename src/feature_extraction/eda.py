from logging import getLogger
from typing import Any
from numpy import ndarray, mean, std, gradient, isnan, array
from scipy.stats import linregress
from neurokit2.eda import eda_peaks

logger = getLogger(__name__)


def get_eda_features(data: ndarray, sampling_rate: int = 4) -> ndarray:
    """This method performs the feature extraction for an EDA signal (be it mixed or phasic).
    The features extracted are: statistical features (minimum, maximum, mean, standard deviation,
    difference between maximum and minimum value or dynamic change, slope, absolute value
    of the slope, mean and standard deviation of the first derivative), number of peaks,
    peaks’ amplitude.
    The features extracted follow what done by Di Lascio et al. (2019).

    Parameters
    ----------
    data : ndarray
        eda data to extract features from.
    sampling_rate : int, optional
        sampling rate of the eda features, in Hz, by default 4.

    Returns
    -------
    ndarray
        the method returns an array of extracted features, in the order given in the
        description, i.e.,
        `[min, max, mean, std, diff_max_min, slope, absolute_slope, mean_derivative,
        std_derivative,number_peaks,peaks_amplitude]`
    """

    data: ndarray = data[~isnan(data)]
    logger.debug(f"Len of eda data after removal of NaN: {len(data)}")
    if len(data) == 0:
        return array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
        try:
            eda_peaks_result: dict[str, Any] = eda_peaks(
                data,
                sampling_rate=sampling_rate,
            )
        except ValueError as e:
            # NOTE: sometimes, when no peaks are detected, as ValueError is thrown by the
            # neurokit2 method. We solve this in a very simplistic way
            logger.warning(f"Could not extract EDA peaks. Reason: {e}")
            eda_peaks_result: tuple[None, dict[str, Any]] = (
                None,
                dict(SCR_Peaks=[], SCR_Amplitude=[0]),
            )

        number_of_peaks_feat: int = len(eda_peaks_result[1]["SCR_Peaks"])
        # NOTE: I am not sure that the sum of the amplitudes is the correct feature to be
        # extracted
        peaks_amplitude_feat: float = sum(eda_peaks_result[1]["SCR_Amplitude"])

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
                number_of_peaks_feat,
                peaks_amplitude_feat,
            ]
        )
