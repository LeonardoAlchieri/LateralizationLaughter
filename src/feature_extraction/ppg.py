from logging import getLogger
from numpy import ndarray, array, isnan, mean, std, trapz, gradient, ediff1d, zeros
from scipy.stats import linregress
from neurokit2.ppg import ppg_findpeaks
from neurokit2.hrv import hrv_time

logger = getLogger(__name__)


def get_ppg_features(data: ndarray, sampling_rate: int = 64) -> ndarray:
    """This method computes the BVP features, as done by Di Lascio et al. (2019). The
    features extracted are:
    - **Time-domain**: statistical features (minimum, maximum, mean, standard deviation,
    dynamic change, slope, area under the curve, number of peaks, ratio between number
    of peaks and length of the segment, mean and standard deviation of the first and
    second derivative, difference between the highest and smallest peak), HR, statistical
    features on HRV (mean of all NN, standard deviation of all NN (SDNN), standard
    deviation of differences between NN (SDSD), the square root of the mean of the
    sum of the squares of differences between NN (RMSSD)


    Parameters
    ----------
    data : ndarray
        bvp data
    sampling_rate : int, optional
        sample rate, in Hz, of the signal, by default 64

    Returns
    -------
    ndarray
        array containing the features extracted, ordered as described above
    """
    # - **BVP pulse**: mean and standard deviation of pulses’ amplitude and pulses’ length
    data: ndarray = data[~isnan(data)]
    if len(data) == 0:
        return zeros(17)
    else:
        min_feat: float = min(data)
        max_feat: float = max(data)
        mean_feat: float = mean(data)
        std_feat: float = std(data)
        dynamic_range_feat: float = max_feat - min_feat
        slope_feat, _, _, _, _ = linregress(range(len(data)), data)
        auc_feat: float = trapz(data, dx=1 / sampling_rate)
        try:
            ppg_peaks: ndarray = ppg_findpeaks(data, sampling_rate=sampling_rate)[
                "PPG_Peaks"
            ]
        except ValueError as e:
            logger.warning(
                f"Some problems with the evaluation of PPG peaks. Most likely due to the way the data was filtered (expected different filter by neurokit2). See {e}"
            )
            return zeros(17)
        except IndexError as e:
            logger.warning(
                f"Some problems with the evaluation of PPG peaks. Most likely due to the way the data was filtered (expected different filter by neurokit2). See {e}"
            )
            return zeros(17)
        num_ppg_peaks_feat: int = len(ppg_peaks)
        ratio_peaks_segment_length_feat: float = num_ppg_peaks_feat / len(data)
        if num_ppg_peaks_feat == 0:
            ratio_highest_smallest_peak_feat = 0
        else:
            ratio_highest_smallest_peak_feat: float = max(data[ppg_peaks]) / min(
                data[ppg_peaks]
            )

        first_derivative_data: ndarray = gradient(data)
        first_derivetive_mean_feat: float = mean(first_derivative_data)
        first_derivative_std_feat: float = std(first_derivative_data)

        hr: ndarray = eval_hr(ppg_peaks, sampling_rate=sampling_rate)
        hr_mean_feat: float = mean(hr)
        if len(ppg_peaks) <= 1:
            # FIXME: this is not really a solution, and some stuff could be done even if
            # there is only 1 peak
            hrv_features: ndarray = zeros(4)
        else:
            hrv_features = hrv_time(ppg_peaks, sampling_rate=sampling_rate)[
                ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD"]
            ].values[0, :]

        # TODO: there should be a few BVP pulse features, but I have no idea what they are
        return array(
            [
                min_feat,
                max_feat,
                mean_feat,
                std_feat,
                dynamic_range_feat,
                slope_feat,
                auc_feat,
                num_ppg_peaks_feat,
                ratio_peaks_segment_length_feat,
                ratio_highest_smallest_peak_feat,
                first_derivetive_mean_feat,
                first_derivative_std_feat,
                hr_mean_feat,
                hrv_features[0],
                hrv_features[1],
                hrv_features[2],
                hrv_features[3],
            ]
        )


def eval_hr(ppg_peaks: ndarray, sampling_rate: int = 64) -> ndarray:
    """A stripped down version of the neurokit2 library to evaluate HR. In our case, we
    do not care about interpolation, minimum length and the sort.

    Parameters
    ----------
    ppg_peaks : ndarray
        array of peaks. Should be in the same format as the output of the
    sampling_rate : int, optional
        sample rate in Hz, by default 64

    Returns
    -------
    ndarray
        array with the heart rate for each peak
    """
    period = ediff1d(ppg_peaks, to_begin=0) / sampling_rate
    period[0] = mean(period[1:])
    return 60 / period
