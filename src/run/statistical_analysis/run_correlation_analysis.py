from sys import path

path.append("./")

from pandas import (
    DataFrame,
    read_parquet,
    read_excel,
    IndexSlice,
)
from pandarallel import pandarallel
from numpy.random import seed as set_np_seed
from random import seed as set_seed
from os.path import basename
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from src.utils.io import load_config
from src.utils.correlation import calculate_correlation_coefficients
from src.utils.experiment_info import (
    add_laughter_to_experiment_info,
    add_events_to_signal_data,
)
from src.utils.plots import correlation_heatmap_plot

# NOTE: fixed seed
pandarallel.initialize(progress_bar=True)

_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/statistical_analysis/{_filename}.log", level=INFO)
logger = getLogger(_filename)


def perform_correlation_analysis(
    different_groupings_signal_data_w_laugh: DataFrame,
    signal_name: str,
    make_plot: bool = True,
) -> DataFrame:
    event_correlations = different_groupings_signal_data_w_laugh.groupby(
        axis=0, level=0
    ).apply(calculate_correlation_coefficients)
    event_correlations = event_correlations.loc[:, IndexSlice[:, "value"]]
    event_correlations.columns = event_correlations.columns.droplevel(level=1)
    if make_plot:
        correlation_heatmap_plot(data=event_correlations, signal_name=signal_name)


def main(seed: int):
    set_np_seed(seed)
    set_seed(seed)

    path_to_config: str = f"src/run/statistical_analysis/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_experiment_info: str = configs["path_to_experiment_info"]
    path_to_laughter_info: str = configs["path_to_laughter_info"]
    path_to_preprocessed_data: str = configs["path_to_preprocessed_data"]

    experiment_info: DataFrame = read_parquet(path_to_experiment_info)
    laughter_info_data: DataFrame = read_excel(
        path_to_laughter_info, header=0, index_col=0
    )
    all_data: DataFrame = read_parquet(path_to_preprocessed_data)

    (
        experimento_info_w_laugh,
        sessions_groupings_w_laugh,
    ) = add_laughter_to_experiment_info(
        laughter_info_data=laughter_info_data, experiment_info=experiment_info
    )

    # TODO: find a better way to handle this
    signals_of_interest: list[str] = {
        "EDA_filt_phasic": 4,
        "EDA_filt_stand": 4,
        "BVP_filt": 64,
        "ACC_filt": 32,
    }
    for signal_to_analyse, sampling_rate in signals_of_interest.items():
        logger.info(f"Current signal: {signal_to_analyse}")
        # NOTE: we drop the nans since the data is recorded in the all_data
        # dataframe with the larget (64Hz) sampling rate, and some signals
        # may have smaller sampling rates
        current_data = all_data.loc[:, IndexSlice[:, signal_to_analyse]].dropna(
            how="any"
        )
        different_groupings_signal_data_w_laugh = add_events_to_signal_data(
            signal_data=current_data,
            experiment_info=experiment_info,
            experimento_info_w_laugh=experimento_info_w_laugh,
            sessions_groupings_w_laugh=sessions_groupings_w_laugh,
        )
        perform_correlation_analysis(
            different_groupings_signal_data_w_laugh=different_groupings_signal_data_w_laugh,
            signal_name=signal_to_analyse,
        )


if __name__ == "__main__":
    # NOTE: suggested to fix seed
    seed: int = 11
    main(seed=seed)
