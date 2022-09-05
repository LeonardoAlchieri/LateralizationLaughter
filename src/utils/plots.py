from itertools import cycle
from numpy import ndarray, asarray, mean, std, unique
from os.path import join as join_paths
from pandas import DataFrame, Series
from matplotlib.pyplot import (
    figure,
    savefig,
    plot,
    xlabel,
    style,
    ylabel,
    title as figtitle,
    legend as make_legend,
)
from matplotlib.axes._subplots import SubplotBase
from matplotlib.pyplot import xticks, xlabel, ylabel, legend, cm, show, title
from matplotlib.patches import Patch
from seaborn import set, heatmap


def bland_altman_plot(
    data: DataFrame, ax: SubplotBase, *args, **kwargs
) -> tuple[SubplotBase, dict[str, float]]:
    """Simple script to evaluate the Bland-Atlman plot between the left and right signal.

    Parameters
    ----------
    data : DataFrame
        data to be plotted, must have 'left' and 'right' columns
    ax : SubplotBase
        axis to plot the plot

    Returns
    -------
    SubplotBase
        the method returns the input axis
    """
    data1: Series = data.loc[:, "left"]
    data2: Series = data.loc[:, "right"]
    data1: ndarray = asarray(data1)
    data2: ndarray = asarray(data2)
    avg = mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = mean(diff)  # Mean of the difference
    sd = std(diff, axis=0)  # Standard deviation of the difference

    ax.scatter(avg, diff, *args, **kwargs)
    ax.axhline(md, color="gray", linestyle="--")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--")
    return ax, dict(mean_diff=md, std_diff=sd)


def make_lineplot(
    data: ndarray | Series | DataFrame,
    which: str,
    savename: str,
    title: str | None = None,
    style_label: str = "seaborn",
):
    """Simple script to make lineplot for time series data. Different methods shall be
    implemented for different types of data in the project, e.g. EDA, BVP etc.

    Parameters
    ----------
    data : ndarray | Series | DataFrame
        data to be plotted; according to the type of data, different methods shall be
        implemented
    which : str
        defines the type of data; current accepted values: 'EDA'
    savename : str
        filename to save the plot (not the whole path)
    title : str | None, optional
        title for the plot, by default None
    style_label : str, optional
        defines the styl for matplot, by default 'seaborn'

    Raises
    ------
    NotImplementedError
        the type of data accepted for 'EDA' is only ndarray, pandas Series or pandas DataFrame.
        If another is given, the method will fail.
    NotImplementedError
        The function will fail if a `which` value different than the accepted ones is given.
    """
    linestyles: cycle[list[str]] = cycle(["-", "-.", "--", ":"])

    with style.context(style_label):
        match which:
            case ("EDA" | "BVP" | "ACC"):
                figure(figsize=(10, 5))
                if isinstance(data, Series):
                    plot(data.index, data.values, label=which)
                elif isinstance(data, ndarray):
                    plot(data, label=which)
                elif isinstance(data, DataFrame):
                    for i, column in enumerate(data.columns):
                        if which in column[-1]:
                            data_to_plot = data[column].dropna()
                            plot(
                                data_to_plot.index.get_level_values("timestamp"),
                                data_to_plot,
                                label=column[-1],
                                linewidth=2,
                                alpha=0.7,
                                linestyle=next(linestyles),
                            )
                        make_legend()
                else:
                    raise NotImplementedError(
                        f"Still have not implemented plot for type {type(data)}."
                    )
                if title:
                    figtitle(title)
                xlabel("Time")
                ylabel(f"{which} value (mV)")
                # TODO: imlement using path joining, and not string concatenation
                savefig(f"./visualizations/{which}/{savename}.pdf")
            case _:
                raise NotImplementedError(
                    f'Unknown plot type "{which}". Currently implemented: "EDA"'
                )


def statistical_test_plot(
    test_results: DataFrame,
    signal_name: str,
    path_to_save: str = "./visualizations/",
    test_name: str = "Wilcoxon",
    threshold: float = 0.05,
) -> None:
    set(font_scale=1.8)
    df_to_save = test_results.iloc[:, 2].unstack(level=1)
    cmap = cm.coolwarm
    figure(figsize=(24, 5))
    heatmap(
        df_to_save,
        xticklabels=df_to_save.columns,
        # vmax=0.2, vmin=-.2, center=0,
        # annot=df_to_save.replace({True: 'Significant', False: 'Non Significant'}, inplace=False).values,
        annot=test_results.iloc[:, 1].round(decimals=3).unstack(level=1).values,
        cmap=cmap,
        fmt="",
        cbar=False,
        yticklabels=df_to_save.index,
    )

    xticks(rotation=30, ha="right")
    xlabel("Feature")
    ylabel("Event")
    title(f"P values of {test_name} test for {signal_name} features")
    custom_handles = [
        Patch(facecolor=cmap(0.0), edgecolor=cmap(0.0), label="Non Significant"),
        Patch(facecolor=cmap(1.0), edgecolor=cmap(1.0), label="Significant"),
    ]
    legend(
        handles=custom_handles,
        bbox_to_anchor=(1.253, 1.05),
        title=f"P value significance ({threshold} threshold)",
    )
    savefig(
        join_paths(path_to_save, f"{test_name}_statistical_heatmap_{signal_name}.pdf"),
        bbox_inches="tight",
    )


def cliff_delta_plot(
    cliff_delta_bins: DataFrame,
    cliff_delta_results_vals: DataFrame,
    signal_name: str,
    path_to_save: str = "./visualizations/",
) -> None:
    cmap = cm.coolwarm
    set(font_scale=2.89)
    figure(figsize=(13, 9))
    ax = heatmap(
        cliff_delta_bins,
        xticklabels=cliff_delta_results_vals.columns,
        vmax=3,
        vmin=-3,
        center=0,
        annot=cliff_delta_results_vals.round(decimals=3),
        cmap=cmap,
        fmt="",
        cbar=False,
        yticklabels=cliff_delta_results_vals.index.str.replace("_", " "),
    )
    ax.tick_params(left=True, bottom=True)
    xticks(rotation=30, ha="right")
    xlabel("Feature")
    ylabel("Event")
    title(f"Cliff Delta values ({signal_name.replace('_filt', '').replace('_', ' ')})", fontsize=40)

    def make_custom_handles(bins: DataFrame, cmap) -> list[Patch]:
        labels: list[Patch] = list()
        for el in unique(bins.values):
            if el == 0:
                labels.append(
                    Patch(facecolor=cmap(0.5), edgecolor=cmap(0.5), label="negligible")
                )
            elif el == 1:
                labels.append(
                    Patch(
                        facecolor=cmap(0.6), edgecolor=cmap(0.6), label="small (left)"
                    )
                )
            elif el == 2:
                labels.append(
                    Patch(
                        facecolor=cmap(0.8), edgecolor=cmap(0.8), label="medium (left)"
                    )
                )
            elif el == 3:
                # NOTE: use 1. and not 1, otherwise something goes wronf with cmap
                labels.append(
                    Patch(
                        facecolor=cmap(1.0), edgecolor=cmap(1.0), label="large (left)"
                    )
                )
            elif el == -1:
                labels.append(
                    Patch(
                        facecolor=cmap(0.4), edgecolor=cmap(0.4), label="small (right)"
                    )
                )
            elif el == -2:
                labels.append(
                    Patch(
                        facecolor=cmap(0.2), edgecolor=cmap(0.2), label="medium (right)"
                    )
                )
            elif el == -3:
                labels.append(
                    Patch(
                        facecolor=cmap(0.0), edgecolor=cmap(0.0), label="large (right)"
                    )
                )

        return labels

    custom_handles = make_custom_handles(bins=cliff_delta_bins, cmap=cmap)
    legend(
        handles=custom_handles,
        bbox_to_anchor=(1.75, 1.05),
        title="Cliff Delta effect (dominant side)",
    )
    savefig(
        join_paths(path_to_save, f"cliff_delta_{signal_name}.pdf"),
        bbox_inches="tight",
    )


def correlation_heatmap_plot(data: DataFrame, signal_name: str, path_to_save: str = './visualizations/') -> None:
    set(font_scale=1.6)
    figure(figsize=(7, 5))
    ax = heatmap(data, 
            xticklabels=data.columns,
            vmax=1, vmin=-1, center=0,
            cmap='coolwarm',
            yticklabels=data.index.str.replace("_", " "), 
            annot=True)
    ax.tick_params(left=True, bottom=True)
    ax.set_ylabel('Event')
    title(f"Correlation coefficient per event ({signal_name.replace('_filt', '').replace('_', ' ')})", fontsize=20)
    savefig(
        join_paths(path_to_save, f"correlation_heatmap_{signal_name}.pdf"),
        bbox_inches="tight",
    )