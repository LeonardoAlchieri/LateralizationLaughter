from pandas import DataFrame, Series, MultiIndex, concat
from scipy.stats import pearsonr, spearmanr, kendalltau, wilcoxon
from affect_size.cliff_delta import cliff_delta
from numpy import isnan, nan, sign


def calculate_correlation_coefficients(x: DataFrame) -> DataFrame:
    """Simple method to calculate a few correlation coeffietients betweenn the left and
    right columns of a given dataframe. At the moment, Pearson's, Spearman's and Kendall's
    coefficients are calculated.

    Parameters
    ----------
    x : DataFrame
        Dataframe with columns to be correlated.

    Returns
    -------
    DataFrame
        the output is a dataframe containing the correlation coefficients between the left and right side.
    """

    pearson = Series(
        pearsonr(x.loc[:, "left"].values, x.loc[:, "right"].values),
        index=MultiIndex.from_tuples(
            [("Pearson's ρ", "value"), ("Pearson's ρ", "p-value")]
        ),
    )
    spearman = Series(
        tuple(spearmanr(x.loc[:, "left"].values, x.loc[:, "right"].values)),
        index=MultiIndex.from_tuples(
            [("Spearman's ρ", "value"), ("Spearman's ρ", "p-value")]
        ),
    )
    kendall = Series(
        tuple(kendalltau(x.loc[:, "left"].values, x.loc[:, "right"].values)),
        index=MultiIndex.from_tuples(
            [("Kendall's τ", "value"), ("Kendall's τ", "p-value")]
        ),
    )
    return concat([pearson, spearman, kendall], axis=0, join="outer")


def calculate_statistical_test(
    x: DataFrame, correction: bool = False, test_method: str = "wilcoxon"
) -> Series:
    """Simple method to calculate a statistical test (two sides) between the left and
    right side of the body (signals) of a given dataframe. At the moment, Wilcoxon's

    Parameters
    ----------
    x : DataFrame
        input dataframe with columns to be confronted.
    correction : bool, optional
        if True, a correction for the test shall be implemented (see scipy documentation), by default False
    test_method : str, optional
        test methdo to implement, by default 'wilcoxon'; allowed values:
        * 'wilcoxon'

    Returns
    -------
    Series
        returns a series w/ index the statistical test value and the p-values

    Raises
    ------
    ValueError
        if an unknown test is given, a ValueError is raised.
    """

    implemented_methods: list[str] = ["wilcoxon"]
    match test_method:
        case "wilcoxon":
            statistical_test = Series(
                wilcoxon(
                    x=x.loc[:, "left"].values,
                    y=x.loc[:, "right"].values,
                    zero_method="wilcox",
                    alternative="two-sided",
                    correction=correction,
                ),
                index=MultiIndex.from_tuples(
                    [("Wilcoxon Test", "statistics"), ("Wilcoxon Test", "p-value")]
                ),
            )
        case _:
            raise ValueError(
                f"{test_method} is not a valid statistical test method. Please choose one of the following: {implemented_methods}"
            )

    return statistical_test


def calculate_cliff_delta(x: DataFrame, alpha: float = 0.05) -> Series:
    """Simple method to calculate the cliff delta between the left and right side of the body (signals) of a given dataframe.

    Parameters
    ----------
    x : DataFrame
        input dataframe with columns to be confronted.
    alpha : float, optional
        significance level for the confidence interval calculation, by default 0.05


    Returns
    -------
    Series
        returns a series w/ index the cliff delta value and its confidence interval
    """

    cliff_delta_res = Series(
        cliff_delta(
            s1=x.loc[:, "left"].values,
            s2=x.loc[:, "right"].values,
            alpha=alpha,
            accurate_ci=True,
            raise_nan=False,
        ),
        index=MultiIndex.from_tuples(
            [("Cliff Delta", "value"), ("Cliff Delta", "confidence interval")]
        ),
    )

    return cliff_delta_res


def get_cliff_bin(
    x: Series, dull: list[str] | None = None, raise_nan: bool = False
) -> int:
    """Simple method to get the cliff bin of a given series. The bin is given as a number between 0 and 3.

    Parameters
    ----------
    x : Series
        the input value to be binned
    dull : list[str] | None, optional
        a disctionary containing the bins, by default None. If None, the default bins are used,
        as proposed b Vargha and Delaney (2000)
    raise_nan : bool, optional
        if True, when a NaN is found, a ValueError is raised, by default False

    Returns
    -------
    int
        a value referring to the bin selected; the value as a sign

    Raises
    ------
    ValueError
        if raise_nan is True and a NaN is found, a ValueError is raised.
    ValueError
        is some value other than Nan or float is found, a ValueError is raised.
    """

    x_sign: int = sign(x)
    x: Series = abs(x)
    if dull is None:
        dull: dict[str, str] = {
            "small": 0.11,
            "medium": 0.28,
            "large": 0.43,
        }  # effect sizes from (Vargha and Delaney (2000)) "negligible" for the rest=
    if x < dull["small"]:
        return 0 * x_sign
    elif dull["small"] <= x < dull["medium"]:
        return 1 * x_sign
    elif dull["medium"] <= x < dull["large"]:
        return 2 * x_sign
    elif x >= dull["large"]:
        return 3 * x_sign
    else:
        if isnan(x):
            if raise_nan:
                raise ValueError("NaN value")
            else:
                return nan
        else:
            raise ValueError(f"{x} is not in the dull range")
