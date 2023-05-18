import numpy as np
import numpy.typing as npt
from functools import lru_cache
import pandas as pd
import scipy.stats as stat
from typing import Literal, Union


@lru_cache(maxsize=20, typed=True)
def profit_factor(
    avg_win_return: float,
    avg_loss_return: float,
) -> float:
    """The profit factor is a measure of how much profit you make per dollar that you lose. It is defined as the ratio of the total amount of money won to the total amount of money lost

    Args:
    -----
        avg_win_return (float): The average winning trade return.

        avg_loss_return (float): The average losing trade return.

    Returns:
    -----
        float: The profit factor, it has no unit.
    """
    return abs(avg_win_return / avg_loss_return)


@lru_cache(maxsize=20, typed=True)
def expectancy(
    winrate: float,
    avg_win_return: float,
    avg_loss_return: float,
) -> float:
    """Expectancy is a measure of the % average amount of money, (or percentage if you prefer) that you can expect to win (or lose) per trade.

    Args:
    -----
        winrate (float): The percentage of winning trades.

        avg_win_return (float): The average winning trade return.

        avg_loss_return (float): The average losing trade return.

    Returns:
    -----
        float: The expectancy.
    """
    return (1 + profit_factor(avg_win_return, avg_loss_return)) * winrate - 1


def sharpe_ratio(
    returns: pd.Series,
    N: Union[int, float] = 365,
    risk_free_rate: float = 0.03,
) -> float:
    """The economist William F. Sharpe proposed the Sharpe ratio in 1966 as an extension of his work on the Capital Asset Pricing Model (CAPM). It is defined as the difference between the returns of the investment and the risk-free return, divided by the standard deviation of the investment.

    Args:
    -----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

        risk_free_rate (float, optional): The risk free rate usually 10-year bond, buy-and-hold or 0. Defaults to 0.0.

    Returns:
    -----
        float: The annualized sharpe ratio.
    """
    return (returns.mean() * N - risk_free_rate) / (returns.std() * (N**0.5))


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    N: Union[int, float] = 365,
    risk_free_rate: float = 0.03,
) -> float:
    """The Treynor ratio is a risk-adjusted measure of return based on systematic risk. It is similar to the Sharpe ratio, except it uses beta as the measurement of volatility instead of standard deviation.

    Args:
    -----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio benchmark not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

        risk_free_rate (float, optional): The risk free rate usually 10-year bond, buy-and-hold or 0. Defaults to 0.0.

    Returns:
    -----
        float: The annualized treynor ratio.
    """
    beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(returns)
    return (returns.mean() * N - risk_free_rate) / (beta * (N**0.5))


def sortino_ratio(
    returns: pd.Series,
    N: Union[int, float] = 365,
    risk_free_rate: float = 0.03,
) -> float:
    """The Sortino ratio is very similar to the Sharpe ratio, the only difference being that where the Sharpe ratio uses all the observations for calculating the standard deviation the Sortino ratio only considers the harmful variance.

    Args:
    -----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

        risk_free_rate (float, optional): The risk free rate usually 10-year bond, buy-and-hold or 0. Defaults to 0.0.

    Returns:
    -----
        float: The annualized sortino ratio.
    """
    return (returns.mean() * N - risk_free_rate) / (downside_risk(returns) * (N**0.5))


def calmar_ratio(
    returns: pd.Series,
    N: Union[int, float] = 365,
) -> float:
    """The final risk/reward ratio we will consider is the Calmar ratio. This is similar to the other ratios, with the key difference being that the Calmar ratio uses max drawdown in the denominator as opposed to standard deviation.

    Args:
    -----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

    Returns:
    -----
        float: The annualized calmar ratio.
    """
    return (returns.mean() * N) / abs(max_drawdown(returns))


def downside_risk(returns: pd.Series) -> float:
    """Downside risk or Semi-Deviation is a method of measuring the fluctuations below the mean, unlike variance or standard deviation it only looks at the negative price fluctuations and it's used to evaluate the downside risk (The risk of loss in an investment) of an investment.

    Args:
    -----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

    Returns:
    ------
        float: The semi-deviation or downside risk of returns.
    """
    return returns.loc[returns < 0].std()


def drawdown(returns: pd.Series) -> pd.Series:
    """Computes the drawdown series of a given returns (not cumulative) time series.

    Args:
    ----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

    Returns:
    ----
        pd.Series: The drawdown series.
    """
    cum_ret = cumulative_returns(returns)
    running_max = cum_ret.cummax()
    return (cum_ret - running_max) / running_max


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Computes the cumulative returns series of a given returns (not cumulative) time series.

    Args:
    ----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

    Returns:
    -----
        pd.Series: The cumulative returns series.
    """
    return (returns + 1).cumprod()


def max_drawdown(
    returns: pd.Series,
) -> float:
    """Max drawdown quantifies the steepest decline from peak to trough observed for an investment.

    Args:
    ----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

    Returns:
    ----
        float: The max drawdown.
    """
    return drawdown(returns).min()


def information_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """The information ratio (IR) is a measurement of portfolio returns beyond the returns of a benchmark, usually an index, compared to the volatility of those returns. The information ratio (IR) measures a portfolio manager's ability to generate excess returns relative to a benchmark but also attempts to identify the consistency of the investor.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio benchmark not cumulative returns.

    Returns:
    -----
        float: The annualized information ratio.
    """
    return (portfolio_returns - benchmark_returns).mean() / tracking_error(
        portfolio_returns, benchmark_returns
    )


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Tracking error is the divergence between the price behavior of a position or a portfolio and the price behavior of a benchmark.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio benchmark not cumulative returns.

    Returns:
    -----
        float: The annualized tracking error.
    """
    return (portfolio_returns - benchmark_returns).std()


def kelly_criterion(returns: pd.Series) -> float:
    p = (returns > 0).mean()
    q = 1 - p
    win = returns[returns > 0].mean()
    loss = returns[returns < 0].mean()
    r = win / abs(loss)
    return float((p * r - q) / r)


def value_at_risk(
    returns: pd.Series,
    level: int = 5,
    method: Literal["historic", "gaussian", "cornish_fischer"] = "historic",
) -> float:
    """Returns the VaR of a Series or DataFrame using the specified method.

    Args:
    -----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

        level (int, optional): Percentile to compute, which must be between 0 and 100 inclusive. Defaults to 5.

        method (Literal[&quot;historic&quot;, &quot;gaussian&quot;, &quot;cornish_fischer&quot;], optional): The method for VaR calculation : historic use the returns provided, gaussian will approximate the returns by a a gaussian parametric distribution and will correct the gaussian VaR using skewness and kurtosis. Defaults to "historic".

    Returns:
    -----
        float: The historical VaR.
    """
    if method == "historic":
        return float(np.percentile(returns.to_numpy(), level))
    elif method == "gaussian":
        return float((returns.mean() + stat.norm.ppf(level / 100) * returns.std()))
    elif method == "cornish_fischer":
        z = stat.norm.ppf(level / 100)
        s = stat.skew(returns.values)
        k = stat.kurtosis(returns.values)

        return float(
            (
                returns.mean()
                + (
                    z
                    + (z**2 - 1) * s / 6
                    + (z**3 - 3 * z) * (k - 3) / 24
                    - (2 * z**3 - 5 * z) * (s**2) / 36
                )
                * returns.std()
            )
        )
    else:
        raise ValueError(
            "VaR calculation method must be historic, gaussian or cornish_fischer"
        )


def conditional_value_at_risk(
    returns: pd.Series,
    level: int = 5,
    method: Literal["historic", "gaussian", "cornish_fischer"] = "historic",
) -> float:
    """Returns the CVaR (conditional value-at-risk) also called expected shortfall of a Series or DataFrame

    Args:
    ----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

        level (int, optional): Percentile to compute, which must be between 0 and 100 inclusive. Defaults to 5.

        method (Literal[&quot;historic&quot;, &quot;gaussian&quot;, &quot;cornish_fischer&quot;], optional): The method for VaR calculation : historic use the returns provided, gaussian will approximate the returns by a a gaussian parametric distribution and will correct the gaussian VaR using skewness and kurtosis. Defaults to "historic".

    Returns:
    ----
        float: The CVaR.
    """
    if method == "historic":
        historic_CVaR = np.mean(
            returns[returns < value_at_risk(returns, level, "historic")].to_numpy()
        )
        return (
            float(historic_CVaR)
            if not np.isnan(historic_CVaR)
            else value_at_risk(returns, level)
        )
    else:
        raise NotImplementedError("Only historic CVaR is currently implemented")
