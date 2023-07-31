import numpy as np
import numpy.typing as npt
from functools import lru_cache
import pandas as pd
import scipy.stats as stat
from typing import Literal, Union

from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from sklearn.metrics import r2_score


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
    beta, _ = np.polyfit(benchmark_returns, returns, 1)
    return (returns.mean() * N - risk_free_rate) / float(beta)


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
    return (returns.mean() * N - risk_free_rate) / (downside_risk(returns, N))


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


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    N: Union[int, float] = 365,
) -> float:
    """The information ratio (IR) is a measurement of portfolio returns beyond the returns of a benchmark, usually an index, compared to the volatility of those returns. The information ratio (IR) measures a portfolio manager's ability to generate excess returns relative to a benchmark but also attempts to identify the consistency of the investor.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio benchmark not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

    Returns:
    -----
        float: The annualized information ratio.
    """
    return (N * (portfolio_returns - benchmark_returns).mean()) / (
        tracking_error(portfolio_returns, benchmark_returns, N)
    )


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    N: Union[int, float] = 365,
) -> float:
    """Tracking error is the divergence between the price behavior of a position or a portfolio and the price behavior of a benchmark.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio benchmark not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

    Returns:
    -----
        float: The annualized tracking error.
    """
    return (portfolio_returns - benchmark_returns).std() * N**0.5


def downside_risk(
    returns: pd.Series,
    N: Union[int, float] = 365,
) -> float:
    """Downside risk or Semi-Deviation is a method of measuring the fluctuations below the mean, unlike variance or standard deviation it only looks at the negative price fluctuations and it's used to evaluate the downside risk (The risk of loss in an investment) of an investment.

    Args:
    -----
        returns (pd.Series): The strategy or portfolio not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

    Returns:
    ------
        float: The semi-deviation or downside risk of returns.
    """
    return returns.loc[returns < 0].std() * (N**0.5)


def portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """The beta coefficient is a measure of the volatility, or systematic risk, of an individual stock in comparison to the unsystematic risk of the entire market. Beta is used in the capital asset pricing model (CAPM), which describes the relationship between systematic risk and expected return for assets (usually stocks).

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio's benchmark not cumulative returns.

    Returns:
    -----
        float: The beta.
    """
    beta, _ = np.polyfit(benchmark_returns, portfolio_returns, 1)
    return beta


def portfolio_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """The alpha coefficient is an indicator of an investment's performance against a market index or benchmark that is often used by portfolio managers in evaluating the performance of a portfolio or fund manager. The alpha coefficient is also referred to as the "excess return" or "abnormal rate of return," which refers to the idea that markets are efficient, and so there is no way to systematically earn returns that exceed the broad market as a whole.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio's benchmark not cumulative returns.

    Returns:
    -----
        float: The alpha.
    """
    _, alpha = np.polyfit(benchmark_returns, portfolio_returns, 1)
    return alpha


def r_squared(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """R-squared is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio's benchmark not cumulative returns.

    Returns:
    -----
        float: The r-squared or coefficient of determination.
    """
    beta, alpha = np.polyfit(benchmark_returns, portfolio_returns, 1)
    return r2_score(portfolio_returns, beta * benchmark_returns + alpha)  # type: ignore


def systematic_risk(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    N: Union[int, float] = 365,
) -> float:
    """Systematic risk refers to the risk due to general market factors and affects the entire industry. It cannot be diversified away. Here we use only one factor the market beta.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio's benchmark not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

    Returns:
    -----
        float: The annualized systematic risk.
    """
    beta, alpha = np.polyfit(benchmark_returns, portfolio_returns, 1)
    return (
        beta * benchmark_returns + alpha
    ).std() * N**0.5  # beta**2*benchmark_returns.std()**2


def specific_risk(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    N: Union[int, float] = 365,
) -> float:
    """is the risk specific to a company, crypto blockchain, ... that arises due to company-specific characteristics. According to portfolio theory, this risk can be eliminated through diversification.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio's benchmark not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

    Returns:
    -----
        float: The annualized specific risk.
    """
    return (
        OLS(portfolio_returns, add_constant(benchmark_returns)).fit().resid.std()
        * N**0.5
    )


def jensen_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    N: Union[int, float] = 365,
) -> float:
    """The Jensen index or Jensen's alpha is a risk-adjusted performance benchmark that represents the average return on a portfolio or investment above or below that predicted by the capital asset pricing model (CAPM) given the portfolio's (or investment's) beta and the average market return.

    Args:
    -----
        portfolio_returns (pd.Series): The strategy or portfolio not cumulative returns.

        benchmark_returns (pd.Series): The strategy or portfolio's benchmark not cumulative returns.

        N (Union[int, float], optional): The number of periods in a year. Defaults to 365.

    Returns:
    -----
        float: The annualized Jensen alpha.
    """
    return (
        OLS(portfolio_returns, add_constant(benchmark_returns)).fit().resid.mean() * N
    )


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


def kelly_criterion(returns: pd.Series) -> float:
    """The Kelly criterion is a mathematical formula relating to the long-term growth of capital developed by John L. Kelly Jr. The formula was developed by Kelly while working at the AT&T Bell Laboratories. The Kelly Criterion is one of the most important concepts in money management and betting. Its simplicity and power are the reasons behind its popularity.

    Args:
    ----
        returns (pd.Series): returns (pd.Series): The strategy or portfolio not cumulative returns.

    Returns:
    -----
        float: The kelly criterion.
    """
    ret = returns[returns != 0]
    p = (ret > 0).mean()
    r = ret[ret > 0].mean() / abs(ret[ret < 0].mean())
    return float((p * r - (1 - p)) / r)


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
