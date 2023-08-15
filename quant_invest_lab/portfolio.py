from typing import Iterable, Literal, Optional
import pandas as pd
import numpy as np
import numpy.typing as npt
import numpy.linalg as linalg
import plotly.graph_objects as go
from tqdm import tqdm
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from quant_invest_lab.constants import PORTFOLIO_METRICS, TIMEFRAME_ANNUALIZED

from quant_invest_lab.reports import (
    construct_report_dataframe,
    plot_from_trade_df_and_ptf_optimization,
    print_portfolio_strategy_report,
)
from quant_invest_lab.types import PortfolioMetric, Timeframe


class ABCPortfolio(ABC):
    def __init__(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        timeframe: Timeframe = "1hour",
    ) -> None:
        """Construct a new 'ABCPortfolio' object. Generic for all portfolio optimization models : `MonteCarloPortfolio`, `ConvexPortfolio`.

        Args:
        -----
            returns (pd.DataFrame): The returns of the assets in the portfolio.

            benchmark_returns (pd.Series): The returns of the portfolio's benchmark.

            timeframe (Timeframe, optional): The timeframe granularity of the returns and the benchmark_returns, it must be the same. Defaults to "1hour".
        """
        self._returns = returns
        self._benchmark_returns = benchmark_returns
        self._returns_mean = self._returns.mean()
        self._returns_cov = self._returns.cov().values
        self._timeframe = timeframe
        self._trading_days = TIMEFRAME_ANNUALIZED[self._timeframe]
        self._already_optimized = False
        self._optimized_weights = None

    @abstractmethod
    def fit(self):
        pass

    def get_allocation(self) -> pd.DataFrame:
        """Plot the allocation of the portfolio and return a DataFrame with the allocation of each asset. This function can't be called before the fit method.

        Returns:
        --------
            pd.DataFrame: The allocation of each asset in the portfolio, columns are the assets and rows are the weights.
        """
        assert (
            self._already_optimized
        ), "You must fit the model before getting the allocation."

        print(f"{'  Results  ':-^40}")

        print_portfolio_strategy_report(
            self._returns.apply(
                lambda row_returns: row_returns @ self._optimized_weights, axis=1
            ),
            self._benchmark_returns,
            self._timeframe,
        )
        allocation_dataframe = pd.DataFrame(
            {
                p: [w]
                for p, w in zip(
                    self._returns.columns,
                    self._optimized_weights,
                )
            }
        )
        plot_from_trade_df_and_ptf_optimization(
            self._returns.apply(
                lambda row_returns: row_returns @ self._optimized_weights, axis=1
            ),
            self._benchmark_returns,
            allocation_dataframe,
        )
        return allocation_dataframe

    def _compute_metrics(
        self,
        weights: npt.NDArray,
    ) -> dict[PortfolioMetric, float | int]:
        """Given an array of weights, compute the Sharpe ratio, the risk and the return of the portfolio. This method in used it the optimization functions of the portfolio optimization models.

        Args:
        -----
            weights (npt.NDArray): The weights of the assets in the portfolio.

        Returns:
        -----
            dict[PortfolioMetric, float | int]: The ratios, the risk, the return all the metrics related to the portfolio.
        """
        # ret = float(np.sum(self._returns_mean * weights * self._trading_days))  # type: ignore
        rets = self._returns.apply(lambda row_returns: row_returns @ weights, axis=1)
        report_df = construct_report_dataframe(rets, self._benchmark_returns)

        co = pd.DataFrame(self._returns.values * weights).corr()
        np.fill_diagonal(co.values, 1)
        co.fillna(0, inplace=True)
        div = float(linalg.norm(co - np.eye(co.shape[1]), ord="fro"))

        return report_df["Portfolio"].T.to_dict()


class RiskParityPortfolio(ABCPortfolio):
    def __compute_objective_metrics(self, weights: npt.NDArray, args: list) -> float:
        """The objective function to optimize. It is used in the `fit` method.

        Args:
        -----
            weights (npt.NDArray): The weights of the assets in the portfolio.

            args (list): The arguments of the objective function.

        Returns:
        -----
            float: The value of the objective function.
        """

        # We convert the weights to a matrix
        weights_matrix = np.matrix(weights)
        assets_risk_budget = np.matrix(args[0])

        if args[1] != "Expected volatility":
            raise NotImplementedError(
                "Only 'Expected volatility' is implemented for now."
            )

        # We calculate the risk of the weights distribution
        portfolio_risk = self._compute_metrics(weights=weights)[args[1]]

        # We calculate the contribution of each asset to the risk of the weights
        # distribution
        assets_risk_contribution = (
            np.multiply(
                weights_matrix.T,
                self._returns_cov * self._trading_days * weights_matrix.T,
            )
            / portfolio_risk
        )

        # We calculate the desired contribution of each asset to the risk of the weights distribution
        assets_risk_target = portfolio_risk * assets_risk_budget

        # Error between the desired contribution and the calculated contribution of each asset
        return np.sum(np.square(assets_risk_contribution - assets_risk_target.T))  # MSE

    def fit(
        self,
        parity_risk_metric: Literal[
            "Expected volatility",
            "VaR",
            "CVaR",
            "Specific risk",
            "Portfolio beta",
            "Tracking error",
        ] = "Expected volatility",
    ) -> None:
        """Optimize the portfolio for an equity risk parity among assets. It uses an optimizer from `scipy.optimize` module. This method returns nothing.

        Args:
            parity_risk_metric (Literal[ &quot;Expected volatility&quot;, &quot;VaR&quot;, &quot;CVaR&quot;, &quot;Specific risk&quot;, &quot;Portfolio beta&quot;, &quot;Tracking error&quot;, ]): The metric to use to compute the risk parity among assets. It can be one of the following: `Expected volatility`, `VaR`, `CVaR`, `Specific risk`, `Portfolio beta`, `Tracking error`. Default to `Expected volatility`.

        """
        cons = (
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1,
            },  # return 0 if sum of the weights is 1
            {"type": "ineq", "fun": lambda x: x},  # Long only
        )

        bounds = tuple([(0.0, 1.0) for _ in range(len(self._returns.columns))])
        init_guess = [
            1 / len(self._returns.columns) for _ in range(len(self._returns.columns))
        ]

        assets_risk_budget = [
            1 / len(self._returns.columns) for _ in range(len(self._returns.columns))
        ]

        opt_results = minimize(
            fun=self.__compute_objective_metrics,
            x0=init_guess,
            args=[assets_risk_budget, parity_risk_metric],
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            tol=1e-10,
        )

        self._already_optimized = True
        self._optimized_weights = opt_results.x


class ConvexPortfolio(ABCPortfolio):
    def __compute_objective_metrics(
        self, weights: npt.NDArray, args: list[str]
    ) -> float:
        """The objective function to optimize. It is used in the `fit` method.

        Args:
        -----
            weights (npt.NDArray): The weights of the assets in the portfolio.

            args (list[str]): The arguments of the objective function.

        Returns:
        -----
            float: The value of the objective function.
        """
        metric: PortfolioMetric = args[0]  # type: ignore
        way: Literal["min", "max"] = args[1]  # type: ignore
        metrics = self._compute_metrics(weights)
        return -metrics.get(metric, 0.0) if way == "max" else metrics.get(metric, 0.0)

    def fit(
        self,
        metric: PortfolioMetric = "Sharpe ratio",
        way: Literal["min", "max"] = "max",
        max_asset_weight: float = 0.3,
        min_asset_weight: float = 0.0,
    ) -> None:
        """Optimize the portfolio using the Sharpe ratio, risk or return as a metric and a maximization or a minimization this metric. It uses an optimizer from `scipy.optimize` module. This method returns nothing.

        Args:
        -----
            metric (Literal[ &quot;Expected return&quot;, &quot;Expected volatility&quot;, &quot;Skewness&quot;, &quot;Kurtosis&quot;, &=quot;VaR&quot;, &quot;CVaR&quot;, &quot;Max drawdown&quot;, &quot;Kelly criterion&quot;, &quot;Sharpe ratio&quot;, &quot;Sortino ratio&quot;, &quot;Burke ratio&quot;, &quot;Calmar ratio&quot;, &quot;Tail ratio&quot;, &quot;Specific risk&quot;, &quot;Systematic risk&quot;, &quot;Portfolio beta&quot;, &quot;Portfolio alpha&quot;, &quot;Jensen alpha&quot;, &quot;R2&quot;, &quot;Tracking error&quot;, &quot;Treynor ratio&quot;, &quot;Information ratio&quot;, ], optional): The metric to optimize. Defaults to "Sharpe ratio".

            way (Literal[&quot;min&quot;, &quot;max&quot;], optional): The type of wanted optimization optimization. Defaults to "max".

            max_asset_weight (float, optional): The maximal weight of an asset in the portfolio. Defaults to 0.3.

            min_asset_weight (float, optional): The minimal weight of an asset in the portfolio. Defaults to 0.0.
        """
        assert way in [
            "min",
            "max",
        ], "Invalid way of metric evaluation, must be a string equals to min or max."
        assert (
            metric in PORTFOLIO_METRICS
        ), f'Invalid metric, must be a string equals to {",".join(PORTFOLIO_METRICS)}.'
        assert max_asset_weight <= 1.0, "Max asset weight must be less or equal to 1.0"
        assert (
            min_asset_weight >= 0.0
        ), "Min asset weight must be greater or equal to 0.0"
        assert (
            max_asset_weight >= min_asset_weight
        ), "Max asset weight must be greater or equal to min asset weight."

        cons = (
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1,
            },  # return 0 if sum of the weights is 1
            {"type": "ineq", "fun": lambda x: x},  # Long only
        )

        bounds = tuple(
            [
                (min_asset_weight, max_asset_weight)
                for _ in range(len(self._returns.columns))
            ]
        )
        init_guess = [
            1 / len(self._returns.columns) for _ in range(len(self._returns.columns))
        ]

        opt_results = minimize(
            fun=self.__compute_objective_metrics,
            x0=init_guess,
            args=[metric, way],
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            tol=1e-10,
        )

        self._already_optimized = True
        self._optimized_weights = opt_results.x


class MonteCarloPortfolio(ABCPortfolio):
    def fit(self, n_portfolios: int = 20000, plot: bool = True):
        """This method run a Monte Carlo simulation to allocate and create many portfolios in order to then find the most efficient. This method returns nothing.

        Args:
        -----
            n_portfolios (int, optional): The number of portefolio to create. Defaults to 20000.

            trading_days (int, optional): The number of trading days in a year, 365 for cryptos, 252 for stocks. Defaults to 365.

            plot (bool, optional): Whether or not to plot the portfolios simulated on a chart, x-axis = risk, y-axis = return. Defaults to True.
        """
        self.__all_weights = np.zeros((n_portfolios, len(self._returns.columns)))
        self.__ret_arr = np.zeros(n_portfolios)
        self.__vol_arr = np.zeros(n_portfolios)
        self.__sharpe_arr = np.zeros(n_portfolios)

        for x in tqdm(range(n_portfolios)):
            weights = np.array(
                np.random.rand(len(self._returns.columns)), dtype=np.float64
            )
            weights = weights / np.sum(weights)

            # Save weights
            self.__all_weights[x, :] = weights

            metrics = self._compute_metrics(weights)

            # Expected return
            self.__ret_arr[x] = metrics[
                "Expected return"
            ]  # np.sum((rets * weights * self._trading_days))

            # Expected volatility
            self.__vol_arr[x] = metrics[
                "Expected volatility"
            ]  # np.sqrt( weights.T @ cov_matrix * self._trading_days @ weights)

            # Sharpe Ratio
            self.__sharpe_arr[x] = metrics[
                "Sharpe ratio"
            ]  # self.__ret_arr[x] / self.__vol_arr[x]
        self._already_optimized = True

        if plot:
            self.plot_portfolios(n_portfolios)

    def plot_portfolios(self, n_portfolios: Optional[int] = None) -> None:
        """Plot the portfolios simulated on a chart, x-axis = risk, y-axis = return.

        Args:
        -----
            n_portfolios (Optional[int], optional): The number of portfolio. Defaults to None.

        """
        assert (
            self._already_optimized
        ), "You must fit the model before getting the allocation."

        fig = go.Figure(
            go.Scatter(
                x=self.__vol_arr,
                y=self.__ret_arr,
                mode="markers",
                marker=dict(
                    color=self.__sharpe_arr,
                    colorbar=dict(title="Sharpe Ratio"),
                    colorscale="viridis",
                ),
                text=[
                    f"Volatility: {vol:.2f}<br>Return: {ret:.2f}<br>Sharpe Ratio: {sharpe:.2f}"
                    for vol, ret, sharpe in zip(
                        self.__vol_arr, self.__ret_arr, self.__sharpe_arr
                    )
                ],
            )
        )

        fig.update_layout(
            title={
                "text": f"Monte Carlo Portfolio Optimization, {n_portfolios} simulations"
                if n_portfolios
                else "Monte Carlo Portfolio Optimization",
                "x": 0.5,
                "y": 0.95,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis_title="Volatility",
            yaxis_title="Return",
            hovermode="closest",
            showlegend=False,
            height=650,
        )

        fig.add_trace(
            go.Scatter(
                x=[self.__vol_arr[self.__sharpe_arr.argmax()]],
                y=[self.__ret_arr[self.__sharpe_arr.argmax()]],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Maximum Sharpe Ratio",
            )
        )

        fig.show()

    def get_allocation(
        self,
        metric: Literal["sharpe", "risk", "return"] = "sharpe",
        way: Literal["min", "max"] = "max",
    ) -> pd.DataFrame:
        """Plot the allocation of the portfolio and return a DataFrame with the allocation of each asset. This function can't be called before the fit method.

        Args:
        -----
            metric (Literal[&quot;sharpe&quot;, &quot;risk&quot;, &quot;return&quot;], optional): The metric to optimize. Defaults to "sharpe".
            way (Literal[&quot;min&quot;, &quot;max&quot;], optional): The type of wanted optimization optimization. Defaults to "max".

        Raises:
        -------
            Exception: Invalid metric chosen.

        Returns:
        --------
            pd.DataFrame: The allocation of each asset in the portfolio, columns are the assets and rows are the weights.
        """
        assert (
            self._already_optimized
        ), "You must fit the model before getting the allocation."
        assert way in [
            "min",
            "max",
        ], "Invalid way of metric evaluation, must be a string equals to min or max."

        match metric:
            case "sharpe":
                ind = (
                    self.__sharpe_arr.argmin()
                    if way == "min"
                    else self.__sharpe_arr.argmax()
                )
            case "risk":
                ind = (
                    self.__vol_arr.argmin() if way == "min" else self.__vol_arr.argmax()
                )
            case "return":
                ind = (
                    self.__vol_arr.argmin() if way == "min" else self.__vol_arr.argmax()
                )
            case _:
                raise Exception("Invalid metric.")

        print(f"{'  Results  ':-^40}")
        print(
            f"- Annualized Sharpe ratio: {self.__sharpe_arr[ind]:.2f}\n- Annualized risk (volatility): {100*self.__vol_arr[ind]:.2f} %\n- Annualized expected return: {100*self.__ret_arr[ind]:.2f} %"
        )
        alloc_df = pd.DataFrame(
            {
                p: [w]
                for p, w in zip(
                    self._returns.columns,
                    self.__all_weights[ind, :],
                )
            }
        )
        plot_from_trade_df_and_ptf_optimization(
            self._returns.apply(
                lambda row_returns: row_returns @ self._optimized_weights, axis=1
            ),
            self._benchmark_returns,
            alloc_df,
        )
        return alloc_df
