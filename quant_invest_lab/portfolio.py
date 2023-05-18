from typing import Iterable, Literal, Optional
import pandas as pd
import numpy as np
import numpy.typing as npt
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px


class ABCPortfolio(ABC):
    def __init__(self, returns: pd.DataFrame, trading_days: int = 365) -> None:
        """Construct a new 'ABCPortfolio' object. Generic for all portfolio optimization models : `MonteCarloPortfolio`, `ConvexPortfolio`.

        Args:
        -----
            returns (pd.DataFrame): The returns of the assets in the portfolio.
            trading_days (int, optional): The number of trading days in a year, 365 for cryptos, 252 for stocks. Defaults to 365.
        """
        self._returns = returns
        self._returns_mean = self._returns.mean()
        self._returns_cov = self._returns.cov().values
        self._trading_days = trading_days
        self._already_optimized = False

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_allocation(self):
        pass

    @staticmethod
    def _plot_allocation(
        weights: npt.NDArray | Iterable,
        assets: npt.NDArray | list[str] | tuple[str] | pd.Index,
    ) -> None:
        """Plot the allocation of the portfolio in  a pie chart.

        Args:
        -----
            weights (npt.NDArray | Iterable): The weights for each asset in the portfolio between 0 and 1.
            assets (npt.NDArray | list[str] | tuple[str]): The assets (their names) in the portfolio.
        """
        res = [(p, float(w)) for p, w in zip(assets, weights)]
        res.sort(key=lambda x: x[1], reverse=True)
        res = np.array(res)
        plt.title("Asset allocation")
        plt.pie(res[:, -1], labels=res[:, 0].tolist(), autopct="%1.1f%%")
        plt.show()
        # fig = go.Figure(
        #     go.Pie(
        #         labels=res[:, 0].tolist(),
        #         values=res[:, -1],
        #         hole=0.4,
        #         text=[f"{p}: {float(w):.2f} %" for p, w in res],
        #         textinfo="label+percent",
        #         hoverinfo="label+value",
        #         marker=dict(colors=px.colors.qualitative.Pastel1),
        #     )
        # )

        # fig.update_layout(
        #     title={
        #         "text": "Asset allocation",
        #         "x": 0.5,
        #         "y": 0.95,
        #         "xanchor": "center",
        #         "yanchor": "top",
        #     },
        #     showlegend=False,
        #     height=600,
        #     width=600,
        # )

        # fig.show()

    def _compute_metrics(
        self,
        weights: npt.NDArray,
    ) -> dict[str, float | int]:
        """Given an array of weights, compute the Sharpe ratio, the risk and the return of the portfolio. This method in used it the optimization functions of the portfolio optimization models.

        Args:
        -----
            weights (npt.NDArray): The weights of the assets in the portfolio.

        Returns:
        -----
            dict[str, float | int]: The Sharpe ratio, the risk and the return of the portfolio. The keys are : "sharpe", "risk" and "return".
        """
        ret = np.sum(self._returns_mean * weights * self._trading_days)
        vol = float(
            np.sqrt(weights.T @ self._returns_cov * self._trading_days @ weights)
        )
        sr = ret / vol

        co = pd.DataFrame(self._returns.values * weights).corr()
        np.fill_diagonal(co.values, 1)
        co.fillna(0, inplace=True)
        div = float(linalg.norm(co - np.eye(co.shape[1]), ord="fro"))

        return {"sharpe": sr, "risk": vol, "return": ret, "ptf_correlation": div}


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

        # We calculate the risk of the weights distribution
        portfolio_risk = self._compute_metrics(weights=weights)["risk"]

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
    ) -> None:
        """Optimize the portfolio for an equity risk parity among assets. It uses an optimizer from `scipy.optimize` module. This method returns nothing."""

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
            args=[assets_risk_budget],
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            tol=1e-10,
        )

        self._already_optimized = True
        self.__optimized_weights = opt_results.x
        self.__optimized_metrics = self._compute_metrics(self.__optimized_weights)

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

        print(
            f"- Annualized Sharpe ratio: {self.__optimized_metrics.get('sharpe',0.0):.2f}\n- Annualized risk (volatility): {100*self.__optimized_metrics.get('risk',1.0):.2f} %\n- Annualized expected return: {100*self.__optimized_metrics.get('return',0.0):.2f} %"
        )
        ConvexPortfolio._plot_allocation(
            self.__optimized_weights, self._returns.columns
        )
        return pd.DataFrame(
            {
                p: [w]
                for p, w in zip(
                    self._returns.columns,
                    self.__optimized_weights,
                )
            }
        )


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
        metric: Literal["sharpe", "risk", "return"] = args[0]  # type: ignore
        way: Literal["min", "max"] = args[1]  # type: ignore
        metrics = self._compute_metrics(weights)
        return -metrics.get(metric, 0.0) if way == "max" else metrics.get(metric, 0.0)

    def fit(
        self,
        metric: Literal["sharpe", "risk", "return"] = "sharpe",
        way: Literal["min", "max"] = "max",
        max_asset_weight: float = 0.3,
        min_asset_weight: float = 0.0,
    ) -> None:
        """Optimize the portfolio using the Sharpe ratio, risk or return as a metric and a maximization or a minimization this metric. It uses an optimizer from `scipy.optimize` module. This method returns nothing.

        Args:
        -----
            metric (Literal[&quot;sharpe&quot;, &quot;risk&quot;, &quot;return&quot;], optional): The metric to optimize. Defaults to "sharpe".
            way (Literal[&quot;min&quot;, &quot;max&quot;], optional): The type of wanted optimization optimization. Defaults to "max".
            max_asset_weight (float, optional): The maximal weight of an asset in the portfolio. Defaults to 0.3.
            min_asset_weight (float, optional): The minimal weight of an asset in the portfolio. Defaults to 0.0.
        """
        assert way in [
            "min",
            "max",
        ], "Invalid way of metric evaluation, must be a string equals to min or max."
        assert metric in [
            "sharpe",
            "risk",
            "return",
        ], "Invalid metric, must be a string equals to sharpe, risk or return."
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
        self.__optimized_weights = opt_results.x
        self.__optimized_metrics = self._compute_metrics(self.__optimized_weights)

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

        print(
            f"- Annualized Sharpe ratio: {self.__optimized_metrics.get('sharpe',0.0):.2f}\n- Annualized risk (volatility): {100*self.__optimized_metrics.get('risk',1.0):.2f} %\n- Annualized expected return: {100*self.__optimized_metrics.get('return',0.0):.2f} %"
        )
        ConvexPortfolio._plot_allocation(
            self.__optimized_weights, self._returns.columns
        )
        return pd.DataFrame(
            {
                p: [w]
                for p, w in zip(
                    self._returns.columns,
                    self.__optimized_weights,
                )
            }
        )


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
                "return"
            ]  # np.sum((rets * weights * self._trading_days))

            # Expected volatility
            self.__vol_arr[x] = metrics[
                "risk"
            ]  # np.sqrt( weights.T @ cov_matrix * self._trading_days @ weights)

            # Sharpe Ratio
            self.__sharpe_arr[x] = metrics[
                "sharpe"
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
        MonteCarloPortfolio._plot_allocation(
            self.__all_weights[ind, :], self._returns.columns
        )
        return pd.DataFrame(
            {
                p: [w]
                for p, w in zip(
                    self._returns.columns,
                    self.__all_weights[ind, :],
                )
            }
        )
