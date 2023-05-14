import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Callable
from abc import abstractmethod
from scipy.optimize import minimize
from quant_invest_lab.backtest import ohlc_long_only_backtester


class ConditionOptimizer:
    def __init__(
        self,
        long_entry_function: Callable[[pd.Series, pd.Series], bool],
        long_exit_function: Callable[[pd.Series, pd.Series, int], bool],
    ) -> None:
        self._long_exit_function: Callable[
            [pd.Series, pd.Series, int], bool
        ] = long_exit_function
        self._long_entry_function: Callable[
            [pd.Series, pd.Series], bool
        ] = long_entry_function

    def _check_range_value(self, value_range: tuple[float | int, float | int]) -> None:
        assert (
            len(value_range) == 2
        ), "range must be a integer or floating tuple of length 2"
        assert value_range[0] < value_range[1], "range must be ascending"
        assert (
            value_range[0] > 0
            and value_range[0] < 1
            and value_range[1] > 0
            and value_range[1] < 1
        ), "range must be positive and between 0 and 1 as a percentage."

    @abstractmethod
    def _compute_return():
        pass

    @abstractmethod
    def optimize():
        pass


class TakeProfitOptimizer(ConditionOptimizer):
    def _compute_return(
        self,
        take_profit: npt.NDArray[np.float64],
        args: list[pd.DataFrame],
    ) -> float:
        return -ohlc_long_only_backtester(
            args[0],
            long_entry_function=self._long_entry_function,
            long_exit_function=self._long_exit_function,
            take_profit=take_profit[0],
            parameter_optimization=True,
        )  # type: ignore

    def optimize(
        self,
        dataframe: pd.DataFrame,
        take_profit_bounds: tuple[float | int, float | int] = (
            0.001,
            0.3,
        ),
    ) -> tuple[float | int, float | int]:
        """Optimize take profit values using `scipy.optimize.minimize` and a SLSQP method.

        Args:
        --------
            dataframe (pd.DataFrame): The dataframe containing the OHLCV data to optimize on the strategy with a given take profit.
            take_profit_bounds (tuple[float  |  int, float  |  int], optional): The tuple of boundaries for the TP between 0.001 and 1. Defaults to (0.001, 0.3)

        Returns:
        --------
            tuple[float | int, float | int]: The result from the optimization process under the form of a tuple containing the optimized take profile values.
        """
        self._check_range_value(take_profit_bounds)

        results = [
            minimize(
                self._compute_return,
                x0=[guess],
                method="SLSQP",  # "SLSQP", #L-BFGS-B
                bounds=tuple(
                    [
                        take_profit_bounds,
                    ]
                ),
                args=[dataframe],
            )
            for guess in np.linspace(
                take_profit_bounds[0],
                take_profit_bounds[-1],
                10,  # Number of init guess to generate
            )
        ]

        results.sort(key=lambda x: x.fun)
        opt_results = results[0]
        return opt_results.x[0]


class StopLossOptimizer(ConditionOptimizer):
    def _compute_return(
        self,
        stop_loss: npt.NDArray[np.float64],
        args: list[pd.DataFrame],
    ) -> float:
        return -ohlc_long_only_backtester(
            args[0],
            long_entry_function=self._long_entry_function,
            long_exit_function=self._long_exit_function,
            stop_loss=stop_loss[0],
            parameter_optimization=True,
        )  # type: ignore

    def optimize(
        self,
        dataframe: pd.DataFrame,
        stop_loss_bounds: tuple[float | int, float | int] = (
            0.001,
            0.3,
        ),
    ) -> tuple[float | int, float | int]:
        """Optimize stop loss values using `scipy.optimize.minimize` and a SLSQP method.

        Args:
        --------
            dataframe (pd.DataFrame): The dataframe containing the OHLCV data to optimize on the strategy with a given stop loss.
            take_profit_and_stop_loss_bounds (tuple[float  |  int, float  |  int], optional): The tuple of boundaries for the SL between 0.001 and 1. Defaults to (0.001, 0.3)

        Returns:
        --------
            tuple[float | int, float | int]: The result from the optimization process under the form of a tuple containing the optimized stop loss values.
        """
        self._check_range_value(stop_loss_bounds)

        results = [
            minimize(
                self._compute_return,
                x0=[guess],
                method="SLSQP",  # "SLSQP", #L-BFGS-B
                bounds=tuple(
                    [
                        stop_loss_bounds,
                    ]
                ),
                args=[dataframe],
            )
            for guess in np.linspace(
                stop_loss_bounds[0],
                stop_loss_bounds[-1],
                5,  # Number of init guess to generate
            )
        ]

        results.sort(key=lambda x: x.fun)
        opt_results = results[0]
        return opt_results.x[0]


class StopLossTakeProfitOptimizer(ConditionOptimizer):
    def _compute_return(
        self,
        take_profit_and_stop_loss: npt.NDArray[np.float64],
        args: list[pd.DataFrame],
    ) -> float:
        return -ohlc_long_only_backtester(
            args[0],
            long_entry_function=self._long_entry_function,
            long_exit_function=self._long_exit_function,
            take_profit=take_profit_and_stop_loss[0],
            stop_loss=take_profit_and_stop_loss[-1],
            parameter_optimization=True,
        )  # type: ignore

    def optimize(
        self,
        dataframe: pd.DataFrame,
        take_profit_and_stop_loss_bounds: tuple[float | int, float | int] = (
            0.001,
            0.3,
        ),
    ) -> tuple[float | int, float | int]:
        """Optimize take profit and stop loss values using `scipy.optimize.minimize` and a SLSQP method.

        Args:
        --------
            dataframe (pd.DataFrame): The dataframe containing the OHLCV data to optimize on the strategy with a given take profit and stop loss.
            take_profit_and_stop_loss_bounds (tuple[float  |  int, float  |  int], optional): The tuple of boundaries for the TP and SL between 0.001 and 1. Defaults to (0.001, 0.3)

        Returns:
        --------
            tuple[float | int, float | int]: The result from the optimization process under the form of a tuple containing the optimized (in this order) take profit and stop loss values.
        """
        self._check_range_value(take_profit_and_stop_loss_bounds)

        results = [
            minimize(
                self._compute_return,
                x0=[guess, guess],
                method="SLSQP",  # "SLSQP", #L-BFGS-B
                bounds=tuple(
                    [
                        take_profit_and_stop_loss_bounds,
                        take_profit_and_stop_loss_bounds,
                    ]
                ),
                args=[dataframe],
            )
            for guess in np.linspace(
                take_profit_and_stop_loss_bounds[0],
                take_profit_and_stop_loss_bounds[-1],
                10,  # Number of init guess to generate
            )
        ]

        results.sort(key=lambda x: x.fun)
        opt_results = results[0]
        return opt_results.x[0], opt_results.x[1]
