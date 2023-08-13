from datetime import datetime
from typing import Callable, Literal, Optional, Union, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from quant_invest_lab.metrics import (
    cumulative_returns,
    drawdown,
)
from quant_invest_lab.constants import TIMEFRAMES
from quant_invest_lab.types import Timeframe
from quant_invest_lab.reports import print_ohlc_backtest_report, plot_from_trade_df


def __ohlc_backtest_one_position_type(
    ohlcv_df: pd.DataFrame,
    entry_function: Callable[[pd.Series, pd.Series], bool],
    exit_function: Callable[[pd.Series, pd.Series, int], bool],
    position_type: Literal["long", "short"],
    take_profit: float = np.inf,
    stop_loss: float = np.inf,
    maker_fees: Optional[float] = 0.001,
    taker_fees: Optional[float] = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Core backtesting function for OHLCV data. It iterate over the OHLCV records and run the entry and exit function when the conditions are satisfied.
    WARNING this function shouldn't be used alone, use the long/short backtest function instead.

    Args:
    ----
        ohlcv_df (pd.DataFrame): The dataframe containing the OHLC data.

        entry_function (Callable[[pd.Series, pd.Series], bool]): The trade entry function, it should take 2 arguments, the current row and the previous row and return True or False depending on your strategy.

        exit_function (Callable[[pd.Series, pd.Series, int], bool]): The long entry function, it should take 2 arguments, the current row and the previous row and return True or False depending on your strategy.

        position_type (Literal[&quot;long&quot;, &quot;short&quot;]): The position type, long or short.

        take_profit (float, optional): The percent of the buy price to add to create a stop order and take the profit associated. Defaults to np.inf.

        stop_loss (float, optional): The percent of the buy price to cut to create a stop order and stop the loss associated. Defaults to np.inf.

        maker_fees (Optional[float], optional): The fees applied, here maker for limit orders (not yet implemented). Defaults to 0.001.

        taker_fees (Optional[float], optional): The fees applied, here taker for spot orders. Defaults to 0.001.

    Returns:
    -----
        tuple[pd.DataFrame, pd.DataFrame]: It returns 2 dataframes, the first is the trades_df : trade summary dataframe and the second is the OHMCV dataframe that contains a new column Strategy returns.
    """
    assert position_type in ["long", "short"], "position_type must be long or short"

    RETURNS_SIGNS = {"long": 1, "short": -1}
    previous_row = ohlcv_df.iloc[0]
    position_opened = False
    timeframe_count = 0
    current_trade: dict[str, float | datetime | int | str | pd.Timestamp] = {}

    trades_df = pd.DataFrame(
        columns=[
            "entry_date",
            "entry_price",
            "entry_reason",
            "exit_date",
            "exit_price",
            "exit_reason",
            "trade_return",
        ]
    )
    ohlcv_df["Strategy_returns"] = 0

    for row in tqdm(
        ohlcv_df[1:-1].itertuples(index=True),
        desc="Backtesting...",
        total=ohlcv_df.shape[0] - 1,
        leave=False,
    ):
        if position_opened is False and entry_function(row, previous_row) is True:
            position_opened = True
            current_trade["entry_date"] = row.Index
            current_trade["entry_price"] = row.Close
            current_trade["entry_reason"] = "Entry position triggered"
            timeframe_count = 1
            entry_price = row.Close
        elif (
            position_type == "long"
            and position_opened is True
            and current_trade["entry_price"] * (1 + take_profit) <= row.High
        ) or (
            position_type == "short"
            and position_opened is True
            and current_trade["entry_price"] * (1 + stop_loss) <= row.High
        ):
            position_opened = False
            current_trade["exit_date"] = row.Index

            current_trade["exit_price"] = (
                current_trade["entry_price"] * (1 + take_profit)
                if position_type == "long"
                else current_trade["entry_price"] * (1 + stop_loss)
            )

            current_trade["exit_reason"] = (
                "Long take profit triggered"
                if position_type == "long"
                else "Short stop loss triggered"
            )

            rets = (
                RETURNS_SIGNS[position_type]
                * ohlcv_df.loc[
                    current_trade["entry_date"] : current_trade["exit_date"]
                ].Returns
            )
            if isinstance(taker_fees, float) and taker_fees > 0:
                rets.iloc[0] = rets.iloc[0] - taker_fees
                rets.iloc[-1] = rets.iloc[-1] - taker_fees

            current_trade["trade_return"] = ((rets + 1).cumprod().iloc[-1]) - 1  # ret

            trades_df = pd.concat(
                [trades_df, pd.DataFrame([current_trade])], ignore_index=True
            )
            ohlcv_df.loc[rets.index[0] : rets.index[-1], "Strategy_returns"] = rets
            timeframe_count = 0

            current_trade = {}
        elif (
            position_type == "long"
            and position_opened is True
            and current_trade["entry_price"] * (1 - stop_loss) >= row.Low
        ) or (
            position_type == "short"
            and position_opened is True
            and current_trade["entry_price"] * (1 - take_profit) >= row.Low
        ):
            position_opened = False
            current_trade["exit_date"] = row.Index

            current_trade["exit_price"] = (
                current_trade["entry_price"] * (1 - stop_loss)
                if position_type == "long"
                else current_trade["entry_price"] * (1 - take_profit)
            )
            current_trade["exit_reason"] = (
                "Long stop loss triggered"
                if position_type == "long"
                else "Short take profit triggered"
            )

            rets = (
                RETURNS_SIGNS[position_type]
                * ohlcv_df.loc[
                    current_trade["entry_date"] : current_trade["exit_date"]
                ].Returns
            )
            if isinstance(taker_fees, float) and taker_fees > 0:
                rets.iloc[0] = rets.iloc[0] - taker_fees
                rets.iloc[-1] = rets.iloc[-1] - taker_fees
            current_trade["trade_return"] = ((rets + 1).cumprod().iloc[-1]) - 1  # ret

            trades_df = pd.concat(
                [trades_df, pd.DataFrame([current_trade])], ignore_index=True
            )
            ohlcv_df.loc[rets.index[0] : rets.index[-1], "Strategy_returns"] = rets
            timeframe_count = 0
            current_trade = {}
        elif (
            position_opened is True
            and exit_function(row, previous_row, timeframe_count) is True
        ):
            next_row_index = (
                ohlcv_df.index.get_loc(row.Index) + 1
            )  # To close on next open.

            position_opened = False
            current_trade["exit_date"] = ohlcv_df.iloc[next_row_index].name
            current_trade["exit_price"] = ohlcv_df.iloc[next_row_index].Open
            current_trade["exit_reason"] = "Exit position triggered"

            rets = (
                RETURNS_SIGNS[position_type]
                * ohlcv_df.loc[
                    current_trade["entry_date"] : current_trade["exit_date"]
                ].Returns
            )

            if isinstance(taker_fees, float) and taker_fees > 0:
                rets.iloc[0] = rets.iloc[0] - taker_fees
                rets.iloc[-1] = rets.iloc[-1] - taker_fees

            current_trade["trade_return"] = ((rets + 1).cumprod().iloc[-1]) - 1  # ret

            trades_df = pd.concat(
                [trades_df, pd.DataFrame([current_trade])], ignore_index=True
            )
            ohlcv_df.loc[rets.index[0] : rets.index[-1], "Strategy_returns"] = rets

            timeframe_count = 0
            current_trade = {}
        else:
            timeframe_count += 1
        previous_row = row
    # Close last trade
    if position_opened is True:
        # print("Closing trade at end of backtest")
        position_opened = False
        current_trade["exit_date"] = ohlcv_df.index[-1]  # type: ignore
        current_trade["exit_price"] = ohlcv_df.iloc[-1].Close
        current_trade["exit_reason"] = "Exit position triggered"

        rets = (
            RETURNS_SIGNS[position_type]
            * ohlcv_df.loc[
                current_trade["entry_date"] : current_trade["exit_date"]
            ].Returns
        )

        if isinstance(taker_fees, float) and taker_fees > 0:
            rets.iloc[0] = rets.iloc[0] - taker_fees
            rets.iloc[-1] = rets.iloc[-1] - taker_fees

        current_trade["trade_return"] = ((rets + 1).cumprod().iloc[-1]) - 1  # ret

        trades_df = pd.concat(
            [trades_df, pd.DataFrame([current_trade])], ignore_index=True
        )
        ohlcv_df.loc[rets.index[0] : rets.index[-1], "Strategy_returns"] = rets

        timeframe_count = 0
        current_trade = {}
    return trades_df, ohlcv_df


def ohlc_long_only_backtester(
    df: pd.DataFrame,
    long_entry_function: Callable[[pd.Series, pd.Series], bool],
    long_exit_function: Callable[[pd.Series, pd.Series, int], bool],
    timeframe: Timeframe,
    take_profit: float = np.inf,
    stop_loss: float = np.inf,
    initial_equity: int = 1000,
    maker_fees: Optional[float] = 0.001,
    taker_fees: Optional[float] = 0.001,
    get_trade_df: bool = False,
    get_returns_df: bool = False,
    parameter_optimization: bool = False,
    plot_result: bool = True,
) -> Union[None, Union[float, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]]:
    """Run a backtest with long only position on a OHLC dataset.

    Args:
    ----
        df (pd.DataFrame): The dataframe containing the OHLC data.

        long_entry_function (Callable[[pd.Series, pd.Series], bool]): The long entry function, it should take 2 arguments, the current row and the previous row and return True or False depending on your strategy.

        long_exit_function (Callable[[pd.Series, pd.Series, int], bool]): The long exit function, it should take 3 arguments, the current row, the previous row and the number of timeframe count since the last entry order, and return True or False depending on your strategy.

        timeframe (Timeframe): The timeframe granularity of the dataframe.

        take_profit (float, optional): The percent of the buy price to add to create a stop order and take the profit associated. Defaults to np.inf.

        stop_loss (float, optional): The percent of the buy price to cut to create a stop order and stop the loss associated. Defaults to np.inf.

        initial_equity (int, optional): The initial capital. Defaults to 1000.

        maker_fees (Optional[float], optional): The fees applied, here maker for limit orders (not yet implemented). Defaults to 0.001.

        taker_fees (Optional[float], optional): The fees applied, here taker for spot orders. Defaults to 0.001.

        get_trade_df (bool, optional): Whether or not to return the trade dataframe (summary of trades) . Defaults to False.

        get_returns_df (bool, optional): Whether or not to return the strategy returns dataframe. Defaults to False.

        parameter_optimization (bool, optional): This parameter is useful when running fitting/optimization it prints nothing but the final strategy return. Defaults to False.

        plot_result (bool, optional): Plot equity, price, drawdown, distribution of the strategy. Defaults to True.

    Returns:
    -----
        Union[None, Union[float, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]]: Nothing or the strategy total return or the trade_df or the return_df or the trade_df and the return_df.
    """
    assert (
        timeframe in TIMEFRAMES
    ), f"timeframe must be one of the following: {','.join(TIMEFRAMES)}"
    assert df.shape[0] > 1, "Dataframe must have at least 2 rows"
    assert set({"Open", "High", "Low", "Close", "Returns"}).issubset(
        df.columns
    ), "Dataframe must have columns Open, High, Low, Close, Returns"

    ohlcv_df = df.copy()
    trades_df, ohlcv_df = __ohlc_backtest_one_position_type(
        ohlcv_df,
        long_entry_function,
        long_exit_function,
        position_type="long",
        take_profit=take_profit,
        stop_loss=stop_loss,
        maker_fees=maker_fees,
        taker_fees=taker_fees,
    )

    ohlcv_df["Strategy_cum_returns"] = cumulative_returns(ohlcv_df["Strategy_returns"])

    if parameter_optimization is True:
        if len(trades_df) > 0:
            return ohlcv_df["Strategy_cum_returns"].iloc[-1]
        return 0.0

    assert len(trades_df) > 0, "No trades were generated"

    ohlcv_df["Strategy_drawdown"] = drawdown(ohlcv_df["Strategy_returns"])
    ohlcv_df["Cum_returns"] = cumulative_returns(ohlcv_df["Returns"])
    ohlcv_df["Drawdown"] = drawdown(ohlcv_df["Returns"])
    trades_df["trade_duration"] = trades_df["exit_date"] - trades_df["entry_date"]

    print(f"{'  Initial informations  ':-^50}")
    print(f"Period: [{str(ohlcv_df.index[0])}] -> [{str(ohlcv_df.index[-1])}]")
    print(f"Intial balance: {initial_equity:.2f} $")
    if taker_fees is not None and maker_fees is not None:
        print(
            f"Taker fees: {taker_fees*100:.2f} %, Maker fees: {maker_fees*100:.2f} %, All the metrics will be calculated considering these fees"
        )
    else:
        print("No fees considered here.")
    if take_profit != np.inf:
        print(f"Take profit is set to {take_profit*100:.2f} % of the buy price")
    if stop_loss != np.inf:
        print(f"Stop loss is set to {stop_loss*100:.2f} % of the buy price")
    print("Long only position")

    print_ohlc_backtest_report(
        trades_df=trades_df,
        ohlcv_df=ohlcv_df,
        timeframe=timeframe,
        initial_equity=initial_equity,
    )

    if plot_result is True:
        plot_from_trade_df(ohlcv_df)
    if get_returns_df and get_trade_df:
        return trades_df, ohlcv_df
    if get_trade_df:
        return trades_df
    if get_returns_df:
        return ohlcv_df


def ohlc_short_only_backtester(
    df: pd.DataFrame,
    short_entry_function: Callable[[pd.Series, pd.Series], bool],
    short_exit_function: Callable[[pd.Series, pd.Series, int], bool],
    timeframe: Timeframe,
    take_profit: float = np.inf,
    stop_loss: float = np.inf,
    initial_equity: int = 1000,
    maker_fees: Optional[float] = 0.001,
    taker_fees: Optional[float] = 0.001,
    get_trade_df: bool = False,
    get_returns_df: bool = False,
    parameter_optimization: bool = False,
    plot_result: bool = True,
) -> Union[None, Union[float, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]]:
    """Run a backtest with short only position on a OHLC dataset.

    Args:
    ----
        df (pd.DataFrame): The dataframe containing the OHLC data.

        short_entry_function (Callable[[pd.Series, pd.Series], bool]): The short entry function, it should take 2 arguments, the current row and the previous row and return True or False depending on your strategy.

        short_exit_function (Callable[[pd.Series, pd.Series, int], bool]): The short exit function, it should take 3 arguments, the current row, the previous row and the number of timeframe count since the last entry order, and return True or False depending on your strategy.

        timeframe (Timeframe): The timeframe granularity of the dataframe.

        take_profit (float, optional): The percent of the buy price to add to create a stop order and take the profit associated. Defaults to np.inf.

        stop_loss (float, optional): The percent of the buy price to cut to create a stop order and stop the loss associated. Defaults to np.inf.

        initial_equity (int, optional): The initial capital. Defaults to 1000.

        maker_fees (Optional[float], optional): The fees applied, here maker for limit orders (not yet implemented). Defaults to 0.001.

        taker_fees (Optional[float], optional): The fees applied, here taker for spot orders. Defaults to 0.001.

        get_trade_df (bool, optional): Whether or not to return the trade dataframe (summary of trades) . Defaults to False.

        get_returns_df (bool, optional): Whether or not to return the strategy returns dataframe. Defaults to False.

        parameter_optimization (bool, optional): This parameter is useful when running fitting/optimization it prints nothing but the final strategy return. Defaults to False.

        plot_result (bool, optional): Plot equity, price, drawdown, distribution of the strategy. Defaults to True.

    Returns:
    -----
        Union[None, Union[float, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]]: Nothing or the strategy total return or the trade_df or the return_df or the trade_df and the return_df.
    """
    assert (
        timeframe in TIMEFRAMES
    ), f"timeframe must be one of the following: {','.join(TIMEFRAMES)}"
    assert df.shape[0] > 1, "Dataframe must have at least 2 rows"
    assert set({"Open", "High", "Low", "Close", "Returns"}).issubset(
        df.columns
    ), "Dataframe must have columns Open, High, Low, Close, Returns"

    ohlcv_df = df.copy()
    trades_df, ohlcv_df = __ohlc_backtest_one_position_type(
        ohlcv_df,
        short_entry_function,
        short_exit_function,
        position_type="short",
        take_profit=take_profit,
        stop_loss=stop_loss,
        maker_fees=maker_fees,
        taker_fees=taker_fees,
    )

    ohlcv_df["Strategy_cum_returns"] = cumulative_returns(ohlcv_df["Strategy_returns"])

    if parameter_optimization is True:
        if len(trades_df) > 0:
            return ohlcv_df["Strategy_cum_returns"].iloc[-1]
        return 0.0

    assert len(trades_df) > 0, "No trades were generated"

    ohlcv_df["Strategy_drawdown"] = drawdown(ohlcv_df["Strategy_returns"])
    ohlcv_df["Cum_returns"] = cumulative_returns(ohlcv_df["Returns"])
    ohlcv_df["Drawdown"] = drawdown(ohlcv_df["Returns"])
    trades_df["trade_duration"] = trades_df["exit_date"] - trades_df["entry_date"]

    print(f"{'  Initial informations  ':-^50}")
    print(f"Period: [{str(ohlcv_df.index[0])}] -> [{str(ohlcv_df.index[-1])}]")
    print(f"Intial balance: {initial_equity:.2f} $")
    if taker_fees is not None and maker_fees is not None:
        print(
            f"Taker fees: {taker_fees*100:.2f} %, Maker fees: {maker_fees*100:.2f} %, All the metrics will be calculated considering these fees"
        )
    else:
        print("No fees considered here.")
    if take_profit != np.inf:
        print(f"Take profit is set to {take_profit*100:.2f} % of the buy price")
    if stop_loss != np.inf:
        print(f"Stop loss is set to {stop_loss*100:.2f} % of the buy price")
    print("Short only positions")
    print_ohlc_backtest_report(
        trades_df=trades_df,
        ohlcv_df=ohlcv_df,
        timeframe=timeframe,
        initial_equity=initial_equity,
    )
    if plot_result is True:
        plot_from_trade_df(ohlcv_df)
    if get_returns_df and get_trade_df:
        return trades_df, ohlcv_df
    if get_trade_df:
        return trades_df
    if get_returns_df:
        return ohlcv_df
