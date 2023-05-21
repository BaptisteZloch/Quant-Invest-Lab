from datetime import datetime
from scipy.stats import skew, kurtosis
from typing import Callable, Literal, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from tqdm import tqdm

from quant_invest_lab.metrics import (
    cumulative_returns,
    expectancy,
    profit_factor,
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown,
    kelly_criterion,
    value_at_risk,
    conditional_value_at_risk,
)
from quant_invest_lab.constants import TIMEFRAME_ANNUALIZED, TIMEFRAMES
from quant_invest_lab.types import Timeframe


def print_ohlc_backtest_report(
    returns_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    timeframe: Timeframe,
    initial_equity: Union[int, float] = 1000,
) -> None:
    good_trades = trades_df.loc[trades_df["trade_return"] > 0]
    bad_trades = trades_df.loc[trades_df["trade_return"] < 0]
    total_trades = len(trades_df)
    print(f"\n{'  Strategy performances  ':-^50}")

    print(
        f'Strategy final net balance: {returns_df["Cum_Returns"].iloc[-1]*initial_equity:.2f} $, return: {(returns_df["Cum_Returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(
        f'Buy & Hold final net balance: {ohlcv_df["Cum_Returns"].iloc[-1]*initial_equity:.2f} $, returns: {(ohlcv_df["Cum_Returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(f"Strategy winrate ratio: {100 * len(good_trades) / total_trades:.2f} %")
    print(
        f"Strategy profit factor ratio: {profit_factor(good_trades['trade_return'].mean(),bad_trades['trade_return'].mean()):.2f}"
    )
    print(
        f"Strategy expectancy: {100*expectancy(len(good_trades) / total_trades,good_trades['trade_return'].mean(),bad_trades['trade_return'].mean()):.2f} %"
    )

    print(f"\n{'  Returns statistical information  ':-^50}")

    print(
        f"Expected return : {100*returns_df['Returns'].mean():.2f} %, annuzalized: {100*returns_df['Returns'].mean()*TIMEFRAME_ANNUALIZED[timeframe]:.2f} %"
    )
    print(
        f"Median return : {100*returns_df['Returns'].median():.2f} %, annuzalized: {100*returns_df['Returns'].median()*TIMEFRAME_ANNUALIZED[timeframe]:.2f} %"
    )
    print(
        f'Expected volatility: {100*returns_df["Returns"].std():.2f} %, annualized: {100*returns_df["Returns"].std()*(TIMEFRAME_ANNUALIZED[timeframe]**0.5):.2f} %'
    )
    print(
        f"Skewness: {skew(returns_df.Returns.values):.2f} vs {skew(ohlcv_df.Returns.values):.2f} (buy and hold), <0 = left tail, >0 = right tail -> the higher the better"
    )
    print(
        f"Kurtosis: {kurtosis(returns_df.Returns.values):.2f} vs {kurtosis(ohlcv_df.Returns.values):.2f} (buy and hold)",
        ", >3 = fat tails, <3 = thin tails -> the lower the better",
    )
    print(
        f"{timeframe}-95%-VaR: {100*value_at_risk(returns_df.Returns):.2f} % vs {100*value_at_risk(ohlcv_df.Returns):.2f} % (buy and hold) -> the lower the better"
    )
    print(
        f"{timeframe}-95%-CVaR: {100*conditional_value_at_risk(returns_df.Returns):.2f} % vs {100*conditional_value_at_risk(ohlcv_df.Returns):.2f} % (buy and hold) -> the lower the better"
    )

    print(f"\n{'  Strategy statistical information  ':-^50}")
    print(
        f"Max drawdown: {100*max_drawdown(returns_df.Returns):.2f} % vs {100*max_drawdown(ohlcv_df.Returns):.2f} % (buy and hold)"
    )
    print(
        f"Kelly criterion: {100*kelly_criterion(returns_df.Returns):.2f} % vs {100*kelly_criterion(ohlcv_df.Returns):.2f} % (buy and hold)"
    )
    print(
        f"Sharpe ratio (annualized): {sharpe_ratio(returns_df['Returns'], TIMEFRAME_ANNUALIZED[timeframe],risk_free_rate=TIMEFRAME_ANNUALIZED[timeframe]*ohlcv_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Sortino ratio (annualized): {sortino_ratio(returns_df['Returns'], TIMEFRAME_ANNUALIZED[timeframe],risk_free_rate=TIMEFRAME_ANNUALIZED[timeframe]*ohlcv_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Calmar ratio (annualized): {calmar_ratio(returns_df['Returns'], TIMEFRAME_ANNUALIZED[timeframe]):.2f} (risk free rate = buy and hold)"
    )

    print(f"\n{'  Trades informations  ':-^50}")
    print(f"Mean trade return : {100*trades_df['trade_return'].mean():.2f} %")
    print(f"Median trade return : {100*trades_df['trade_return'].median():.2f} %")
    print(f'Mean trade volatility: {100*trades_df["trade_return"].std():.2f} %')
    print(
        f"Mean trade duration: {str((trades_df['trade_duration']).mean()).split('.')[0]}"
    )

    print(f"Total trades: {total_trades}")

    print(f"\n  Total good trades: {len(good_trades)}")
    print(f"  Mean good trades return: {100*good_trades['trade_return'].mean():.2f} %")
    print(
        f"  Median good trades return: {100*good_trades['trade_return'].median():.2f} %"
    )
    print(
        f"  Best trades return: {100*trades_df['trade_return'].max():.2f} % | Date: {trades_df.iloc[trades_df['trade_return'].idxmax()]['exit_date']} | Duration: {trades_df.iloc[trades_df['trade_return'].idxmax()]['trade_duration']}"
    )
    print(
        f"  Mean good trade duration: {str((good_trades['trade_duration']).mean()).split('.')[0]}"
    )
    print(f"\n  Total bad trades: {len(bad_trades)}")
    print(f"  Mean bad trades return: {100*bad_trades['trade_return'].mean():.2f} %")
    print(
        f"  Median bad trades return: {100*bad_trades['trade_return'].median():.2f} %"
    )
    print(
        f"  Worst trades return: {100*trades_df['trade_return'].min():.2f} % | Date: {trades_df.iloc[trades_df['trade_return'].idxmin()]['exit_date']} | Duration: {trades_df.iloc[trades_df['trade_return'].idxmin()]['trade_duration']}"
    )
    print(
        f"  Mean bad trade duration: {str((bad_trades['trade_duration']).mean()).split('.')[0]}"
    )

    print(f"\nExit reasons repartition: ")
    for reason, val in zip(
        trades_df.exit_reason.value_counts().index, trades_df.exit_reason.value_counts()
    ):
        print(f"- {reason}: {val}")


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
        tuple[pd.DataFrame, pd.DataFrame]: It returns 2 dataframes, the first is the trades_df : trade summary dataframe and the second is the returns dataframe : returns_df.
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

    returns_df = pd.DataFrame(columns=["Returns"])

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
            returns_df = pd.concat(
                [returns_df, pd.DataFrame({"Returns": rets.values}, index=rets.index)],
                ignore_index=False,
            )
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
            returns_df = pd.concat(
                [returns_df, pd.DataFrame({"Returns": rets.values}, index=rets.index)],
                ignore_index=False,
            )
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
            returns_df = pd.concat(
                [returns_df, pd.DataFrame({"Returns": rets.values}, index=rets.index)],
                ignore_index=False,
            )
            timeframe_count = 0
            current_trade = {}
        else:
            timeframe_count += 1
        previous_row = row
    # Close last trade
    if position_opened is True:
        # print("Closing trade at end of backtest")
        position_opened = False
        current_trade["exit_date"] = ohlcv_df.index[-1]
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
        returns_df = pd.concat(
            [returns_df, pd.DataFrame({"Returns": rets.values}, index=rets.index)],
            ignore_index=False,
        )
        timeframe_count = 0
        current_trade = {}
    return trades_df, returns_df


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
    trades_df, returns_df = __ohlc_backtest_one_position_type(
        ohlcv_df,
        long_entry_function,
        long_exit_function,
        position_type="long",
        take_profit=take_profit,
        stop_loss=stop_loss,
        maker_fees=maker_fees,
        taker_fees=taker_fees,
    )

    returns_df["Cum_Returns"] = cumulative_returns(returns_df["Returns"])

    if parameter_optimization is True:
        if len(trades_df) > 0:
            return returns_df["Cum_Returns"].iloc[-1]
        return 0.0

    assert len(trades_df) > 0, "No trades were generated"

    returns_df["Drawdown"] = drawdown(returns_df["Returns"])
    ohlcv_df["Cum_Returns"] = cumulative_returns(df["Returns"])
    ohlcv_df["Drawdown"] = drawdown(df["Returns"])
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
        returns_df=returns_df,
        trades_df=trades_df,
        ohlcv_df=ohlcv_df,
        timeframe=timeframe,
        initial_equity=initial_equity,
    )

    if plot_result is True:
        plot_from_trade_df(trades_df, ohlcv_df, returns_df)
    if get_returns_df and get_trade_df:
        return trades_df, returns_df
    if get_trade_df:
        return trades_df
    if get_returns_df:
        return returns_df


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
    trades_df, returns_df = __ohlc_backtest_one_position_type(
        ohlcv_df,
        short_entry_function,
        short_exit_function,
        position_type="short",
        take_profit=take_profit,
        stop_loss=stop_loss,
        maker_fees=maker_fees,
        taker_fees=taker_fees,
    )

    returns_df["Cum_Returns"] = cumulative_returns(returns_df["Returns"])

    if parameter_optimization is True:
        if len(trades_df) > 0:
            return returns_df["Cum_Returns"].iloc[-1]
        return 0.0

    assert len(trades_df) > 0, "No trades were generated"

    returns_df["Drawdown"] = drawdown(returns_df["Returns"])
    ohlcv_df["Cum_Returns"] = cumulative_returns(df["Returns"])
    ohlcv_df["Drawdown"] = drawdown(df["Returns"])
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
        returns_df=returns_df,
        trades_df=trades_df,
        ohlcv_df=ohlcv_df,
        timeframe=timeframe,
        initial_equity=initial_equity,
    )
    if plot_result is True:
        plot_from_trade_df(trades_df, ohlcv_df, returns_df)
    if get_returns_df and get_trade_df:
        return trades_df, returns_df
    if get_trade_df:
        return trades_df
    if get_returns_df:
        return returns_df


def plot_from_trade_df(
    trade_df: pd.DataFrame, price_df: pd.DataFrame, returns_df: pd.DataFrame
) -> None:
    """Plot historical price, equity progression, drawdown evolution and return distribution.

    Args:
    ----
        trade_df (pd.DataFrame): The trade summary dataframe.
        price_df (pd.DataFrame): The historical price dataframe.
        returns_df (pd.DataFrame): The detailed strategy returns dataframe.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Historical price",
            "Equity progression",
            "Drawdown Evolution",
            "Return distribution",
        ),
        shared_xaxes=True,
    )

    fig.add_trace(
        go.Candlestick(
            name="Historical price",
            x=price_df.index,
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Historical price (USDT)", row=1, col=1)
    fig.update_xaxes(title_text="Datetime", row=1, col=1)
    trades = trade_df.copy()
    trades["date"] = trades["exit_date"]
    trades = trades.set_index("date")

    fig.add_trace(
        go.Scatter(
            name="Buy and hold cumulative return",
            x=price_df.index,
            y=price_df["Cum_Returns"],
            line={"shape": "hv", "color": "violet"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Strategy cumulative return",
            x=returns_df.index,
            y=returns_df["Cum_Returns"],
            line={"shape": "hv", "color": "salmon"},
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_yaxes(title_text="Log cumulative returns", row=2, col=1)
    fig.update_xaxes(title_text="Datetime", row=2, col=1)
    fig.add_trace(
        go.Scatter(
            name="Buy and hold drawdown",
            x=price_df.index,
            y=price_df["Drawdown"],
            line={"shape": "hv", "color": "violet"},
            fill="tozeroy",
            fillcolor="rgba(238, 130, 238, 0.35)",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Strategy drawdown",
            x=returns_df.index,
            y=returns_df["Drawdown"],
            line={"shape": "hv", "color": "salmon"},
            fill="tozeroy",
            fillcolor="rgba(255, 99, 71, 0.35)",
        ),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="Drawdown", row=3, col=1)
    fig.update_xaxes(title_text="Datetime", row=3, col=1)
    distplot_bench = ff.create_distplot(
        [price_df["Returns"]],
        ["Benchmark Returns"],
        colors=["violet"],
        curve_type="kde",
        bin_size=3.5
        * price_df["Returns"].std()
        / (len(price_df["Returns"]) ** (1 / 3)),
    )
    fig.add_trace(distplot_bench["data"][0], row=4, col=1)
    distplot = ff.create_distplot(
        [returns_df["Returns"]],
        ["Strategy Returns"],
        colors=["salmon"],
        curve_type="kde",
        bin_size=3.5
        * returns_df["Returns"].std()
        / (len(returns_df["Returns"]) ** (1 / 3)),
    )
    fig.add_trace(distplot["data"][0], row=4, col=1)

    fig.update_xaxes(title_text="Returns", row=4, col=1)
    fig.update_yaxes(title_text="Density", row=4, col=1)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        title_text="Historical price, strategy equity evolution/drawdown and returns distribution",
        height=1000,
    )
    fig.show()
