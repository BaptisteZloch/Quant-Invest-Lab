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
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown,
    kelly_criterion,
    value_at_risk,
    conditional_value_at_risk,
)


def ohlc_long_only_backtester(
    df: pd.DataFrame,
    long_entry_function: Callable[[pd.Series, pd.Series], bool],
    long_exit_function: Callable[[pd.Series, pd.Series, int], bool],
    timeframe: Literal[
        "1min",
        "2min",
        "5min",
        "15min",
        "30min",
        "1hour",
        "2hour",
        "4hour",
        "12hour",
        "1day",
    ],
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

        timeframe (Literal[ &quot;1min&quot;, &quot;2min&quot;, &quot;5min&quot;, &quot;15min&quot;, &quot;30min&quot;, &quot;1hour&quot;, &quot;2hour&quot;, &quot;4hour&quot;, &quot;12hour&quot;, &quot;1day&quot;, ]): The timeframe granularity of the dataframe.

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
    assert timeframe in [
        "1min",
        "2min",
        "5min",
        "15min",
        "30min",
        "1hour",
        "2hour",
        "4hour",
        "12hour",
        "1day",
    ], "timeframe must be one of the following: 1min, 2min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 12hour, 1day"
    assert df.shape[0] > 1, "Dataframe must have at least 2 rows"
    assert set({"Open", "High", "Low", "Close", "Returns"}).issubset(
        df.columns
    ), "Dataframe must have columns Open, High, Low, Close, Returns"
    timeframe_annualized = {
        "1min": int(365 * 24 / 1 * 60),
        "2min": int(365 * 24 / 1 * 30),
        "5min": int(365 * 24 / 1 * 12),
        "15min": int(365 * 24 / 1 * 4),
        "30min": int(365 * 24 / 1 * 2),
        "1hour": int(365 * 24 / 1),
        "2hour": int(365 * 24 / 2),
        "4hour": int(365 * 24 / 4),
        "12hour": int(365 * 24 / 12),
        "1day": 365,
    }
    N = timeframe_annualized[timeframe]
    ohlcv_df = df.copy()
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

    for index, row in tqdm(
        ohlcv_df[1:].iterrows(),
        desc="Backtesting...",
        total=ohlcv_df.shape[0] - 1,
        leave=False,
    ):
        if position_opened is False and long_entry_function(row, previous_row) is True:
            position_opened = True
            current_trade["entry_date"] = index
            current_trade["entry_price"] = row.Close
            current_trade["entry_reason"] = "Entry position triggered"
            timeframe_count = 1
            entry_price = row.Close
        elif (
            position_opened is True
            and current_trade["entry_price"] * (1 + take_profit) <= row.High
        ):
            position_opened = False
            current_trade["exit_date"] = index
            current_trade["exit_price"] = current_trade["entry_price"] * (
                1 + take_profit
            )
            current_trade["exit_reason"] = "Take profit triggered"

            rets = ohlcv_df.loc[
                current_trade["entry_date"] : current_trade["exit_date"]
            ].Returns
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
            and current_trade["entry_price"] * (1 - stop_loss) >= row.Low
        ):
            position_opened = False
            current_trade["exit_date"] = index
            current_trade["exit_price"] = float(current_trade["entry_price"]) * (
                1 - stop_loss
            )
            current_trade["exit_reason"] = "Stop loss triggered"

            rets = ohlcv_df.loc[
                current_trade["entry_date"] : current_trade["exit_date"]
            ].Returns
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
            and long_exit_function(row, previous_row, timeframe_count) is True
        ):
            position_opened = False
            current_trade["exit_date"] = index
            current_trade["exit_price"] = row.Close
            current_trade["exit_reason"] = "Exit position triggered"

            rets = ohlcv_df.loc[
                current_trade["entry_date"] : current_trade["exit_date"]
            ].Returns

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

    returns_df["Cum_Returns"] = (returns_df["Returns"] + 1).cumprod()
    returns_df["Drawdown"] = drawdown(returns_df["Returns"])
    ohlcv_df["Cum_Returns"] = (df["Returns"] + 1).cumprod()
    ohlcv_df["Drawdown"] = drawdown(df["Returns"])
    if parameter_optimization is True:
        if len(trades_df) > 0:
            return returns_df["Cum_Returns"].iloc[-1]
        return 0.0

    assert len(trades_df) > 0, "No trades were generated"
    trades_df["trade_duration"] = trades_df["exit_date"] - trades_df["entry_date"]

    good_trades = trades_df.loc[trades_df["trade_return"] > 0]
    bad_trades = trades_df.loc[trades_df["trade_return"] < 0]
    total_trades = len(trades_df)

    print(f"{'  Initial informations  ':-^50}")
    print(f"Period: [{str(ohlcv_df.index[0])}] -> [{str(ohlcv_df.index[-1])}]")
    print(f"Intial balance: {initial_equity:.2f} $")
    if taker_fees is not None and maker_fees is not None:
        print(
            f"Taker fees: {taker_fees*100:.2f} %, Maker fees: {maker_fees*100:.2f} %, All the metrics will be calculated considering these fees"
        )
    else:
        print("No fees considered here.")

    print(f"\n{'  Strategy performances  ':-^50}")

    print(
        f'Strategy final net balance: {returns_df["Cum_Returns"].iloc[-1]*initial_equity:.2f} $, return: {(returns_df["Cum_Returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(
        f'Buy & Hold final net balance: {ohlcv_df["Cum_Returns"].iloc[-1]*initial_equity:.2f} $, returns: {(ohlcv_df["Cum_Returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(f"Strategy winrate ratio: {100 * len(good_trades) / total_trades:.2f} %")
    print(
        f"Strategy profit factor ratio: {abs(good_trades['trade_return'].mean()/bad_trades['trade_return'].mean()):.2f}"
    )

    print(f"\n{'  Returns statistical information  ':-^50}")

    print(
        f"Expected return : {100*returns_df['Returns'].mean():.2f} %, annuzalized: {100*returns_df['Returns'].mean()*N:.2f} %"
    )
    print(
        f"Median return : {100*returns_df['Returns'].median():.2f} %, annuzalized: {100*returns_df['Returns'].median()*N:.2f} %"
    )
    print(
        f'Expected volatility: {100*returns_df["Returns"].std():.2f} %, annualized: {100*returns_df["Returns"].std()*(N**0.5):.2f} %'
    )
    print(
        f'Skewness: {skew(returns_df["Returns"].values):.2f}, <0 = left tail, >0 = right tail -> the higher the better'
    )
    print(
        f'Kurtosis: {kurtosis(returns_df["Returns"].values):.2f}',
        ", >3 = fat tails, <3 = thin tails -> the lower the better",
    )
    print(f"{timeframe}-95%-VaR: {100*value_at_risk(returns_df['Returns']):.2f} %")
    print(
        f"{timeframe}-95%-CVaR: {100*conditional_value_at_risk(returns_df['Returns']):.2f} %"
    )

    print(f"\n{'  Strategy statistical information  ':-^50}")
    print(f"Max drawdown: {100*max_drawdown(returns_df['Returns']):.2f} %")
    print(f"Kelly criterion: {100*kelly_criterion(returns_df.Returns):.2f} %")
    print(
        f"Sharpe ratio (annualized): {sharpe_ratio(returns_df['Returns'], N,risk_free_rate=N*ohlcv_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Sortino ratio (annualized): {sortino_ratio(returns_df['Returns'], N,risk_free_rate=N*ohlcv_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Calmar ratio (annualized): {calmar_ratio(returns_df['Returns'], N):.2f} (risk free rate = buy and hold)"
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
    timeframe: Literal[
        "1min",
        "2min",
        "5min",
        "15min",
        "30min",
        "1hour",
        "2hour",
        "4hour",
        "12hour",
        "1day",
    ],
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

        timeframe (Literal[ &quot;1min&quot;, &quot;2min&quot;, &quot;5min&quot;, &quot;15min&quot;, &quot;30min&quot;, &quot;1hour&quot;, &quot;2hour&quot;, &quot;4hour&quot;, &quot;12hour&quot;, &quot;1day&quot;, ]): The timeframe granularity of the dataframe.

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
    assert timeframe in [
        "1min",
        "2min",
        "5min",
        "15min",
        "30min",
        "1hour",
        "2hour",
        "4hour",
        "12hour",
        "1day",
    ], "timeframe must be one of the following: 1min, 2min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 12hour, 1day"
    assert df.shape[0] > 1, "Dataframe must have at least 2 rows"
    assert set({"Open", "High", "Low", "Close", "Returns"}).issubset(
        df.columns
    ), "Dataframe must have columns Open, High, Low, Close, Returns"
    timeframe_annualized = {
        "1min": int(365 * 24 / 1 * 60),
        "2min": int(365 * 24 / 1 * 30),
        "5min": int(365 * 24 / 1 * 12),
        "15min": int(365 * 24 / 1 * 4),
        "30min": int(365 * 24 / 1 * 2),
        "1hour": int(365 * 24 / 1),
        "2hour": int(365 * 24 / 2),
        "4hour": int(365 * 24 / 4),
        "12hour": int(365 * 24 / 12),
        "1day": 365,
    }
    N = timeframe_annualized[timeframe]
    ohlcv_df = df.copy()
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

    for index, row in tqdm(
        ohlcv_df[1:].iterrows(),
        desc="Backtesting...",
        total=ohlcv_df.shape[0] - 1,
        leave=False,
    ):
        if position_opened is False and short_entry_function(row, previous_row) is True:
            position_opened = True
            current_trade["entry_date"] = index
            current_trade["entry_price"] = row.Close
            current_trade["entry_reason"] = "Entry position triggered"
            timeframe_count = 1
            entry_price = row.Close
        elif (
            position_opened is True
            and current_trade["entry_price"] * (1 + stop_loss) <= row.High
        ):
            position_opened = False
            current_trade["exit_date"] = index
            current_trade["exit_price"] = current_trade["entry_price"] * (1 + stop_loss)
            current_trade["exit_reason"] = "Stop loss triggered"

            rets = -ohlcv_df.loc[
                current_trade["entry_date"] : current_trade["exit_date"]
            ].Returns
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
            and current_trade["entry_price"] * (1 - take_profit) >= row.Low
        ):
            position_opened = False
            current_trade["exit_date"] = index
            current_trade["exit_price"] = float(current_trade["entry_price"]) * (
                1 - take_profit
            )
            current_trade["exit_reason"] = "Take profit triggered"

            rets = -ohlcv_df.loc[
                current_trade["entry_date"] : current_trade["exit_date"]
            ].Returns
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
            and short_exit_function(row, previous_row, timeframe_count) is True
        ):
            position_opened = False
            current_trade["exit_date"] = index
            current_trade["exit_price"] = row.Close
            current_trade["exit_reason"] = "Exit position triggered"

            rets = -ohlcv_df.loc[
                current_trade["entry_date"] : current_trade["exit_date"]
            ].Returns

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

    returns_df["Cum_Returns"] = (returns_df["Returns"] + 1).cumprod()
    returns_df["Drawdown"] = drawdown(returns_df["Returns"])
    ohlcv_df["Cum_Returns"] = (df["Returns"] + 1).cumprod()
    ohlcv_df["Drawdown"] = drawdown(df["Returns"])
    if parameter_optimization is True:
        if len(trades_df) > 0:
            return returns_df["Cum_Returns"].iloc[-1]
        return 0.0

    assert len(trades_df) > 0, "No trades were generated"
    trades_df["trade_duration"] = trades_df["exit_date"] - trades_df["entry_date"]

    good_trades = trades_df.loc[trades_df["trade_return"] > 0]
    bad_trades = trades_df.loc[trades_df["trade_return"] < 0]
    total_trades = len(trades_df)

    print(f"{'  Initial informations  ':-^50}")
    print(f"Period: [{str(ohlcv_df.index[0])}] -> [{str(ohlcv_df.index[-1])}]")
    print(f"Intial balance: {initial_equity:.2f} $")
    if taker_fees is not None and maker_fees is not None:
        print(
            f"Taker fees: {taker_fees*100:.2f} %, Maker fees: {maker_fees*100:.2f} %, All the metrics will be calculated considering these fees"
        )
    else:
        print("No fees considered here.")

    print(f"\n{'  Strategy performances  ':-^50}")

    print(
        f'Strategy final net balance: {returns_df["Cum_Returns"].iloc[-1]*initial_equity:.2f} $, return: {(returns_df["Cum_Returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(
        f'Buy & Hold final net balance: {ohlcv_df["Cum_Returns"].iloc[-1]*initial_equity:.2f} $, returns: {(ohlcv_df["Cum_Returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(f"Strategy winrate ratio: {100 * len(good_trades) / total_trades:.2f} %")
    print(
        f"Strategy profit factor ratio: {abs(good_trades['trade_return'].mean()/bad_trades['trade_return'].mean()):.2f}"
    )

    print(f"\n{'  Returns statistical information  ':-^50}")

    print(
        f"Expected return : {100*returns_df['Returns'].mean():.2f} %, annuzalized: {100*returns_df['Returns'].mean()*N:.2f} %"
    )
    print(
        f"Median return : {100*returns_df['Returns'].median():.2f} %, annuzalized: {100*returns_df['Returns'].median()*N:.2f} %"
    )
    print(
        f'Expected volatility: {100*returns_df["Returns"].std():.2f} %, annualized: {100*returns_df["Returns"].std()*(N**0.5):.2f} %'
    )
    print(
        f'Skewness: {skew(returns_df["Returns"].values):.2f}, <0 = left tail, >0 = right tail -> the higher the better'
    )
    print(
        f'Kurtosis: {kurtosis(returns_df["Returns"].values):.2f}',
        ", >3 = fat tails, <3 = thin tails -> the lower the better",
    )
    print(f"{timeframe}-95%-VaR: {100*value_at_risk(returns_df['Returns']):.2f} %")
    print(
        f"{timeframe}-95%-CVaR: {100*conditional_value_at_risk(returns_df['Returns']):.2f} %"
    )

    print(f"\n{'  Strategy statistical information  ':-^50}")
    print(f"Max drawdown: {100*max_drawdown(returns_df['Returns']):.2f} %")
    print(f"Kelly criterion: {100*kelly_criterion(returns_df.Returns):.2f} %")
    print(
        f"Sharpe ratio (annualized): {sharpe_ratio(returns_df['Returns'], N,risk_free_rate=N*ohlcv_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Sortino ratio (annualized): {sortino_ratio(returns_df['Returns'], N,risk_free_rate=N*ohlcv_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Calmar ratio (annualized): {calmar_ratio(returns_df['Returns'], N):.2f} (risk free rate = buy and hold)"
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
