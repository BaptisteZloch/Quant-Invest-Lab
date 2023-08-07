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
    burke_ratio,
    cumulative_returns,
    expectancy,
    profit_factor,
    sharpe_ratio,
    calmar_ratio,
    information_ratio,
    tail_ratio,
    tracking_error,
    treynor_ratio,
    sortino_ratio,
    jensen_alpha,
    r_squared,
    systematic_risk,
    specific_risk,
    portfolio_beta,
    portfolio_alpha,
    max_drawdown,
    drawdown,
    kelly_criterion,
    value_at_risk,
    conditional_value_at_risk,
)
from quant_invest_lab.constants import TIMEFRAME_ANNUALIZED, TIMEFRAMES
from quant_invest_lab.types import Timeframe


def print_portfolio_strategy_report(
    portfolio_and_benchmark_df: pd.DataFrame, timeframe: Timeframe
) -> None:
    assert timeframe in TIMEFRAMES, f"Timeframe {timeframe} not supported"
    assert set(portfolio_and_benchmark_df.columns.tolist()).issuperset(
        {
            "Strategy_returns",
            "Returns",
        }
    ), f"Missing Strategy_returns and Returns columns in dataframe: {portfolio_and_benchmark_df.columns}"

    print(f"\n{'  Returns statistical information  ':-^50}")

    print(
        f"Expected return annualized: {100*portfolio_and_benchmark_df['Strategy_returns'].mean()*TIMEFRAME_ANNUALIZED[timeframe]:.2f} % vs {100*portfolio_and_benchmark_df['Returns'].mean()*TIMEFRAME_ANNUALIZED[timeframe]:.2f} % (buy and hold)"
    )
    print(
        f'Expected volatility annualized: {100*portfolio_and_benchmark_df["Strategy_returns"].std()*(TIMEFRAME_ANNUALIZED[timeframe]**0.5):.2f} % vs {100*portfolio_and_benchmark_df["Returns"].std()*(TIMEFRAME_ANNUALIZED[timeframe]**0.5):.2f} % (buy and hold)'
    )
    print(
        f'Specific volatility (diversifiable) annualized: {100*specific_risk(portfolio_and_benchmark_df["Strategy_returns"], portfolio_and_benchmark_df["Returns"], TIMEFRAME_ANNUALIZED[timeframe]):.2f} %'
    )
    print(
        f'Systematic volatility annualized: {100*systematic_risk(portfolio_and_benchmark_df["Strategy_returns"], portfolio_and_benchmark_df["Returns"], TIMEFRAME_ANNUALIZED[timeframe]):.2f} %'
    )
    print(
        f"Skewness: {skew(portfolio_and_benchmark_df['Strategy_returns'].values):.2f} vs {skew(portfolio_and_benchmark_df.Returns.values):.2f} (buy and hold), <0 = left tail, >0 = right tail -> the higher the better"
    )
    print(
        f"Kurtosis: {kurtosis(portfolio_and_benchmark_df['Strategy_returns'].values):.2f} vs {kurtosis(portfolio_and_benchmark_df.Returns.values):.2f} (buy and hold)",
        ", >3 = fat tails, <3 = thin tails -> the lower the better",
    )
    print(
        f"{timeframe}-95%-VaR: {100*value_at_risk(portfolio_and_benchmark_df['Strategy_returns']):.2f} % vs {100*value_at_risk(portfolio_and_benchmark_df['Returns']):.2f} % (buy and hold) -> the lower the better"
    )
    print(
        f"{timeframe}-95%-CVaR: {100*conditional_value_at_risk(portfolio_and_benchmark_df['Strategy_returns']):.2f} % vs {100*conditional_value_at_risk(portfolio_and_benchmark_df['Returns']):.2f} % (buy and hold) -> the lower the better"
    )

    print(f"\n{'  Strategy statistical information  ':-^50}")
    print(
        f"Max drawdown: {100*max_drawdown(portfolio_and_benchmark_df['Strategy_returns']):.2f} % vs {100*max_drawdown(portfolio_and_benchmark_df['Returns']):.2f} % (buy and hold)"
    )
    print(
        f"Kelly criterion: {100*kelly_criterion(portfolio_and_benchmark_df['Strategy_returns']):.2f} % vs {100*kelly_criterion(portfolio_and_benchmark_df['Returns']):.2f} % (buy and hold)"
    )
    print(
        f"Benchmark sensitivity (beta): {portfolio_beta(portfolio_and_benchmark_df['Strategy_returns'], portfolio_and_benchmark_df['Returns']):.2f} vs 1 (buy and hold)"
    )
    print(
        f"Excess return (alpha): {portfolio_alpha(portfolio_and_benchmark_df['Strategy_returns'], portfolio_and_benchmark_df['Returns']):.4f} vs 0 (buy and hold)"
    )
    print(
        f"Jensen alpha: {jensen_alpha(portfolio_and_benchmark_df['Strategy_returns'], portfolio_and_benchmark_df['Returns'], TIMEFRAME_ANNUALIZED[timeframe]):.4f}"
    )
    print(
        f"Determination coefficient R²: {r_squared(portfolio_and_benchmark_df['Strategy_returns'], portfolio_and_benchmark_df['Returns']):.2f}"
    )
    print(
        f"Tracking error annualized: {100*tracking_error(portfolio_and_benchmark_df['Strategy_returns'], portfolio_and_benchmark_df['Returns'], TIMEFRAME_ANNUALIZED[timeframe]):.2f} %"
    )
    print(f"\n{'  Strategy ratios  ':-^50}")
    print(
        f"Sharpe ratio annualized: {sharpe_ratio(portfolio_and_benchmark_df['Strategy_returns'], TIMEFRAME_ANNUALIZED[timeframe],risk_free_rate=TIMEFRAME_ANNUALIZED[timeframe]*portfolio_and_benchmark_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Sortino ratio annualized: {sortino_ratio(portfolio_and_benchmark_df['Strategy_returns'], TIMEFRAME_ANNUALIZED[timeframe],risk_free_rate=TIMEFRAME_ANNUALIZED[timeframe]*portfolio_and_benchmark_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Treynor ratio annualized: {treynor_ratio(portfolio_and_benchmark_df['Strategy_returns'], portfolio_and_benchmark_df['Returns'], TIMEFRAME_ANNUALIZED[timeframe],risk_free_rate=TIMEFRAME_ANNUALIZED[timeframe]*portfolio_and_benchmark_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Burke ratio annualized: {burke_ratio(portfolio_and_benchmark_df['Strategy_returns'],n_drawdowns=5, risk_free_rate=TIMEFRAME_ANNUALIZED[timeframe]*portfolio_and_benchmark_df.Returns.mean()):.2f} (risk free rate = buy and hold)"
    )
    print(
        f"Calmar ratio annualized: {calmar_ratio(portfolio_and_benchmark_df['Strategy_returns'], TIMEFRAME_ANNUALIZED[timeframe]):.2f}"
    )
    print(
        f"Information ratio annualized: {information_ratio(portfolio_and_benchmark_df['Strategy_returns'], portfolio_and_benchmark_df['Returns'], TIMEFRAME_ANNUALIZED[timeframe]):.2f}"
    )
    print(
        f"Tail ratio annualized: {tail_ratio(portfolio_and_benchmark_df['Strategy_returns']):.2f}"
    )


def print_ohlc_backtest_report(
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
        f'Strategy final net balance: {ohlcv_df["Strategy_cum_returns"].iloc[-1]*initial_equity:.2f} $, return: {(ohlcv_df["Strategy_cum_returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(
        f'Buy & Hold final net balance: {ohlcv_df["Cum_returns"].iloc[-1]*initial_equity:.2f} $, returns: {(ohlcv_df["Cum_returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(f"Strategy winrate ratio: {100 * len(good_trades) / total_trades:.2f} %")
    print(
        f"Strategy profit factor ratio: {profit_factor(good_trades['trade_return'].mean(),bad_trades['trade_return'].mean()):.2f}"
    )
    print(
        f"Strategy expectancy: {100*expectancy(len(good_trades) / total_trades,good_trades['trade_return'].mean(),bad_trades['trade_return'].mean()):.2f} %"
    )

    print_portfolio_strategy_report(ohlcv_df, timeframe)

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
        f"  Best trades return: {100*trades_df['trade_return'].max():.2f} % | Date: {trades_df.iloc[trades_df['trade_return'].idxmax()]['exit_date']} | Duration: {trades_df.iloc[trades_df['trade_return'].idxmax()]['trade_duration']}"  # type: ignore
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
        f"  Worst trades return: {100*trades_df['trade_return'].min():.2f} % | Date: {trades_df.iloc[trades_df['trade_return'].idxmin()]['exit_date']} | Duration: {trades_df.iloc[trades_df['trade_return'].idxmin()]['trade_duration']}"  # type: ignore
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


def plot_from_trade_df(price_df: pd.DataFrame) -> None:
    """Plot historical price, equity progression, drawdown evolution and return distribution.

    Args:
    ----
        price_df (pd.DataFrame): The historical price dataframe.

    """
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Historical price",
            "Return distribution",
            "Equity progression",
            "Expected return profile",
            "Drawdown evolution",
            "Return per decile",
        ),
        shared_xaxes=False,
    )

    fig.add_trace(
        go.Candlestick(
            name="Historical price",
            x=price_df.index,
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            legendgroup="1",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Historical price", row=1, col=1)
    fig.update_xaxes(title_text="Datetime", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            name="Buy and hold cumulative return",
            x=price_df.index,
            y=price_df["Cum_returns"],
            line={"shape": "hv", "color": "violet"},
            legendgroup="2",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Strategy cumulative return",
            x=price_df.index,
            y=price_df["Strategy_cum_returns"],
            line={"shape": "hv", "color": "salmon"},
            legendgroup="2",
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
            legendgroup="3",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Strategy drawdown",
            x=price_df.index,
            y=price_df["Strategy_drawdown"],
            line={"shape": "hv", "color": "salmon"},
            fill="tozeroy",
            fillcolor="rgba(255, 99, 71, 0.35)",
            legendgroup="3",
        ),
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text="Drawdown",
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text="Datetime",
        row=3,
        col=1,
    )

    windows_bh = [day for day in range(5, price_df["Cum_returns"].shape[0] // 3, 30)]

    fig.add_trace(
        go.Scatter(
            name="Strategy expected return profile",
            legendgroup="2",
            x=windows_bh,
            y=[
                price_df["Strategy_cum_returns"]
                .rolling(window)
                .apply(lambda prices: (prices[-1] / prices[0]) - 1)
                .mean()
                for window in windows_bh
            ],
            line={"color": "salmon"},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name="Buy and Hold expected return profile",
            legendgroup="2",
            x=windows_bh,
            y=[
                price_df["Cum_returns"]
                .rolling(window)
                .apply(lambda prices: (prices[-1] / prices[0]) - 1)
                .mean()
                for window in windows_bh
            ],
            line={"color": "violet"},
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(
        title_text="Expected return",
        row=2,
        col=2,
    )
    fig.update_xaxes(
        title_text="Horizon (in candles)",
        row=2,
        col=2,
    )

    distplot_bench = ff.create_distplot(
        [price_df["Returns"]],
        ["Benchmark Returns"],
        colors=["violet"],
        curve_type="kde",
        bin_size=3.5
        * price_df["Returns"].std()
        / (len(price_df["Returns"]) ** (1 / 3)),
    )
    fig.add_trace(distplot_bench["data"][0], row=1, col=2)
    distplot = ff.create_distplot(
        [price_df["Strategy_returns"]],
        ["Strategy Returns"],
        colors=["salmon"],
        curve_type="kde",
        bin_size=3.5
        * price_df["Strategy_returns"].std()
        / (len(price_df["Strategy_returns"]) ** (1 / 3)),
    )
    fig.add_trace(distplot["data"][0], row=1, col=2)

    fig.update_xaxes(title_text="Returns", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    deciles = np.array(
        [
            (chunks["Returns"].mean(), chunks["Strategy_returns"].mean())
            for chunks in np.array_split(
                price_df.sort_values(by="Returns", ascending=True), 10
            )
        ]
    )

    fig.add_trace(
        go.Bar(
            x=np.arange(1, deciles[:, 0].shape[0] + 1),
            y=deciles[:, 0],
            name="Buy and hold returns",
            marker_color="violet",
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=np.arange(1, deciles[:, 1].shape[0] + 1),
            y=deciles[:, 1],
            name="Strategy returns",
            marker_color="salmon",
        ),
        row=3,
        col=2,
    )

    fig.update_xaxes(title_text="Deciles", row=3, col=2)
    fig.update_yaxes(title_text="Returns", row=3, col=2)

    fig.update_layout(
        legend=dict(
            orientation="v",  # Set the orientation of the legends to horizontal
            # yanchor="top",
            # y=0,
            # xanchor="center",
            # x=0.0,
        ),
        # legend_tracegroupgap=180,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        title_text="Historical price, strategy equity evolution/drawdown and returns distribution",
        height=1000,
    )

    fig.show()
