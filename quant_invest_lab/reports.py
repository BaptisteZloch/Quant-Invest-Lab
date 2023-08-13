from scipy.stats import skew, kurtosis
from typing import Callable, Literal, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from quant_invest_lab.metrics import (
    burke_ratio,
    cumulative_returns,
    expectancy,
    omega_ratio,
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
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    timeframe: Timeframe = "1hour",
) -> pd.DataFrame:
    assert timeframe in TIMEFRAMES, f"Timeframe {timeframe} not supported"

    if benchmark_returns is not None:
        report_df = pd.DataFrame(columns=["Portfolio", "Benchmark"])
        assert (
            portfolio_returns.shape[0] == benchmark_returns.shape[0]
        ), f"Error: portfolio and benchmark returns must have the same length"
        report_df = pd.DataFrame(columns=["Portfolio", "Benchmark"])
        report_df.loc["Expected return", "Portfolio"] = (
            portfolio_returns.mean() * TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Expected volatility", "Portfolio"] = portfolio_returns.std() * (
            TIMEFRAME_ANNUALIZED[timeframe] ** 0.5
        )
        report_df.loc["Specific risk", "Portfolio"] = specific_risk(
            portfolio_returns, benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Systematic risk", "Portfolio"] = systematic_risk(
            portfolio_returns, benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Portfolio beta", "Portfolio"] = portfolio_beta(
            portfolio_returns, benchmark_returns
        )
        report_df.loc["Portfolio alpha", "Portfolio"] = portfolio_alpha(
            portfolio_returns, benchmark_returns
        )
        report_df.loc["Jensen alpha", "Portfolio"] = jensen_alpha(
            portfolio_returns, benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Skewness", "Portfolio"] = skew(portfolio_returns.values)
        report_df.loc["Kurtosis", "Portfolio"] = kurtosis(portfolio_returns.values)
        report_df.loc["VaR", "Portfolio"] = value_at_risk(portfolio_returns)
        report_df.loc["CVaR", "Portfolio"] = conditional_value_at_risk(
            portfolio_returns
        )
        report_df.loc["Max drawdown", "Portfolio"] = max_drawdown(portfolio_returns)
        report_df.loc["Kelly criterion", "Portfolio"] = kelly_criterion(
            portfolio_returns
        )
        report_df.loc["R2", "Portfolio"] = r_squared(
            portfolio_returns, benchmark_returns
        )
        report_df.loc["Tracking error", "Portfolio"] = tracking_error(
            portfolio_returns, benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )

        report_df.loc["Sharpe ratio", "Portfolio"] = sharpe_ratio(
            portfolio_returns, TIMEFRAME_ANNUALIZED[timeframe], risk_free_rate=0.0
        )
        report_df.loc["Sortino ratio", "Portfolio"] = sortino_ratio(
            portfolio_returns, TIMEFRAME_ANNUALIZED[timeframe], risk_free_rate=0
        )
        report_df.loc["Burke ratio", "Portfolio"] = burke_ratio(
            portfolio_returns,
            n_drawdowns=5,
            risk_free_rate=0,
            N=TIMEFRAME_ANNUALIZED[timeframe],
        )
        report_df.loc["Calmar ratio", "Portfolio"] = calmar_ratio(
            portfolio_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Tail ratio", "Portfolio"] = tail_ratio(portfolio_returns)

        report_df.loc["Treynor ratio", "Portfolio"] = treynor_ratio(
            portfolio_returns,
            benchmark_returns,
            TIMEFRAME_ANNUALIZED[timeframe],
            risk_free_rate=0,
        )
        report_df.loc["Information ratio", "Portfolio"] = information_ratio(
            portfolio_returns, benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )

        report_df.loc["Expected return", "Benchmark"] = (
            benchmark_returns.mean() * TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Expected volatility", "Benchmark"] = benchmark_returns.std() * (
            TIMEFRAME_ANNUALIZED[timeframe] ** 0.5
        )
        report_df.loc["Specific risk", "Benchmark"] = 0
        report_df.loc["Systematic risk", "Benchmark"] = report_df.loc[
            "Expected volatility", "Benchmark"
        ]
        report_df.loc["Portfolio beta", "Benchmark"] = 1
        report_df.loc["Portfolio alpha", "Benchmark"] = 0
        report_df.loc["Jensen alpha", "Benchmark"] = 0
        report_df.loc["Skewness", "Benchmark"] = skew(benchmark_returns.values)
        report_df.loc["Kurtosis", "Benchmark"] = kurtosis(benchmark_returns.values)
        report_df.loc["VaR", "Benchmark"] = value_at_risk(benchmark_returns)
        report_df.loc["CVaR", "Benchmark"] = conditional_value_at_risk(
            benchmark_returns
        )
        report_df.loc["Max drawdown", "Benchmark"] = max_drawdown(benchmark_returns)
        report_df.loc["Kelly criterion", "Benchmark"] = kelly_criterion(
            benchmark_returns
        )
        report_df.loc["R2", "Benchmark"] = 1
        report_df.loc["Tracking error", "Benchmark"] = 0

        report_df.loc["Sharpe ratio", "Benchmark"] = sharpe_ratio(
            benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe], risk_free_rate=0.0
        )
        report_df.loc["Sortino ratio", "Benchmark"] = sortino_ratio(
            benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe], risk_free_rate=0
        )
        report_df.loc["Burke ratio", "Benchmark"] = burke_ratio(
            benchmark_returns,
            n_drawdowns=5,
            risk_free_rate=0,
            N=TIMEFRAME_ANNUALIZED[timeframe],
        )
        report_df.loc["Calmar ratio", "Benchmark"] = calmar_ratio(
            benchmark_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Tail ratio", "Benchmark"] = tail_ratio(benchmark_returns)

        report_df.loc["Treynor ratio", "Benchmark"] = 0
        report_df.loc["Information ratio", "Benchmark"] = 0

        print(f"\n{'  Returns statistical information  ':-^50}")

        print(
            f"Expected return annualized: {100*report_df.loc['Expected return', 'Portfolio']:.2f} % vs {100*report_df.loc['Expected return', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Expected volatility annualized: {100*report_df.loc['Expected volatility', 'Portfolio']:.2f} % vs {100*report_df.loc['Expected volatility', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Specific volatility (diversifiable) annualized: {100*report_df.loc['Specific risk', 'Portfolio'] :.2f} %"
        )
        print(
            f"Systematic volatility annualized: {100*report_df.loc['Systematic risk', 'Portfolio'] :.2f} %"
        )
        print(
            f"Skewness: {report_df.loc['Skewness', 'Portfolio']:.2f} vs {report_df.loc['Skewness', 'Benchmark']:.2f} (buy and hold), <0 = left tail, >0 = right tail"
        )
        print(
            f"Kurtosis: {report_df.loc['Kurtosis', 'Portfolio']:.2f} vs {report_df.loc['Skewness', 'Benchmark']:.2f} (buy and hold)",
            ", >3 = fat tails, <3 = thin tails",
        )
        print(
            f"{timeframe}-95%-VaR: {100*report_df.loc['VaR', 'Portfolio']:.2f} % vs {100*report_df.loc['VaR', 'Benchmark']:.2f} % (buy and hold) -> the lower the better"
        )
        print(
            f"{timeframe}-95%-CVaR: {100*report_df.loc['CVaR', 'Portfolio']:.2f} % vs {100*report_df.loc['CVaR', 'Benchmark']:.2f} % (buy and hold) -> the lower the better"
        )

        print(f"\n{'  Strategy statistical information  ':-^50}")
        print(
            f"Max drawdown: {100*report_df.loc['Max drawdown', 'Portfolio']:.2f} % vs {100*report_df.loc['Max drawdown', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Kelly criterion: {100*report_df.loc['Kelly criterion', 'Portfolio']:.2f} % vs {100*report_df.loc['Kelly criterion', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Benchmark sensitivity (beta): {report_df.loc['Portfolio beta', 'Portfolio']:.2f} vs 1 (buy and hold)"
        )
        print(
            f"Excess return (alpha): {report_df.loc['Portfolio alpha', 'Portfolio']:.4f} vs 0 (buy and hold)"
        )
        print(f"Jensen alpha: {report_df.loc['Jensen alpha', 'Portfolio']:.4f}")
        print(f"Determination coefficient R²: {report_df.loc['R2', 'Portfolio']:.2f}")
        print(
            f"Tracking error annualized: {100*report_df.loc['Tracking error', 'Portfolio']:.2f} %"
        )
        print(f"\n{'  Strategy ratios  ':-^50}")
        print("No risk free rate considered for the following ratios.\n")
        print(
            f"Sharpe ratio annualized: {report_df.loc['Sharpe ratio', 'Portfolio']:.2f} vs {report_df.loc['Sharpe ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Sortino ratio annualized: {report_df.loc['Sortino ratio', 'Portfolio']:.2f} vs {report_df.loc['Sortino ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Burke ratio annualized: {report_df.loc['Burke ratio', 'Portfolio']:.2f} vs {report_df.loc['Burke ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Calmar ratio annualized: {report_df.loc['Calmar ratio', 'Portfolio']:.2f} vs {report_df.loc['Calmar ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Tail ratio annualized: {report_df.loc['Tail ratio', 'Portfolio']:.2f} vs {report_df.loc['Tail ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Treynor ratio annualized: {report_df.loc['Treynor ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Information ratio annualized: {report_df.loc['Information ratio', 'Portfolio']:.2f}"
        )
    else:
        report_df = pd.DataFrame(columns=["Portfolio"])
        report_df.loc["Expected return annualized", "Portfolio"] = (
            portfolio_returns.mean() * TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc[
            "Expected volatility annualized", "Portfolio"
        ] = portfolio_returns.std() * (TIMEFRAME_ANNUALIZED[timeframe] ** 0.5)
        report_df.loc["Skewness", "Portfolio"] = skew(portfolio_returns.values)
        report_df.loc["Kurtosis", "Portfolio"] = kurtosis(portfolio_returns.values)
        report_df.loc["VaR", "Portfolio"] = value_at_risk(portfolio_returns)
        report_df.loc["CVaR", "Portfolio"] = conditional_value_at_risk(
            portfolio_returns
        )
        report_df.loc["Max drawdown", "Portfolio"] = max_drawdown(portfolio_returns)
        report_df.loc["Kelly criterion", "Portfolio"] = kelly_criterion(
            portfolio_returns
        )
        report_df.loc["Sharpe ratio", "Portfolio"] = sharpe_ratio(
            portfolio_returns, TIMEFRAME_ANNUALIZED[timeframe], risk_free_rate=0.0
        )
        report_df.loc["Sortino ratio", "Portfolio"] = sortino_ratio(
            portfolio_returns, TIMEFRAME_ANNUALIZED[timeframe], risk_free_rate=0
        )
        report_df.loc["Burke ratio", "Portfolio"] = burke_ratio(
            portfolio_returns,
            n_drawdowns=5,
            risk_free_rate=0,
            N=TIMEFRAME_ANNUALIZED[timeframe],
        )
        report_df.loc["Calmar ratio", "Portfolio"] = calmar_ratio(
            portfolio_returns, TIMEFRAME_ANNUALIZED[timeframe]
        )
        report_df.loc["Tail ratio", "Portfolio"] = tail_ratio(portfolio_returns)

        print(f"\n{'  Returns statistical information  ':-^50}")

        print(
            f"Expected return annualized: {100*report_df.loc['Expected return', 'Portfolio']:.2f} %"
        )
        print(
            f"Expected volatility annualized: {100*report_df.loc['Expected volatility', 'Portfolio']:.2f} %"
        )
        print(
            f"Skewness: {report_df.loc['Skewness', 'Portfolio'] :.2f}, <0 = left tail, >0 = right tail"
        )
        print(
            f"Kurtosis: {report_df.loc['Kurtosis', 'Portfolio']:.2f}",
            ", >3 = fat tails, <3 = thin tails",
        )
        print(
            f"{timeframe}-95%-VaR: {100*report_df.loc['VaR', 'Portfolio']:.2f} % -> the lower the better"
        )
        print(
            f"{timeframe}-95%-CVaR: {100*report_df.loc['CVaR', 'Portfolio']:.2f} % -> the lower the better"
        )

        print(f"\n{'  Strategy statistical information  ':-^50}")
        print(f"Max drawdown: {100*report_df.loc['Max drawdown', 'Portfolio']:.2f} %")
        print(
            f"Kelly criterion: {100*report_df.loc['Kelly criterion', 'Portfolio']:.2f} %"
        )
        print(f"\n{'  Strategy ratios  ':-^50}")
        print("No risk free rate considered for the following ratios.\n")
        print(
            f"Sharpe ratio annualized: {report_df.loc['Sharpe ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Sortino ratio annualized: {report_df.loc['Sortino ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Burke ratio annualized: {report_df.loc['Burke ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Calmar ratio annualized: {report_df.loc['Calmar ratio', 'Portfolio']:.2f}"
        )
        print(f"Tail ratio annualized: {report_df.loc['Tail ratio', 'Portfolio']:.2f}")
    return report_df


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

    print_portfolio_strategy_report(
        ohlcv_df["Strategy_returns"], ohlcv_df["Returns"], timeframe
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


def plot_from_trade_df(price_df: pd.DataFrame) -> None:
    """Plot historical price, equity progression, drawdown evolution and return distribution.

    Args:
    ----
        price_df (pd.DataFrame): The historical price dataframe.

    """
    n_rolling = price_df["Strategy_returns"].shape[0] // 10
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Historical price",
            "Return distribution",
            "Equity progression",
            "Expected return profile",
            "Drawdown evolution",
            "Return per decile",
            f"{n_rolling} candles rolling Sharpe ratio",
            "Omega curve",
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

    thresholds = np.linspace(0.01, 0.5, 100)
    omega_bench = []
    omega_ptf = []
    for threshold in thresholds:
        omega_ptf.append(omega_ratio(price_df["Strategy_returns"], threshold))
        omega_bench.append(omega_ratio(price_df["Returns"], threshold))

    fig.add_trace(
        go.Scatter(
            name="Strategy omega curve",
            legendgroup="2",
            x=thresholds,
            y=omega_ptf,
            line={"color": "salmon"},
        ),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name="Benchmark omega curve",
            legendgroup="2",
            x=thresholds,
            y=omega_bench,
            line={"color": "violet"},
        ),
        row=4,
        col=2,
    )
    fig.update_yaxes(
        title_text="Omega ratio",
        row=4,
        col=2,
    )
    fig.update_xaxes(
        title_text="Annual return thresholds",
        row=4,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            name="Strategy sharpe ratio",
            legendgroup="4",
            x=price_df.index,
            y=price_df["Strategy_returns"]
            .rolling(n_rolling)
            .apply(
                lambda rets: (365 * rets.mean()) / (rets.std() * (365**0.5)),
            )
            .fillna(0),
            line={"color": "salmon"},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Buy and Hold sharpe ratio",
            legendgroup="4",
            x=price_df.index,
            y=price_df["Returns"]
            .rolling(n_rolling)
            .apply(
                lambda rets: (365 * rets.mean()) / (rets.std() * (365**0.5)),
            )
            .fillna(0),
            line={"color": "violet"},
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(
        title_text="Rolling sharpe ratio",
        row=4,
        col=1,
    )
    fig.update_xaxes(
        title_text="Datetime",
        row=4,
        col=1,
    )

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
        height=1100,
    )

    fig.show()
