from quant_invest_lab.types import Timeframe
from bokeh.models import (
    DatetimeTickFormatter,
    NumeralTickFormatter,
)

TIMEFRAME_ANNUALIZED = {
    "1min": int(365 * 24 / 1 * 60),
    "3min": int(365 * 24 / 1 * 30),
    "5min": int(365 * 24 / 1 * 12),
    "15min": int(365 * 24 / 1 * 4),
    "30min": int(365 * 24 / 1 * 2),
    "1hour": int(365 * 24 / 1),
    "2hour": int(365 * 24 / 2),
    "4hour": int(365 * 24 / 4),
    "12hour": int(365 * 24 / 12),
    "1day": 365,
}

TIMEFRAME_TO_FREQ = {
    "1min": "1T",
    "3min": "2T",
    "5min": "5T",
    "15min": "15T",
    "30min": "30T",
    "1hour": "1H",
    "2hour": "2H",
    "4hour": "4H",
    "12hour": "12H",
    "1day": "1D",
}

TIMEFRAMES: tuple = (
    "1min",
    "3min",
    "5min",
    "15min",
    "30min",
    "1hour",
    "2hour",
    "4hour",
    "12hour",
    "1day",
)

TIMEFRAME_IN_S = {
    "1min": 60,
    "3min": 60 * 2,
    "5min": 60 * 5,
    "15min": 60 * 15,
    "30min": 60 * 15,
    "1hour": 60**2,
    "2hour": 2 * 60**2,
    "4hour": 4 * 60**2,
    "12hour": 12 * 60**2,
    "1day": 24 * 60**2,
}

PORTFOLIO_METRICS = [
    "Expected return",
    "CAGR",
    "Expected volatility",
    "Skewness",
    "Kurtosis",
    "VaR",
    "CVaR",
    "Profit factor",
    "Expectancy",
    "Payoff ratio",
    "Max drawdown",
    "Kelly criterion",
    "Sharpe ratio",
    "Sortino ratio",
    "Burke ratio",
    "Calmar ratio",
    "Tail ratio",
    "Specific risk",
    "Systematic risk",
    "Portfolio beta",
    "Portfolio alpha",
    "Jensen alpha",
    "R2",
    "Tracking error",
    "Treynor ratio",
    "Information ratio",
]


DT_FORMATTER = DatetimeTickFormatter(
    hours="%d-%m-%Y",
    days="%d-%m-%Y",
    months="%d-%m-%Y",
    years="%d-%m-%Y",
)
RETURNS_FORMATTER = NumeralTickFormatter(format="0%")

SALMON_COLOR = "rgba(255, 99, 71,1)"
VIOLET_COLOR = "rgba(238, 130, 238, 1)"
BACKGROUND_COLOR = "#E5ECF6"
GRID_COLOR = "#FFFFFF"
UNIT_PLOT_HEIGHT = 500
UNIT_PLOT_WIDTH = 550
