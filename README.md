# Quant Invest Lab
<p align="left">
<a href="https://pypi.org/project/quant-invest-lab/"><img alt="PyPI" src="https://img.shields.io/pypi/v/quant-invest-lab"></a>
<a><img alt="commit update" src="https://img.shields.io/github/last-commit/BaptisteZloch/Quant-Invest-Lab"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://codeclimate.com/github/BaptisteZloch/Quant-Invest-Lab"><img alt="Code Climate" src="https://codeclimate.com/github/BaptisteZloch/Quant-Invest-Lab/badges/gpa.svg"></a>
<a href="https://github.com/BaptisteZloch/Quant-Invest-Lab/blob/master/.github/workflows/python-publish.yml"><img alt="GitHub Actions CI" src="https://github.com/BaptisteZloch/Quant-Invest-Lab/actions/workflows/python-publish.yml/badge.svg"></a>

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/BaptisteZloch/Quant-Invest-Lab/issues)



**Quant Invest Lab** is a project aimed to provide a set of basic tools for quantitative experiments. By quantitative experiment I mean trying to build you own set of investments solution. The project is still in its early stage, but I hope it will grow in the future. 

Initially this project was aimed to be a set of tools for my own experiments, but I decided to make it open source. Of courses it already exists some awesome packages, more detailed, better suited for some use cases. But I hope it will be useful for someone else (learn, practice, understand and create). Feel free to use it, modify it and contribute to it. This package is basically the package I wanted to find when I started to learn quantitative finance.
## Main features
- **Data**: download data from external data provider without restriction on candle stick, the main provider is kucoin for now (currently only crypto data are supported).
- **Backtesting**: backtest your trading strategy (Long only for now but soon short and leverage) on historical data for different timeframe. Optimize you take profit, stop loss. Access full metrics of your strategy.
- **Indicators**: a set of indicators to help you build your strategy.
- **Portfolio**: a set of portfolio optimization tools to help you build your portfolio.
- **Simulation**: simulate your data based on real data using statistics to get a better understanding of its behavior during backtesting.
- **Metrics**: a set of metrics to help you evaluate your strategy through performances and risks.

## Installation
To install **Quant Invest Lab** through pip, run the following command:
```bash
pip install quant-invest-lab --upgrade
```
You can install it using poetry the same way :
```bash
poetry add quant-invest-lab
```

# Basic examples
## Backtest a basic EMA crossover strategy
```python
import pandas as pd

from quant_invest_lab.backtest import ohlc_long_only_backtester
from quant_invest_lab.data_provider import download_crypto_historical_data

symbol = "BTC-USDT"
timeframe = "4hour"
df_BTC = download_crypto_historical_data(symbol, timeframe)

# Define your indicators
df_BTC["EMA20"] = df_BTC.Close.ewm(20).mean()
df_BTC["EMA60"] = df_BTC.Close.ewm(60).mean()

df_BTC = df_BTC.dropna()

# Define your strategy entry and exit functions
def buy_func(row: pd.Series, prev_row: pd.Series) -> bool:
    return True if row.EMA20 > row.EMA60 else False

def sell_func(row: pd.Series, prev_row: pd.Series, trading_days: int) -> bool:
    return True if row.EMA20 < row.EMA60 else False

# Backtest your strategy
ohlc_long_only_backtester(
    df=df_BTC,
    long_entry_function=buy_func,
    long_exit_function=sell_func,
    timeframe=timeframe,
    initial_equity=1000,
)

``` 

## Optimize a portfolio (mean-variance)
```python
from quant_invest_lab.portfolio import MonteCarloPortfolio, ConvexPortfolio, RiskParityPortfolio
from quant_invest_lab.data_provider import build_multi_crypto_dataframe

symbols = set(
    [
        "BNB-USDT",
        "BTC-USDT",
        "NEAR-USDT",
        "ETH-USDT",
        "SOL-USDT",
        "EGLD-USDT",
        "ALGO-USDT",
        "FTM-USDT",
        "ADA-USDT",
    ]
)

closes = build_multi_crypto_dataframe(symbols)
returns = closes.pct_change().dropna()

cvx_ptf = ConvexPortfolio(returns)

cvx_ptf.fit("sharpe", "max", max_asset_weight=0.2) # maximize sharpe ratio with a max weight of 20% per asset

cvx_ptf.get_allocation()

# or
mc_ptf = MonteCarloPortfolio(returns)

mc_ptf.fit(n_portfolios=20000, plot=True)

mc_ptf.get_allocation("sharpe", "max") # maximize sharpe ratio

``` 
## Next steps
- Create official docs and add more examples
- Short, leverage and margin backtesting
- Add more data provider (Stock, bonds...)
- Make montecarlo candle data generation process more realistic
## Disclaimer
This package is only for educational purpose or experimentation it is not intended to be used in production. I am not responsible for any loss of money you may have using this package. Use it at your own risk.