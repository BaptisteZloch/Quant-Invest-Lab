# Quant Invest Lab
**Quant Invest Lab** is a project aimed to provide a set of basic tools for quantitative experiments. By quantitative experiment I mean trying to build you own set of investments solution. The project is still in its early stage, but I hope it will grow in the future.

Initially this project was aimed to be a set of tools for my own experiments, but I decided to make it open source. Of courses it already exists some awesome packages, more detailed, better suited for some use cases. But I hope it will be useful for someone else. Feel free to use it, modify it and contribute to it.*
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
pip install quant-invest-lab -U
```
You can install it using poetry the same way :
```bash
poetry add quant-invest-lab
```

# Basic examples
## Backtest a basic EMA crossover strategy
```python
``` 
## Disclaimer
This package is only for educational purpose or experimentation it is not intended to be used in production. I am not responsible for any loss of money you may have using this package. Use it at your own risk.