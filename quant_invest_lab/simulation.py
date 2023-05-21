from datetime import datetime
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Literal
import matplotlib.pyplot as plt
from tqdm import tqdm
from fitter import Fitter
from psutil import cpu_count

from quant_invest_lab.constants import (
    TIMEFRAME_ANNUALIZED,
    TIMEFRAME_TO_FREQ,
    TIMEFRAMES,
)
from quant_invest_lab.types import Timeframe


def generate_brownian_paths(
    n_paths: int,
    n_steps: int,
    T: float | int,
    mu: float | int,
    sigma: float | int,
    s0: float | int,
    brownian_type: Literal["ABM", "GBM"] = "ABM",
    get_time: bool = True,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.log(s0)

    for i in tqdm(
        range(0, n_paths), desc="Simulating Brownian Motion paths...", leave=False
    ):
        for j in tqdm(
            range(n_steps),
            desc="Simulating Brownian Motion path's steps...",
            leave=False,
        ):
            paths[i, j + 1] = (
                paths[i, j]
                + (mu * 0.5 * sigma**2) * dt
                + sigma * np.random.normal(0, np.sqrt(dt))
            )

    if brownian_type == "GBM":
        paths = np.exp(paths)

    return np.linspace(0, T, n_steps + 1), paths if get_time is True else paths


def account_evolution(risk_free_rate: float, time: float) -> float:
    return np.exp(risk_free_rate * time)


def plot_paths_with_distribution(
    time: npt.NDArray[np.float64],
    paths: npt.NDArray[np.float64],
    title: str = "Simulated Brownian Motion",
) -> None:
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(title)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for y in paths:
        ax1.plot(time, y)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{paths.shape[0]} simulations")
    ax1.grid(True)
    ax2.hist(
        paths[:, -1],
        density=True,
        bins=75,
        facecolor="blue",
        alpha=0.3,
        label="Frequency of X(T)",
    )
    ax2.hist(
        paths[:, paths.shape[-1] // 4],
        density=True,
        bins=75,
        facecolor="red",
        alpha=0.3,
        label="Frequency of X(T/4)",
    )
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution at X(T) and X(T/4)")
    ax2.legend()
    ax2.grid(True)


def generate_open(
    open_divided_by_close: pd.Series, generated_close: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    f = Fitter(
        open_divided_by_close.values,
        distributions=["laplace"],
    )
    f.fit(max_workers=cpu_count(), n_jobs=cpu_count(), progress=False)
    return (
        np.random.laplace(
            f.fitted_param["laplace"][0],
            f.fitted_param["laplace"][-1],
            generated_close.shape[0],
        )
    ) * generated_close


def generate_high_coef(
    high_from_body: pd.Series, n: int = 1000
) -> npt.NDArray[np.float64]:
    f = Fitter(
        high_from_body.values,
        distributions=["expon"],
    )
    f.fit(max_workers=cpu_count(), n_jobs=cpu_count(), progress=False)

    return 1 + np.random.exponential(f.fitted_param["expon"][-1], n)


def generate_low_coef(
    low_from_body: pd.Series, n: int = 1000
) -> npt.NDArray[np.float64]:
    f = Fitter(
        low_from_body.values,
        distributions=["expon"],
    )
    f.fit(max_workers=cpu_count(), n_jobs=cpu_count(), progress=False)
    return np.random.exponential(f.fitted_param["expon"][-1], n)


def generate_brownian_candle_from_dataframe(
    dataframe: pd.DataFrame,
    timeframe: Timeframe = "1day",
    generated_dataframe_length: int = 5000,
    date_index: bool = True,
    compute_return: bool = True,
) -> pd.DataFrame:
    """Generate a new dataframe with the same structure (global continuous distribution) as the initial dataframe but with generated data using a geometric brownian motion.

    Args:
    ----
        dataframe (pd.DataFrame): The initial dataframe, it should be a real dataframe containing OHLC data. I must contains OHLC columns. This dataframe will to extract the mean return, the std return, the initial price and the distribution of the open, high and low prices.

        timeframe (Timeframe, optional): The timeframe or data interval frequency, it's used to compute the period parameter. Defaults to "1day".

        generated_dataframe_length (int, optional): The number of record to generate in the new dataframe. Defaults to 5000.

        date_index (bool, optional): Create as of today date index. Defaults to True.

        compute_return (bool, optional): Compute the returns of the generated dataframe. Defaults to True.
    Returns:
    ----
        pd.DataFrame: The generated dataframe and it's candle stick.
    """
    assert (
        timeframe in TIMEFRAMES
    ), f"timeframe must be one of the following: {','.join(TIMEFRAMES)}"
    assert isinstance(dataframe, pd.DataFrame), "dataframe must be a pandas DataFrame"
    assert set(dataframe.columns).issuperset(
        ["Open", "High", "Low", "Close"]
    ), "dataframe must contains Open, High, Low, Close columns"
    assert (
        generated_dataframe_length >= 10
    ), "generated_dataframe_length must be greater than 10"

    if not "Returns" in dataframe.columns:
        dataframe["Returns"] = dataframe.Close.pct_change().fillna(0)

    dataframe["High_from_body"] = dataframe.apply(
        lambda row: row["High"] / row["Close"]
        if row["Open"] <= row["Close"]
        else row["High"] / row["Open"],
        axis=1,
    )

    dataframe["Low_from_body"] = dataframe.apply(
        lambda row: 1 - (row["Low"] / row["Close"])
        if row["Open"] >= row["Close"]
        else 1 - (row["Low"] / row["Open"]),
        axis=1,
    )

    dataframe["Open_by_close"] = dataframe["Open"] / dataframe["Close"]

    time, sim = generate_brownian_paths(
        1,
        generated_dataframe_length - 1,
        generated_dataframe_length / TIMEFRAME_ANNUALIZED.get(timeframe, 365),
        dataframe.Returns.mean() * TIMEFRAME_ANNUALIZED.get(timeframe, 365),
        dataframe.Returns.std() * TIMEFRAME_ANNUALIZED.get(timeframe, 365) ** 0.5,
        dataframe.Close.iloc[0],
        "GBM",
    )

    generated_close = sim[-1, :]
    generated_open = generate_open(dataframe["Open_by_close"], generated_close)

    generated_high_coefs = generate_high_coef(
        dataframe["High_from_body"], generated_dataframe_length
    )
    generated_low_coefs = generate_low_coef(
        dataframe["Low_from_body"], generated_dataframe_length
    )

    generated_ohlc = pd.DataFrame(
        {
            "Open": generated_open,
            "High_from_body": generated_high_coefs,
            "Low_from_body": generated_low_coefs,
            "Close": generated_close,
        }
    )

    generated_ohlc["High"] = generated_ohlc.apply(
        lambda row: row["High_from_body"] * row["Close"]
        if row["Open"] <= row["Close"]
        else row["High_from_body"] * row["Open"],
        axis=1,
    )

    generated_ohlc["Low"] = generated_ohlc.apply(
        lambda row: (1 - row["Low_from_body"]) * row["Close"]
        if row["Open"] >= row["Close"]
        else (1 - row["Low_from_body"]) * row["Open"],
        axis=1,
    )
    if date_index:
        generated_ohlc.set_index(
            pd.date_range(
                end=datetime.now(),
                periods=generated_dataframe_length,
                freq=TIMEFRAME_TO_FREQ[timeframe],
                normalize=True,
                name="Date",
            ),
            inplace=True,
        )
    if compute_return:
        generated_ohlc["Returns"] = generated_ohlc.Close.pct_change().fillna(0.0)
    return generated_ohlc.drop(columns=["High_from_body", "Low_from_body"])
