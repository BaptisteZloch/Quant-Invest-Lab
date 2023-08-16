import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Literal, Callable, Optional
from scipy import stats, signal
from scipy.signal import hilbert
from ta.trend import ema_indicator, sma_indicator
from ta.momentum import ppo_hist
from ta.volatility import average_true_range


def zma_indicator(close: pd.Series, window: int = 20) -> pd.Series:
    """ZMA is smoother than SMA due to the integration (cumsum) and then differentiation (slope), which is important for point of change detections. In other words, it generates fewer false signals in technical analysis for trading.

    Args:
    ----
        close (pd.Series): The close series from the OHLCV Data.

        window (int, optional): The indicator window for the rolling window. Defaults to 60.

    Returns:
    ----
        pd.Series: The ZMA indicator.
    """

    def zma(closes: pd.Series) -> float:
        z = closes.cumsum()
        x = np.arange(closes.shape[0])
        return np.corrcoef(x, z)[0, 1] * np.std(z) / np.std(x)

    return close.rolling(window).apply(zma)


def efficiency_ratio_indicator(close: pd.Series, window: int = 60) -> pd.Series:
    """The Kaufman Efficiency Ratio is a ratio of the price direction to the price volatility. The result is a number, which oscillates between +1 and -1. The center point is 0. +1 indicates a financial instrument with a perfectly efficient upward trend.  -1 indicates a financial instrument with a perfectly efficient downward trend. It is virtually impossible for an instrument to have a perfect efficiency ratio (+1 or -1). The more efficient the ratio, the clearer the trend.

    Args:
    ----
        close (pd.Series): The close series from the OHLCV Data.

        window (int, optional): The indicator window for the rolling window. Defaults to 60.

    Returns:
    ----
        pd.Series: The Kauffman efficiency ratio indicator.
    """
    return close.rolling(window).apply(
        lambda prices: ((prices[-1] / prices[0]) - 1)
        / prices.pct_change().fillna(0).abs().sum()
    )


def dynamic_time_wrapping_indicator(
    timeseries_1: pd.Series, timeseries_2: pd.Series, window: int = 20
) -> pd.Series:
    """Calculate the dynamic time warping cost over all dates possible. Inspired from this article : https://medium.datadriveninvestor.com/one-cool-way-to-spot-correlations-in-finance-dynamic-time-warping-763a6e7f0ee3

    Args:
    -----
        timeseries_1 (pd.Series): The first time series.

        timeseries_2 (pd.Series): The second time series.

        window (int, optional): How long the dynamic time-warping comparison is. Defaults to 20.

    Returns:
    -----
        pd.Series: The dynamic time warping cost indicator over all dates possible.
    """

    def get_dynamic_time_wrapping_score(
        ts1: npt.NDArray[np.float64], ts2: npt.NDArray[np.float64]
    ) -> float:
        """Calculate the dynamic time warping cost between two time series

        Args:
        -----
            ts1 (npt.NDArray[np.float64]): The first time series.

            ts2 (npt.NDArray[np.float64]): The second time series.

        Returns:
        -----
            float: The dynamic time warping cost between the two time series.
        """
        C = np.full(
            shape=(ts1.shape[0] + 1, ts2.shape[0] + 1),
            fill_value=np.inf,
        )  # Initialise a full cost matrix, filled with np.inf. This is so we can start the algorithm and not get stuck on the boundary

        C[
            0, 0
        ] = 0  # Place the corner to zero, so that we don't have the minimum of 3 infs

        for i in range(1, ts1.shape[0] + 1):
            for j in range(1, ts2.shape[0] + 1):
                # Euclidian distance between the two points
                dist = abs(ts1[i - 1] - ts2[j - 1])

                # Find the cheapest cost of all three neighbours
                prev_min = min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])

                # Populate the entry in the cost matrix
                C[i, j] = dist + prev_min

        return C[-1, -1]

    z_scale: Callable[[pd.Series], npt.NDArray[np.float64]] = lambda ts: np.array(
        ((ts - ts.mean()) / ts.std()).values, dtype=np.float64
    )  # Lambda function to Z-scale the time series

    scores = (
        pd.Series(data=np.full(timeseries_1.shape, 1), index=timeseries_1.index)
        .rolling(window)
        .apply(
            lambda rows: get_dynamic_time_wrapping_score(
                ts1=z_scale(timeseries_1.loc[rows.index[0] : rows.index[-1]]),
                ts2=z_scale(timeseries_2.loc[rows.index[0] : rows.index[-1]]),
            )
        )
    )

    return scores.fillna(scores.min())


def vertical_horizontal_filter_indicator(
    close: pd.Series, window: int = 60
) -> pd.Series:
    """THe Vertical Horizontal Filter (VHF) is a technical analysis indicator used to determine whether prices are in a trending or non-trending phase. It was introduced by Adam White in an article in the August, 1991 issue of Futures Magazine. The VHF uses the highest high and lowest low of the last n periods to determine whether the market is in a trending phase or a congestion phase.

    Args:
    ----
        close (pd.Series): The close series from the OHLCV Data.

        window (int, optional): The indicator window for the sum and min/max. Defaults to 60.

    Returns:
    ----
        pd.Series: The Vertical Horizontal Filter indicator.
    """
    return (
        close.rolling(window).max() - close.rolling(window).min()
    ) / close.diff().fillna(0).apply(lambda x: abs(x)).rolling(window).sum()


def extreme_euphoria_pattern(
    open: pd.Series,
    close: pd.Series,
    euphoria_side: Literal["bull", "bear"] = "bull",
    n_candles: int = 3,
):
    assert (
        n_candles > 0 and n_candles < 10
    ), "n_candles must be greater than 0 and less than 10"
    open_close_return = close - open  # (close/open) - 1

    def euphoria(close_open: pd.Series) -> int:
        close_open_abs = close_open.abs()
        if euphoria_side == "bear":
            if np.all(np.sign(close_open.to_numpy()) == 1) and all(
                close_open_abs.iloc[i] <= close_open_abs.iloc[i + 1]
                for i in range(len(close_open_abs) - 1)
            ):
                return 0
            return 1
        elif euphoria_side == "bull":
            if np.all(np.sign(close_open.to_numpy()) == -1) and all(
                close_open_abs.iloc[i] <= close_open_abs.iloc[i + 1]
                for i in range(len(close_open_abs) - 1)
            ):
                return 1
            return 0
        else:
            raise ValueError("euphoria_side parameter must be either bull or bear")

    return open_close_return.rolling(n_candles).apply(euphoria)


def hilbert_transform_dominant_cycle_period_indicator(
    close: pd.Series, window: int
) -> pd.Series:
    analytic_signal = pd.Series(np.imag(hilbert(close)), index=close.index)
    instantaneous_phase = pd.Series(
        np.unwrap(np.angle(analytic_signal)), index=close.index
    )

    # Calculate the first differences of the instantaneous phase
    phase_diff = instantaneous_phase.diff().fillna(0)

    def htdcp_calculation(x: float) -> float:
        numerator = np.sum(np.arange(1, window + 1) * x)
        denominator = np.sum(x)
        return numerator / denominator if denominator != 0 else 0

    return phase_diff.rolling(window=window).apply(htdcp_calculation)


def chande_momentum_oscillator(
    high: pd.Series, low: pd.Series, window: int
) -> pd.Series:
    """The Chande oscillator is similar to other momentum indicators such as Wilder's relative strength index (RSI) and the stochastic oscillator. It measures momentum on both up and down days and does not smooth results, triggering more frequent oversold and overbought penetrations. The indicator oscillates between +100 and -100

    Args:
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.
        window (int): The indicator window for the sum. Default to 20.

    Returns:
        pd.Series: The Chande momentum oscillator.
    """
    high_sum = high.rolling(window).sum()
    low_sum = low.rolling(window).sum()
    return 100 * (high_sum - low_sum) / (high_sum + low_sum)


def ulcer_index_indicator(close: pd.Series, high: pd.Series, window: int) -> pd.Series:
    """The Ulcer Index is a volatility indicator that measures downside risk. It was developed by Peter Martin and is based on the idea that investors are more concerned with the downside risk than the upside potential. The Ulcer Index is calculated as follows:
    UI = sqrt(sum[(P(i)/Max(P) — 1)², i=1 to N] / N) * 100
    """
    rolling_max_high = high.rolling(window).max()
    return (
        ((100 * (close - rolling_max_high) / rolling_max_high) ** 2) / window
    ) ** 0.5


def correlation_adjusted_reversal_indicator(
    close: pd.Series, window: int = 10
) -> pd.Series:
    """The idea for the correlation-adjusted reversal indicator is to detect average extremes where the correlation between returns and prices is high enough to justify a possible market inflection.

    Args:
        close (pd.Series): The close series from the OHLCV Data.
        window (int, optional): The indicator window for the correlation calculation. Defaults to 10.

    Returns:
        pd.Series: The correlation adjusted reversal indicator.
    """
    shifted_diff = pd.DataFrame()
    for i in range(1, window + 1):
        shifted_diff = pd.concat(
            [
                shifted_diff,
                pd.Series(close - close.shift(i), name=f"Shift_diff_{i}").fillna(0),
            ],
            axis=1,
        )

    return close.rolling(10).corr(shifted_diff.mean(axis=1)).fillna(0)


def garman_klass_volatility_indicator(
    open: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series, window: int = 20
) -> pd.Series:
    """Calculate the Garman Klass volatility indicator.

    Args:
        open (pd.Series): The open series from the OHLCV Data.
        close (pd.Series): The close series from the OHLCV Data.
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.
        window (int): The indicator window for the moving averages. Default to 20.

    Returns:
        pd.Series: The Garman Klass volatility indicator.
    """
    return ((np.log(high) - np.log(low)) ** 2) / 2 - (2 * np.log(2) - 1) * (
        (np.log(close) - np.log(open)) ** 2
    )


def amihud_illiquidity_indicator(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate the Amihud illiquidity indicator.

    Args:
        close (pd.Series): The close series from the OHLCV Data.
        volume (pd.Series): The volume series from the OHLCV Data.

    Returns:
        pd.Series: The Amihud illiquidity indicator.
    """
    return close.pct_change().abs() / (close * volume)


def equilibrium_indicator(close: pd.Series, window: int = 5) -> pd.Series:
    """Calculate equilibrum indicator aimed to detect mean reversion.

    Args:
        close (pd.Series): The close series from the OHLCV Data.
        window (int): The indicator window for the moving averages. Default to 5.

    Returns:
        pd.Series: The equilibrum indicator.
    """

    return ema_indicator(sma_indicator(close, window).fillna(method="backfill") - close)


def gap_indicator(open: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate the % gap between the current open and the previous close.

    Args:
        open (pd.Series): The open series from the OHLCV Data.
        close (pd.Series): The close series from the OHLCV Data.

    Returns:
        pd.Series: The gap indicator.
    """

    def gap(opens: pd.Series) -> float:
        assert len(opens) == 2, "Error, open_rows must be a series of 2 values."
        prior_close = close.loc[opens.index[0]]
        current_open = opens[-1]
        return 100 * (current_open - prior_close) / prior_close

    return open.rolling(2).apply(gap)


def choppiness_index_indicator(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Compute the choppiness index indicator which indicates whether the market is trending, reversing or ranging. Boundaries are 38.2, 61.8.

    Args:
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.
        close (pd.Series): The close series from the OHLCV Data.
        window (int, optional): The window to calculate the CHOP on a period. Defaults to 20.

    Returns:
        pd.Series: The choppiness index
    """
    return (
        100
        * np.log(
            average_true_range(high, low, close, window + 5).rolling(window).sum()
            / (high.rolling(window).max() - low.rolling(window).min())
        )
        / np.log(window)
    ).fillna(0)


def support_resistance_breakout_indicator(
    support_resistance_levels: list[float], open: pd.Series, low: pd.Series
) -> pd.Series:
    """Generate support resistance breakout rolling indicator.

    Args:
        support_resistance_levels (list[float]): The support and resistance levels.
        open (pd.Series[float | int]): The open series from the OHLCV Data.
        low (pd.Series[float | int]): The low series from the OHLCV Data.

    Returns:
        pd.Series: The S/R breakout rolling indicator.
    """

    def has_breakout(open_rows: pd.Series) -> int:
        """Base on the candle stick, detect a breakout or not.

        Args:
            open_rows (pd.Series): A series of 2 rolling open measures.

        Returns:
            int: 1 = breakout, 0 no breakout.
        """
        assert len(open_rows) == 2, "Error, open_rows must be a series of 2 values."
        last_index = open_rows.index[-1]
        for level in support_resistance_levels:
            if (
                (open_rows[0] < level)
                and (open_rows[-1] > level)
                and (low.loc[last_index] > level)
            ):
                return int(1)
        return int(0)

    return open.rolling(2).apply(has_breakout)


def zlema(
    close: pd.Series,
    window: int = 26,
) -> pd.Series:
    """ZLEMA is an abbreviation of Zero Lag Exponential Moving Average. It was developed by John Ehlers and Rick Way.
    ZLEMA is a kind of Exponential moving average but its main idea is to eliminate the lag arising from the very nature of the moving averages
    and other trend following indicators. As it follows price closer, it also provides better price averaging and responds better to price swings.

    Args:
        close (pd.Series): The close series from the OHLCV Data.
        window (int, optional): The window to calculate the zlema on a period. Defaults to 26.

    Returns:
        pd.Series: The ZLEMA indicator
    """
    return (close + (close.diff((window - 1) // 2))).ewm(span=window).mean()


def fibonacci_moving_average(
    close: pd.Series, fibonacci_n_terms: int = 17
) -> pd.Series:
    terms = tuple(
        [
            2,
            3,
            5,
            8,
            13,
            21,
            34,
            55,
            89,
            144,
            233,
            377,
            610,
            987,
            1597,
            2584,
            4181,
            6765,
            10946,
        ]
    )
    if fibonacci_n_terms > len(terms):
        raise ValueError(
            "Error, provide less fibonacci terms, it must be lower that 17."
        )

    temp_df = pd.DataFrame({"Close": close.values})
    for i in range(fibonacci_n_terms):
        temp_df[f"EMA{terms[i]}"] = temp_df["Close"].ewm(terms[i]).mean()
    temp_df.drop(columns=["Close"], inplace=True)
    return temp_df.apply(lambda row: row.mean(), axis=1)


def fibonacci_distance_indicator(
    close: pd.Series,
    fibonacci_ma: pd.Series,
    normalization: bool = True,
    normalization_window: int = 20,
    ma_smoothing: bool = True,
    ma_smoothing_window: int = 10,
) -> pd.Series:
    """_summary_

    Args:
        close (pd.Series): The market, i.e. often the closing price.
        fibonacci_ma (pd.Series): The pre-computed fibonacci moving average.
        normalization (bool, optional): Whether to normalize the distance indicator with min-max scaling. Defaults to True.
        normalization_window (int, optional): The min-max scaling window. Defaults to 20.
        ma_smoothing (bool, optional):  Whether to smooth or not the distance indicator. True includes the normalization step. Defaults to True.
        ma_smoothing_window (int, optional): The smoothing window. Defaults to 10.

    Returns:
        pd.Series: The fibonacci distance indicator.
    """
    dist = close - fibonacci_ma
    if normalization is True or ma_smoothing is True:
        dist_normalized = (dist - dist.rolling(normalization_window).min()) / (
            dist.rolling(normalization_window).max()
            - dist.rolling(normalization_window).min()
        )
        if ma_smoothing is True:
            return dist_normalized.rolling(ma_smoothing_window).mean()
        else:
            return dist_normalized
    return dist


def fischer_transformation(close: pd.Series, window: int = 5) -> pd.Series:
    """Compute the fischer transformation on a series.

    Args:
        close (pd.Series): The series to transform.
        window (int): The window to compute the fischer transformation on. Default to 5.

    Returns:
        pd.Series: The transformed series.
    """
    close_normalized: pd.Series = (close - close.rolling(window=window).min()) / (
        close.rolling(window=window).max() - close.rolling(window=window).min()
    )
    close_new: pd.Series = (2 * close_normalized) - 1
    smooth: pd.Series = close_new.ewm(span=5, adjust=True).mean()

    return (
        pd.Series((np.log((1 + smooth) / (1 - smooth)))).ewm(span=3, adjust=True).mean()
    )


def min_max_normalization_indicator(close: pd.Series, window: int = 4) -> pd.Series:
    """Indicator based on min max normalization over a specific window.

    Args:
        close (pd.Series): The time series to normalize.
        window (int, optional): The lookback for normalizing the time series. Defaults to 4.

    Returns:
        pd.Series: The normalized time series.
    """
    return (close - close.rolling(window).min()) / (
        close.rolling(window).max() - close.rolling(window).min()
    )


def z_score_indicator(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute the z-score of a time series over a window.

    Args:
        close (pd.Series): The time series to compute the z-score on.
        window (int, optional): The window length of the data used to compute the z-score. Defaults to 14.

    Returns:
        pd.Series: The z-score of the time series.
    """
    return (close - close.rolling(window=window).mean()) / close.rolling(
        window=window
    ).std()


def mean_break_out_indicator(close: pd.Series, window: int = 20) -> pd.Series:
    """Compute the MBO, mean breakout indicator on a time series. It compares the difference between the closing price of a candle and a moving average over N periods to the difference between the min and max value of the closing price over the same N periods.

    Args:
        close (pd.Series): The time series to process.
        window (int, optional): The window length of the data used to compute the MBO. Defaults to 20.

    Returns:
        pd.Series: The MBO indicator.
    """
    return (close - close.ewm(window).mean()) / (
        close.rolling(window).max() - close.rolling(window).min()
    )


def trend_intensity_indicator(close: pd.Series, window: int = 20) -> pd.Series:
    """The Trend Intensity indicator is a measure of the accumulation of the number of periods the price is above the EMA against the price is under the EMA. Given a number of periods equal to the number of periods of the EMA it gives an information on the trend strength and if the trend will tend to reverse.

    Args:
        close (pd.Series): The time series to process.
        window (int, optional): The window length of the data used to compute the trend intensity. Defaults to 20.

    Returns:
        pd.Series: The trend intensity indicator.
    """
    return (
        100
        * pd.DataFrame({"EMA": close.ewm(window).mean().values, "Close": close.values})
        .apply(lambda row: 1 if row["Close"] > row["EMA"] else 0, axis=1)
        .rolling(window)
        .sum()
        / window
    )


def ppo_slope_indicator(close: pd.Series, slope_window: int = 4) -> pd.Series:
    ppo_histo = ppo_hist(close=close)

    def calculate_slope(ppo_values: pd.Series) -> float:
        x = np.arange(1, len(ppo_values) + 1, 1)
        y = np.array(ppo_values)
        m, c = np.polyfit(x, y, 1)
        return m

    return ppo_histo.rolling(slope_window).apply(calculate_slope)


def detect_windowing_support_resistance(high: pd.Series, low: pd.Series) -> list[float]:
    """Detects the supports and resistances according to the windowing method

    Args:
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.

    Returns:
        list[float]: The supports and resistances levels.
    """
    pivots: list[float] = []
    max_list = []
    min_list = []
    for i in range(5, len(high) - 5):
        # taking a window of 9 candles
        high_range = high.iloc[i - 5 : i + 4]
        current_max = high_range.max()
        # if we find a new maximum value, empty the max_list
        if current_max not in max_list:
            max_list = []
        max_list.append(current_max)
        # if the maximum value remains the same after shifting 5 times
        if len(max_list) == 5 and is_far_from_level(
            current_max,
            pivots,
            high,
            low,
        ):
            pivots.append(float(current_max))

        low_range = low[i - 5 : i + 5]
        current_min = low_range.min()
        if current_min not in min_list:
            min_list = []
        min_list.append(current_min)
        if len(min_list) == 5 and is_far_from_level(
            current_min,
            pivots,
            high,
            low,
        ):
            pivots.append(float(current_min))
    return pivots


def detect_fractal_support_resistance(high: pd.Series, low: pd.Series) -> list[float]:
    """Detects the supports and resistances according to the fractal method.

    Args:
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.

    Returns:
        list[float]: The supports and resistances levels.
    """

    def is_fractal_support(low: pd.Series, i: int) -> bool:
        """Determine bullish fractal supports.

        Args:
            low (pd.Series): The low series from the OHLCV Data.
            i (int): The current index.

        Returns:
            bool: Whether it's a support or not according to fractal method.
        """
        return (
            low[i] < low[i - 1]
            and low[i] < low[i + 1]
            and low[i + 1] < low[i + 2]
            and low[i - 1] < low[i - 2]
        )

    def is_fractal_resistance(high: pd.Series, i: int) -> bool:
        """Determine bearish fractal resistances.

        Args:
            low (pd.Series): The high series from the OHLCV Data.
            i (int): The current index.

        Returns:
            bool: Whether it's a resistance or not according to fractal method.
        """
        return (
            high[i] > high[i - 1]
            and high[i] > high[i + 1]
            and high[i + 1] > high[i + 2]
            and high[i - 1] > high[i - 2]
        )

    levels: list[float] = []
    k = 5
    for i in range(k, high.shape[0] - k):
        if is_fractal_support(low, i):
            if is_far_from_level(
                low.iloc[i],
                levels,
                high,
                low,
            ):
                levels.append(float(low.iloc[i]))
        elif is_fractal_resistance(high, i):
            if is_far_from_level(
                high.iloc[i],
                levels,
                high,
                low,
            ):
                levels.append(float(high.iloc[i]))

    return levels


def detect_profiled_support_resistance(
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    kde_factor: float = 0.075,
    total_levels: str | int = "all",
) -> pd.DataFrame:
    """Detect the support and resistance level over an historical time period using price and volume.

    Args:
        close (pd.Series[float | int]): The price history usually closing price.
        volume (pd.Series[float | int], optional): The volume history. Defaults to None.
        kde_factor (float, optional):  The coefficient used to calculate the estimator bandwidth. The higher coefficient is the strongest levels will only be detected. Defaults to 0.075.
        total_levels (str | int, optional): The total number of levels to detect. If "all" is provided, all levels will be detected. Defaults to "all".

    Returns:
        pd.DataFrame: The DataFrame containing the levels [min price, max price] and weights [0, 1] associated with each.
    """
    if volume is not None:
        assert len(close) == len(
            volume
        ), "Error, provide same size price and volume Series."

    # Generate a number of sample of the complete price history range in order to apply density estimation.
    xr = np.linspace(start=close.min(), stop=close.max(), num=len(close))

    # Generate the kernel density estimation of the price weighted by volume over a certain number of sample xr.
    # It's possible to interpolate less precisely with decreasing the num parameter above.
    if volume is not None:
        estimated_density = stats.gaussian_kde(
            dataset=close, weights=volume, bw_method=kde_factor
        )(xr)
    else:
        estimated_density = stats.gaussian_kde(dataset=close, bw_method=kde_factor)(xr)

    def min_max_scaling(
        to_scale_array: npt.NDArray[np.float64],
        min_limit: int = 0,
        max_limit: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Min max scaling between 0 and 1.

        Args:
            to_scale_array (npt.NDArray[np.float64]): The array to scale.
            min_limit (int, optional): The lower limit of the range. Defaults to 0.
            max_limit (int, optional): The higher limit of the range. Defaults to 1.

        Returns:
            npt.NDArray[np.float64]: The scaled array.
        """
        return (to_scale_array - to_scale_array.min(axis=0)) / (
            to_scale_array.max(axis=0) - to_scale_array.min(axis=0)
        ) * (max_limit - min_limit) + min_limit

    # Find the index of the peaks over on a signal, here the estimated density.
    peaks, _ = signal.find_peaks(estimated_density)

    levels = xr[peaks]
    weights = min_max_scaling(estimated_density[peaks])

    df = pd.DataFrame({"levels": levels, "weights": weights})

    if isinstance(total_levels, int):
        assert (
            total_levels > 0
        ), "Error, provide a positive not null value for the total_levels parameter."

        if total_levels < len(levels):
            return df.sort_values(by="weights", ascending=False).head(total_levels)

    return df


def is_far_from_level(
    value: float, levels: list[float], high: pd.Series, low: pd.Series
) -> bool:
    """Function to make sure the new level area does not exist already

    Args:
        value (float): The value to check with the existing levels.
        levels (list[float]): The list of levels.
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.

    Returns:
        bool: Whether the level is far from the others or not.
    """
    return np.sum([abs(value - level) < np.mean(high - low) for level in levels]) == 0
