import pandas as pd
import numpy as np
from ta.momentum import rsi, roc, ppo_hist
from ta.trend import ema_indicator


class TechnicalScreener:
    @staticmethod
    def __calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed to do the technical scoring.

        Args:
        ----
            df (pd.DataFrame): The dataframe to calculate the indicators on that needs to have the following columns: Open, High, Low, Close, Volume.

        Returns:
        ----
            pd.DataFrame: The dataframe with the calculated indicators.
        """
        assert df.shape[0] > 1, "Dataframe must have at least 2 rows"
        assert set({"Open", "High", "Low", "Close"}).issubset(
            df.columns
        ), "Dataframe must have columns Open, High, Low, Close"
        df_clone = df.copy()
        #  Long-term
        df_clone["EMA_200"] = ema_indicator(df_clone["Close"], 200, fillna=True)
        df_clone["EMA_200_CLOSE_PC"] = (df_clone["Close"] / df_clone["EMA_200"]) * 100
        df_clone["ROC_125"] = roc(df_clone["Close"], 125, fillna=True)
        #  Mid-term
        df_clone["EMA_50"] = ema_indicator(df_clone["Close"], 50, fillna=True)
        df_clone["EMA_50_CLOSE_PC"] = (df_clone["Close"] / df_clone["EMA_50"]) * 100
        df_clone["ROC_20"] = roc(df_clone["Close"], 20, fillna=True)
        # Short-term
        df_clone["PPO_HIST"] = ppo_hist(
            df_clone["Close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
            fillna=True,
        )
        #  Calculate PPO histogram slope
        df_clone["PPO_HIST_SLOPE"] = (
            df_clone["PPO_HIST"]
            .rolling(3)
            .apply(
                lambda rows: np.polyfit(
                    np.arange(1, len(rows) + 1, 1), np.array(rows.values), 1
                )[0]
            )
            .fillna(0)
        )
        df_clone["RSI"] = rsi(df_clone["Close"], window=14, fillna=True)
        return df_clone

    @staticmethod
    def __calculate_weights(dataframe: pd.DataFrame) -> pd.Series:
        """Calculate the indicators and the weights for each indicator.

        Args:
        ----
            dataframe (pd.DataFrame): The dataframe to calculate the indicators on that needs to have the following columns: Open, High, Low, Close, Volume.

        Returns:
        ----
            pd.Series: The calculated weighted score.
        """
        df = TechnicalScreener.__calculate_indicators(dataframe)
        #  Long-term
        df["EMA_200_CLOSE_PC_WEIGHTED"] = df["EMA_200_CLOSE_PC"] * 0.3
        df["ROC_125_WEIGHTED"] = df["ROC_125"] * 0.3
        #  Mid-term
        df["EMA_50_CLOSE_PC_WEIGHTED"] = df["EMA_50_CLOSE_PC"] * 0.15
        df["ROC_20_WEIGHTED"] = df["ROC_20"] * 0.15
        #  Short-term
        df["RSI_WEIGHTED"] = df["RSI"] * 0.05
        df["PPO_HIST_SLOPE_WEIGHTED"] = 0
        df.loc[df["PPO_HIST_SLOPE"] < -1, "PPO_HIST_SLOPE_WEIGHTED"] = 0
        df.loc[df["PPO_HIST_SLOPE"] >= -1, "PPO_HIST_SLOPE_WEIGHTED"] = (
            (df["PPO_HIST_SLOPE"] + 1) * 50 * 0.05
        )
        df.loc[df["PPO_HIST_SLOPE"] > 1, "PPO_HIST_SLOPE_WEIGHTED"] = 5

        return (
            df["EMA_200_CLOSE_PC_WEIGHTED"]
            + df["ROC_125_WEIGHTED"]
            + df["EMA_50_CLOSE_PC_WEIGHTED"]
            + df["ROC_20_WEIGHTED"]
            + df["RSI_WEIGHTED"]
            + df["PPO_HIST_SLOPE_WEIGHTED"]
        )

    @classmethod
    def score(cls, df: pd.DataFrame) -> pd.Series:
        return cls.__calculate_weights(dataframe=df)
