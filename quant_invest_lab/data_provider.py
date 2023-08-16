from datetime import datetime
from functools import lru_cache
from typing import Optional, Callable, Literal
import os
import pandas as pd
import json
import time
from random import randint
from kucoin.client import Market
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from quant_invest_lab.constants import TIMEFRAME_IN_S, TIMEFRAMES
from quant_invest_lab.types import Timeframe


def build_multi_crypto_dataframe(
    symbols: set,
    drop_na: bool = False,
    column_to_keep: Literal[
        "Close", "Open", "High", "Low", "Volume", "Amount"
    ] = "Close",
    timeframe: Timeframe = "1day",
) -> pd.DataFrame:
    """Build a multi cryptocurrencies dataframe from a set of symbols. This dataframe will contain the closing price of each symbol or any other column specified in `column_to_keep`. This function is useful to build a dataframe for a multi cryptocurrencies portfolio.

    Args:
    -----
        symbols (set): The name of the symbols to download.

        drop_na (bool, optional): Whether or not to drop NaNs. Defaults to False.

        column_to_keep (Literal[&quot;Close&quot;,&quot;Open&quot;,&quot;High&quot;,&quot;Low&quot;,&quot;Volume&quot;,&quot;Amount&quot;], optional): The name of the column to keep. Defaults to "Close".

        timeframe (Timeframe, optional): The data granularity. Defaults to "1day".

    Returns:
    -----
        pd.DataFrame: The multi cryptocurrencies dataframe.
    """
    df = pd.DataFrame()
    for symbol in tqdm(symbols, desc="Fetching symbols..."):
        currency_df = download_crypto_historical_data(
            symbol, timeframe, compute_return=False, refresh_list_of_symbol=False
        )
        # currency_df.index = currency_df.index.map(
        #     lambda x: datetime(x.year, x.month, x.day)
        # )
        assert (
            column_to_keep in currency_df.columns
        ), f"Column {column_to_keep} not in {symbol} dataframe. Please choose another column."

        df = pd.concat(
            [
                df,
                currency_df[[f"{column_to_keep}"]].rename(
                    columns={f"{column_to_keep}": f"{symbol}_{column_to_keep}"}
                ),
            ],
            axis=1,
        )
    return df if not drop_na else df.dropna()


def download_crypto_historical_data(
    symbol: str,
    timeframe: Timeframe = "1hour",
    compute_return: bool = True,
    refresh_list_of_symbol: bool = True,
    keep_timestamps_col: bool = False,
) -> pd.DataFrame:
    """Fetch a cryptocurrency historical data from Kucoin for a given symbol and timeframe.

    Args:
    ----
        symbol (str): The symbol to download.

        timeframe (Timeframe, optional): The timeframe of the data to download. Defaults to "1hour".

        compute_return (bool, optional): Whether or not to compute the return on close T and T-1. Defaults to True.

        refresh_list_of_symbol (bool, optional): Refresh the list of symbols/ticker from kucoin, it must be true for the first execution of the code. Defaults to True.

        keep_timestamps_col (bool, optional): Whether or not to keep the timestamps. Defaults to False.

    Returns:
    ----
        pd.DataFrame: The historical data of the symbol with date index
    """
    service = CryptoService()
    if refresh_list_of_symbol is True:
        service.refresh_list_of_symbols()  # Uncomment for the first usage
    df = service.get_history_of_symbol(f"{symbol}", timeframe)
    df["Date"] = df["Timestamp"].apply(datetime.fromtimestamp)
    df = df.set_index("Date")
    df = df.loc[~df.index.duplicated(), :]
    df = df.sort_index()

    if compute_return is True:
        df["Returns"] = df["Close"].pct_change()
    if keep_timestamps_col is False:
        df = df.drop(columns=["Timestamp"])

    df = df.fillna(0.0)
    return df  # .asfreq(TIMEFRAME_TO_FREQ[timeframe])#.ffill()


class CryptoService:
    class KucoinDataFetcher:
        __client = Market(url="https://api.kucoin.com")

        def __construct_timestamp_list(
            self,
            start_timestamp: int,
            end_timestamp: int,
            timeframe: Timeframe,
            exchange_limit: int = 1500,
        ) -> list[int]:
            """Private function that generates a list of timestamps spaced of `exchange_limit` times `timeframe`.
            Args:
                start_timestamp (str): The initial timestamp.
                end_timestamp (str): The final timestamp.
                timeframe (Timeframe): The desired timeframe, sould be 1min, 3min, 5min, 15min, 1hour, 4hour, 1day...
                exchange_limit (int, optional): The exchange limit : 1500 for Kucoin here.. Defaults to 1500.
            Returns:
                list[int]: The list of timestamps.
            """
            remaining = (end_timestamp - start_timestamp) // TIMEFRAME_IN_S[timeframe]

            timestamp_i = end_timestamp
            timestamps = [timestamp_i]

            while remaining > exchange_limit:
                timestamp_i = timestamp_i - TIMEFRAME_IN_S[timeframe] * exchange_limit
                remaining = remaining - exchange_limit
                timestamps.append(timestamp_i)

            timestamps.append(start_timestamp)

            return sorted(timestamps, reverse=True)

        @staticmethod
        def __handle_429_error(func: Callable) -> Callable:
            """Static private decorator that handle error 429 response from Kucoin API.
            Args:
                func (Callable): The function to wrap.
            Returns:
                Callable: The wrapper.
            """

            def wrapper(*args, **kwargs):
                passed = False
                while passed == False:
                    try:
                        return func(*args, **kwargs)
                    except:
                        time.sleep(randint(10, 100) / 100)
                        pass

            return wrapper

        @__handle_429_error
        def __get_data(
            self,
            symbol: str,
            start_at: int,
            end_at: int,
            timeframe: Timeframe = "15min",
        ) -> pd.DataFrame:
            """Private function that uses Kucoin API to get the data for a specific symbol and timeframe.
            Args:
                symbol (str): The symbol for the data we want to extract. Defaults to "BTC-USDT".
                start_at (int): The starting timestamp and. Note that this function could only outputs 1500 records. If the timeframe and the timestamps don't satisfy it, it will return a dataframe with 1500 records from the starting timestamp.
                end_at (int): The ending timestamp.
                timeframe (Timeframe, optional): The timeframe, it must be 1min, 3min, 5min, 15min, 1hour, 4hour, 1day... Defaults to '15min'.
            Returns:
                Optional[pd.DataFrame]: The dataframe containing historical records.
            """
            klines = self.__client.get_kline(
                f"{symbol}", timeframe, startAt=start_at, endAt=end_at
            )
            df = pd.DataFrame(
                klines,
                columns=[
                    "Timestamp",
                    "Open",
                    "Close",
                    "High",
                    "Low",
                    "Amount",
                    "Volume",
                ],
                dtype=float,
            )
            df["Timestamp"] = df["Timestamp"].astype(int)

            return df

        def get_symbols(self) -> list[str]:
            """Get a list of all symbols in kucoin
            Returns:
                list[str]: The symbol's list.
            """
            tickers = self.__client.get_all_tickers()
            if tickers is not None:
                return [tik["symbol"] for tik in tickers["ticker"]]
            raise ValueError("Error, no symbols found.")

        def download_history(
            self, symbol: str, since: str, timeframe: Timeframe, jobs: int = -1
        ) -> pd.DataFrame:
            """Download a set of historical data and save it.
            Args:
                symbol (str): The symbol for the data we want to extract. Defaults to "BTC-USDT".
                since (str): The initial date in format : dd-mm-yyyy.
                timeframe (Timeframe): The timeframe, it must be 1min, 3min, 5min, 15min, 1hour, 4hour, 1day... Defaults to '15min'.
                jobs (int, optional): The number of thread to parallelize the code. Defaults to -1.
            Raises:
                ValueError: Error in using parallelism.
            Returns:
                pd.DataFrame: The dataframe containing historical records.
            """
            try:
                start_timestamp = int(datetime.strptime(since, "%d-%m-%Y").timestamp())
            except:
                raise ValueError(
                    "Error, wrong date format, provide something in this format: dd-mm-yyyy"
                )
            end_timestamp = int(datetime.now().timestamp())

            assert (
                start_timestamp < end_timestamp
            ), "Error, the starting timestamp must be less than ending timestamp"

            timestamps = self.__construct_timestamp_list(
                start_timestamp, end_timestamp, timeframe
            )

            df = pd.DataFrame(
                columns=[
                    "Timestamp",
                    "Open",
                    "Close",
                    "High",
                    "Low",
                    "Amount",
                    "Volume",
                ],
                dtype=float,
            )
            if jobs == -1 or jobs == 1:
                for i in range(len(timestamps) - 1):
                    df = pd.concat(
                        [
                            df,
                            self.__get_data(
                                symbol, timestamps[i + 1], timestamps[i], timeframe
                            ),
                        ]
                    )

            elif jobs > 1 and jobs <= 25:
                with ThreadPoolExecutor(
                    max_workers=(len(timestamps) if len(timestamps) <= 25 else jobs)
                ) as executor:
                    processes = [
                        executor.submit(
                            self.__get_data,
                            symbol,
                            timestamps[i + 1],
                            timestamps[i],
                            timeframe,
                        )
                        for i in range(len(timestamps) - 1)
                    ]

                for task in as_completed(processes):
                    df = pd.concat([df, task.result()])
            else:
                raise ValueError(
                    "Error, jobs must be between 25 and 2 to use parallelism or -1 and 1 to do it sequentially."
                )

            df = df.sort_values(by="Timestamp")
            df.drop_duplicates(inplace=True)
            return df

    __kucoin_fetcher = KucoinDataFetcher()
    __base_dir = "./data/"
    __absolute_start_date = "01-01-2017"

    def get_list_of_symbols(
        self, base_currency: Optional[str] = None, quote_currency: Optional[str] = None
    ) -> list[str]:
        """Return the list of symbol in the database's file.
        Args:
            base_currency (Optional[str]): Filter by base currency.
            quote_currency (Optional[str]): Filter by quote currency.
        Raises:
            ValueError: If there is an error.
        Returns:
            list[str]: The list of symbols.
        """
        symbols = self.__open_symbols_list()
        if base_currency == None and quote_currency == None:
            return symbols
        elif base_currency == None and quote_currency is not None:
            return list(
                filter(
                    lambda symbol: symbol.split("-")[-1] == quote_currency.upper(),
                    symbols,
                )
            )
        elif quote_currency == None and base_currency is not None:
            return list(
                filter(
                    lambda symbol: symbol.split("-")[0] == base_currency.upper(),
                    symbols,
                )
            )
        elif quote_currency is not None and base_currency is not None:
            return list(
                filter(
                    lambda symbol: symbol.split("-")[0] == base_currency.upper()
                    and symbol.split("-")[-1] == quote_currency.upper(),
                    symbols,
                )
            )
        raise ValueError("Error, wrong parameters.")

    def get_history_of_symbol(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        """Function that get the history of a symbol.
        Args:
            symbol (str): The crypto symbol file.
            timeframe (Timeframe): The history's timeframe.
        Returns:
            pd.DataFrame: Records corresponding to the history.
        """
        assert (
            timeframe in TIMEFRAMES
        ), f"Error, timeframe must be in {','.join(TIMEFRAMES)}"

        assert (
            symbol in self.get_list_of_symbols()
        ), f"Error, wrong symbol {symbol}, provide something like 'BTC-USDT'."
        return self.__refresh_or_download(symbol, timeframe)

    @lru_cache(maxsize=32, typed=True)
    def refresh_list_of_symbols(self) -> None:
        """Function that refreshes the database's crypto listing."""
        self.__init_directories()
        with open(f"{self.__base_dir}list_available/crypto_available.json", "w") as f:
            json.dump({"listing": self.__kucoin_fetcher.get_symbols()}, f)

    def check_file_exists(self, symbol: str, timeframe: Timeframe) -> bool:
        """Verify if the history has already been fetched
        Args:
            symbol (str): The crypto symbol file.
            timeframe (Timeframe): The history's timeframe.
        Returns:
            bool: Whether or not the history is present.
        """
        return os.path.exists(f"{self.__base_dir}{timeframe}/{symbol}.csv")

    def __open_symbols_list(self) -> list[str]:
        """Private function that opens, reads and returns the list of symbols from the database's file.
        Returns:
            list[str]: The list of symbols.
        """
        with open(f"{self.__base_dir}list_available/crypto_available.json", "r") as f:
            return json.load(f)["listing"]

    def __refresh_or_download(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        """Private function used to refresh or download the history of a symbol.
        Args:
            symbol (str): The crypto symbol file.
            timeframe (Timeframe): The history's timeframe.
        Returns:
            pd.DataFrame: The complete history refreshed or not.
        """
        if self.check_file_exists(symbol, timeframe):
            # print("History present -> Checking for refresh...")
            data = pd.read_csv(
                f"{self.__base_dir}{timeframe}/{symbol}.csv",
                sep=",",
                dtype=float,
            )
            last_timestamp = data["Timestamp"].iloc[-1]

            since = datetime.fromtimestamp(last_timestamp).strftime("%d-%m-%Y")
            # print(f"Last timestamp : {last_timestamp} = {since}")
            if since == datetime.now().strftime("%d-%m-%Y"):
                return data
            new_data = self.__kucoin_fetcher.download_history(
                symbol, since, timeframe, 16
            )

            data = pd.concat([data, new_data]).drop_duplicates(ignore_index=True)
            data["Timestamp"] = data["Timestamp"].astype(int)
            data.to_csv(
                f"{self.__base_dir}{timeframe}/{symbol}.csv", sep=",", index=False
            )
            # print("Finished")
            return data
        else:  # Download full history
            # print("No history -> Downloading full history...")
            data = self.__kucoin_fetcher.download_history(
                symbol, self.__absolute_start_date, timeframe, 16
            )
            data["Timestamp"] = data["Timestamp"].astype(int)
            data.to_csv(
                f"{self.__base_dir}{timeframe}/{symbol}.csv", sep=",", index=False
            )
            # print("Finished")
            return data

    def __init_directories(self) -> None:
        """Private function that init all directories."""
        for tf in TIMEFRAMES:
            os.makedirs(f"{self.__base_dir}{tf}", exist_ok=True)
        os.makedirs(f"{self.__base_dir}list_available", exist_ok=True)
