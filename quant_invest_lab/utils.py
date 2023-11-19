from sklearn.decomposition import PCA
from functools import lru_cache
import numpy as np
import numpy.typing as npt
from bokeh.palettes import all_palettes
from typing import Literal
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

CAC40_URL = "https://companiesmarketcap.com/cac-40/largest-companies-by-market-cap/"
AMERICAN_LARGEST_COMPANIES = (
    "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/"
)


@lru_cache(maxsize=128, typed=True)
def get_cac40_symbols() -> list[str]:
    soup = BeautifulSoup(requests.get(CAC40_URL).content, "lxml")
    raw_codes = soup.find_all("div", class_="company-code")
    return list(map(lambda input: str(re.sub("<[^>]*>", "", str(input))), raw_codes))


@lru_cache(maxsize=128, typed=True)
def get_largest_symbols_by_country_normal(n: int = 500) -> list[str]:
    n_per_page = 100
    assert (
        n >= n_per_page and n <= 3571
    ), "Error, provide a valid number of symbol, <= 3571 and >= 100."

    raw_codes = []
    pages = (n // n_per_page) + 1

    for i in range(pages):
        soup = BeautifulSoup(
            requests.get(f"{AMERICAN_LARGEST_COMPANIES}?page={i+1}").content, "lxml"
        )
        raw_codes.extend(soup.find_all("div", class_="company-code"))
    return list(map(lambda input: str(re.sub("<[^>]*>", "", str(input))), raw_codes))[
        :n
    ]


def reduce_dimentionality(
    full_dataframe: pd.DataFrame,
    mode: Literal["pca", "tsne"] = "pca",
    target_explained_variance: float = 0.95,
    result_as_df: bool = True,
) -> npt.NDArray[np.float64] | pd.DataFrame:
    """Reduce the dimension of a dataset given a target explained variance.

    Args:
    -----
        full_dataframe (pd.DataFrame): The dataframe to reduce the dimension.

        mode (Literal[&quot;pca&quot;, &quot;tsne&quot;], optional): The dimensionality reduction algorithm. Defaults to "pca".

        target_explained_variance (float, optional): The minimum threshold for explained variance. Defaults to 0.95.

        result_as_df (bool, optional): Whether or not return the result as a dataframe or as a numpy array. Defaults to True.

    Returns:
    -------
        npt.NDArray[np.float64] | pd.DataFrame: The reduced dataset.
    """
    assert (
        isinstance(full_dataframe, pd.DataFrame)
        and full_dataframe.shape[0] != 0
        and full_dataframe.shape[1] > 1
    ), "full_dataframe must be a pandas DataFrame containing at least 2 columns and 1 row"
    assert (
        target_explained_variance > 0 and target_explained_variance < 1
    ), "target_explained_variance must be a float between 0 and 1"
    X = full_dataframe.to_numpy()
    match mode:
        case "pca":
            calibration_pca = PCA()
            calibration_pca.fit(X)

            d = (
                np.argmax(
                    np.cumsum(calibration_pca.explained_variance_ratio_)
                    >= target_explained_variance
                )
                + 1
            )
            pca = PCA(n_components=d)
            pca.fit(X)

            if result_as_df:
                return pd.DataFrame(
                    pca.transform(X), columns=[f"PC_{i}" for i in range(1, d + 1)]
                )
            return pca.transform(X)
        case "tsne":
            raise NotImplementedError("TSNE is not implemented yet")
        case _:
            raise ValueError("mode must be either pca or tsne")


def from_returns_to_bins_count(
    returns: pd.Series,
    method: Literal["sturge", "freedman-diaconis"] = "freedman-diaconis",
) -> int:
    try:
        if method == "freedman-diaconis":
            iqr = returns.quantile(0.75) - returns.quantile(0.25)
            bin_width = (2 * iqr) / (returns.shape[0] ** (1 / 3))
            bins = int(np.ceil((returns.max() - returns.min()) / bin_width))
        else:
            bins = int(np.ceil(np.log2(returns.shape[0])) + 1)
    except:
        bins = int(np.ceil(np.log2(returns.shape[0])) + 1)
    return bins 

def get_color_palette(n_colors: int) -> npt.NDArray:
    return np.random.choice(all_palettes["Viridis"][256], n_colors)
