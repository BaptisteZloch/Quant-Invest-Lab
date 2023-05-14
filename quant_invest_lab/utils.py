from bs4 import BeautifulSoup
import requests
import re

CAC40_URL = "https://companiesmarketcap.com/cac-40/largest-companies-by-market-cap/"
AMERICAN_LARGEST_COMPANIES = (
    "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/"
)


def get_cac40_symbols() -> list[str]:
    soup = BeautifulSoup(requests.get(CAC40_URL).content, "lxml")
    raw_codes = soup.find_all("div", class_="company-code")
    return list(map(lambda input: str(re.sub("<[^>]*>", "", str(input))), raw_codes))


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
