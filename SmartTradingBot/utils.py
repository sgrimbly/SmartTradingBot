from datetime import date
from typing import Tuple

import yfinance as yf

"""Utility functions used by other files."""


def get_data(
    stock, start_date, end_date=None, test_split=0.8, data_type="Adj Close"
) -> Tuple:
    """Get the Yahoo data for a given stock in a specified date range."""
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    # TODO Check for cached data

    # auto_adjust=True -> removing Adjust Close column
    # actions =”inline” -> dowload dividend and splits of stock
    # ['FB','AAPL',...] -> download data for multiple stocks
    df_yahoo = yf.download(stock, start=start_date, end=end_date)
    data = df_yahoo[data_type]
    test_split_index = int(test_split * len(data))  # Keep 20% for validation
    return data[:test_split_index], data[test_split_index:]
