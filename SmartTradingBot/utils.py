from datetime import date
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import yfinance as yf  # type: ignore

"""Utility functions used by other files."""


def get_data(
    stock: str,
    start_date: str = "2021-01-01",
    end_date: Optional[str] = None,
    test_split: float = 0.8,
    data_type: str = "Adj Close",
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

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def normalised_difference(data):
    data = np.diff(data,n=1)
    data = tf.keras.utils.normalize(data)[0]
    return data

def padded_window(data, timestep, window_size=10):
    """Returns window of size window_size from data. Items are padded by data[0]"""
    t = timestep
    d = t - window_size + 1
    data=list(data)
    data=data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1] 
    data=tf.constant(data)
    return tf.reshape(data, shape=(1,-1))

