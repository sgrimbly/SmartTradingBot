from datetime import date
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import yfinance as yf  # type: ignore
from pandas.core.frame import DataFrame

"""Utility functions used by other files."""


def get_data(
    stock: str,
    start_date: str = "2021-01-01",
    end_date: Optional[str] = None,
    test_split: float = 0.8,
    data_type: str = "Adj Close",
    normalise: bool = True,
    differences: int = 1,  # 0 for no differences
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

    if normalise:
        data = normalised_difference(data=data)
    if differences != 0:
        data = sigmoid(data)

    test_split_index = int(test_split * len(data))  # Keep 20% for validation
    return data[:test_split_index], data[test_split_index:]


def sigmoid(z: np.array) -> np.array:
    return 1 / (1 + np.exp(-z))


def normalised_difference(data: DataFrame) -> np.array:
    data = np.diff(data, n=1)
    data = tf.keras.utils.normalize(data)[0]
    return data


def padded_window(
    data: Union[DataFrame, np.array], timestep: int, window_size: int = 10
) -> tf.Tensor:
    """Returns window of size window_size from data. Items are padded by
    data[0]"""
    t = timestep
    d = t - window_size + 1
    data = list(data)
    if d >= 0:
        data = data[d : t + 1]  # noqa: E203
    else:
        data = -d * [data[0]] + data[0 : t + 1]  # noqa: E203
    data = tf.constant(data)
    return tf.reshape(data, shape=(1, -1))
