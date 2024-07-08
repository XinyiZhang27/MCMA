import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


def time_features(times, freq='s'):
    """
    > * s - [Second of minute, minute of hour, hour of day, day of week]
    > all encoded between [-0.5 and 0.5]
    """
    times = pd.to_datetime(times.time.values)

    offset = to_offset(freq)
    feature_classes = [SecondOfMinute(),  MinuteOfHour(), HourOfDay(), DayOfWeek()]

    if isinstance(offset, offsets.Second):
        return np.vstack([cls(times) for cls in feature_classes]).transpose(1,0)

    supported_freq_msg = f"""Unsupported frequency {freq}"""
    raise RuntimeError(supported_freq_msg)
