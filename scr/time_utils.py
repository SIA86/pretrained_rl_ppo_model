"""Вспомогательные функции для работы с временными индексами."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

IntervalLike = pd.Index | np.ndarray | Iterable[pd.Timestamp]


def align_intervals(index: pd.DatetimeIndex, interval: IntervalLike) -> pd.DatetimeIndex:
    """Привести произвольный интервал к ``DatetimeIndex`` исходного индекса.

    Функция конвертирует входной ``interval`` в ``DatetimeIndex``, выполняет
    согласование часовых поясов с ``index`` и ограничивает интервал рамками
    исходного индекса.
    """

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas.DatetimeIndex")

    seg = pd.DatetimeIndex(interval)

    if index.tz is not None:
        if seg.tz is None:
            seg = seg.tz_localize(index.tz)
        elif str(seg.tz) != str(index.tz):
            seg = seg.tz_convert(index.tz)
    elif seg.tz is not None:
        seg = seg.tz_convert("UTC").tz_localize(None)

    return seg.intersection(index)

