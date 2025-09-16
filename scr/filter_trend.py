import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable

def get_intervals(
    df: pd.DataFrame,
    col: str,
    threshold: int = 1000,
    return_numpy: bool = False,
) -> list:
    """
    Возвращает список непрерывных интервалов, где последовательность от 1 до -1
    имеет длину > threshold. Каждый элемент списка — DatetimeIndex (по умолчанию)
    или np.ndarray(datetime64[ns]) при return_numpy=True.

    :param df: DataFrame с DatetimeIndex и колонкой сигналов {-1,0,1}
    :param col: имя колонки сигналов
    :param threshold: минимальная длина (в строках)
    :param return_numpy: если True — вернуть массивы numpy вместо DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index должен быть DatetimeIndex")
    if col not in df.columns:
        raise KeyError(f"Колонка '{col}' не найдена")

    intervals = []
    start_pos = None

    # перечисляем позиции и значения
    for pos, val in enumerate(df[col].to_numpy()):
        if val == 1 and start_pos is None:
            start_pos = pos
        elif val == -1 and start_pos is not None:
            length = pos - start_pos + 1
            if length > threshold:
                seg_idx = df.index[start_pos:pos+1]
                intervals.append(
                    seg_idx if not return_numpy else np.asarray(seg_idx.values.astype("datetime64[ns]"))
                )
            start_pos = None

    return intervals


def plot_price_with_valid(
    df: pd.DataFrame,
    price_col: str,
    valid_segments: Iterable,   # список DatetimeIndex или массивов datetime64[ns]
    *,
    n_plots: int = 5,
    seed: int | None = None,
    assume_df_tz: str | None = None,
    line_width: float = 1.5,
    fig_size: tuple[int, int] = (10, 4),
):
    """
    Для n_plots случайных валидных сегментов строит отдельные графики цены.

    :param df: DataFrame с DatetimeIndex
    :param price_col: колонка цены
    :param valid_segments: список интервалов (DatetimeIndex или np.ndarray datetime64[ns])
    :param n_plots: сколько случайных участков показать
    :param seed: фиксировать случайность
    :param assume_df_tz: если df.index naive, а valid_segments в UTC (или наоборот)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index должен быть DatetimeIndex")
    if price_col not in df.columns:
        raise KeyError(f"Колонка '{price_col}' не найдена")

    di = df.index

    # нормализуем список сегментов к DatetimeIndex
    norm_segments: list[pd.DatetimeIndex] = []
    for seg in valid_segments:
        seg_di = pd.DatetimeIndex(seg)

        # согласование часовых поясов
        if di.tz is not None and seg_di.tz is not None:
            if str(di.tz) != str(seg_di.tz):
                seg_di = seg_di.tz_convert(di.tz)
        elif di.tz is not None and seg_di.tz is None:
            seg_di = seg_di.tz_localize(di.tz)
        elif di.tz is None and seg_di.tz is not None:
            if assume_df_tz:
                di2 = di.tz_localize(assume_df_tz)
                df = df.copy()
                df.index = di2
                di = di2
                seg_di = seg_di.tz_convert(assume_df_tz)
            else:
                seg_di = seg_di.tz_convert("UTC").tz_localize(None)

        seg_di = seg_di.intersection(di)
        if len(seg_di) > 0:
            norm_segments.append(seg_di)

    if len(norm_segments) == 0:
        print("Нет валидных сегментов в пределах индекса DataFrame")
        return

    # выбираем случайные сегменты
    rng = np.random.default_rng(seed)
    k = min(n_plots, len(norm_segments))
    chosen = rng.choice(len(norm_segments), size=k, replace=False)
    chosen_segments = [norm_segments[i] for i in chosen]

    # отдельный график для каждого сегмента
    for i, seg in enumerate(chosen_segments, start=1):
        segment_df = df.loc[seg[0]:seg[-1], price_col]

        plt.figure(figsize=fig_size)
        plt.plot(segment_df.index, segment_df.values, linewidth=line_width)
        plt.title(f"Участок {i}: {seg[0]} → {seg[-1]} Длина: {len(segment_df)}")
        plt.xlabel("Дата")
        plt.ylabel(price_col)
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()