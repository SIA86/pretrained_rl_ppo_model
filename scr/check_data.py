import numpy as np
import pandas as pd
from typing import Literal, Optional, List, Dict
from pandas.tseries.frequencies import to_offset


def _detect_unix_unit(x) -> str:
    """Определение единицы UNIX-времени: 's'|'ms'|'us'|'ns'."""
    try:
        xv = float(x)
    except Exception:
        return 's'
    ax = abs(xv)
    if ax < 1e11:
        return 's'
    if ax < 1e14:
        return 'ms'
    if ax < 1e17:
        return 'us'
    return 'ns'


def _robust_infer_freq(ts: pd.DatetimeIndex) -> Optional[str]:
    """Сначала pd.infer_freq, затем медианный лаг → ближайший Offset."""
    if ts.size < 3:
        return None
    try:
        f = pd.infer_freq(ts)
        if f is not None:
            return f
    except Exception:
        pass
    diffs = np.diff(ts.asi8)
    if diffs.size == 0:
        return None
    med = int(np.median(diffs))
    if med <= 0:
        return None
    try:
        return to_offset(pd.Timedelta(med, unit='ns')).freqstr
    except Exception:
        return None


def prepare_time_series(
    df: pd.DataFrame,
    timestamp_col: str,
    tz: str = "UTC",
    from_date: Optional[pd.Timestamp] = None,
    to_date: Optional[pd.Timestamp] = None,
    allow_backfill: bool = False,
    volume_columns: Optional[List[str]] = ['Volume'],
    target_columns: Optional[List[str]] = None,
    dedup_agg: Optional[Dict[str, str]] = None,
    fallback_freq: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Готовит финансовый временной ряд:
    - ВСЕГДА сбрасывает индекс и строит новый DatetimeIndex из `timestamp_col`
    - Нормализует таймзону к `tz` (по умолчанию UTC)
    - Удаляет (или агрегирует) дубликаты по меткам времени
    - Восстанавливает регулярную частоту (устойчивый infer); при необходимости использует `fallback_freq`
    - Заполняет пропуски: цены/фичи — ffill; объёмы — 0; таргеты — не трогает; (опц.) bfill
    - Делает срез по окну [from_date, to_date)

    Возвращает DataFrame с tz-aware DatetimeIndex.
    """
    if df is None or len(df) == 0:
        raise ValueError("Пустой DataFrame.")

    if timestamp_col not in df.columns:
        raise ValueError(f"Колонка '{timestamp_col}' не найдена в DataFrame.")

    df = df.reset_index(drop=True).copy()

    unit = _detect_unix_unit(df[timestamp_col].iloc[0])
    is_numeric_ts = pd.api.types.is_numeric_dtype(df[timestamp_col])
    if is_numeric_ts:
        dt = pd.to_datetime(df[timestamp_col], unit=unit, errors='coerce', utc=True)
    else:
        dt = pd.to_datetime(df[timestamp_col], errors='coerce', utc=True)
    if dt.isna().any():
        raise ValueError("Некоторые метки времени не удалось преобразовать в datetime.")
    dt = dt.dt.tz_convert(tz)
    df.index = dt
    df.sort_index(inplace=True)
    df.index.name = 'datetime'

    if dedup_agg:
        dup_cnt = df.index.duplicated().sum()
        if dup_cnt and verbose:
            print(f"Обнаружено дубликатов: {dup_cnt}. Агрегирую по {dedup_agg}")
        df = df.groupby(level=0).agg(dedup_agg).sort_index()
    else:
        before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        dropped = before - len(df)
        if verbose and dropped:
            print(f"Удалено дубликатов: {dropped}")

    inferred_freq = _robust_infer_freq(df.index)
    if inferred_freq is None:
        inferred_freq = fallback_freq
    if verbose:
        print(f"Инферированная частота: {inferred_freq}")

    if inferred_freq:
        full_idx = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=inferred_freq,
            tz=df.index.tz
        )
        if len(full_idx) != len(df.index) or not full_idx.equals(df.index):
            if verbose:
                missing = full_idx.difference(df.index)
                if len(missing):
                    print(f"Пропущено точек: {len(missing)}; первая/последняя: {missing[0]}, {missing[-1]}")
            df = df.reindex(full_idx)

    volume_columns = volume_columns or [c for c in df.columns if 'vol' in c.lower() or 'volume' in c.lower()]
    target_columns = set(target_columns or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    price_like = [c for c in numeric_cols if c not in volume_columns and c not in target_columns]
    if price_like:
        df[price_like] = df[price_like].ffill()
        if allow_backfill:
            df[price_like] = df[price_like].bfill()

    if volume_columns:
        for c in volume_columns:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)

    non_numeric = [c for c in df.columns if c not in numeric_cols and c not in target_columns]
    if non_numeric:
        df[non_numeric] = df[non_numeric].ffill()
        if allow_backfill:
            df[non_numeric] = df[non_numeric].bfill()

    if from_date is not None:
        frm_ts = pd.Timestamp(from_date)
        frm_ts = frm_ts.tz_convert(tz) if frm_ts.tz is not None else frm_ts.tz_localize(tz)
        df = df[df.index >= frm_ts]
    if to_date is not None:
        to_ts = pd.Timestamp(to_date)
        to_ts = to_ts.tz_convert(tz) if to_ts.tz is not None else to_ts.tz_localize(tz)
        df = df[df.index < to_ts]

    if verbose and len(df):
        print(f"Начало {df.index[0]} конец {df.index[-1]} (len={len(df)})")

    return df
