import numpy as np

try:
    from numba import njit, int8
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    int8 = np.int8


@njit(cache=False)
def _wilder_smooth(arr, period):
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    if period <= 0 or period > n:
        for i in range(n):
            out[i] = np.nan
        return out
    s = 0.0
    for i in range(period):
        s += arr[i]
    out[period-1] = s
    for i in range(period, n):
        s = out[i-1] - (out[i-1] / period) + arr[i]
        out[i] = s
    for i in range(period-1):
        out[i] = np.nan
    return out


@njit(cache=False)
def _rolling_mean_var(x, period):
    """O(n) скользящее среднее и дисперсия через суммы и суммы квадратов."""
    n = x.shape[0]
    mu = np.empty(n, dtype=np.float64)
    var = np.empty(n, dtype=np.float64)
    if period <= 0 or period > n:
        for i in range(n):
            mu[i] = np.nan
            var[i] = np.nan
        return mu, var
    s = 0.0
    s2 = 0.0
    for i in range(period):
        v = x[i]
        s += v
        s2 += v*v
    mu[period-1] = s / period
    var[period-1] = (s2/period) - mu[period-1]*mu[period-1]
    for i in range(period, n):
        out_v = x[i-period]
        in_v  = x[i]
        s += in_v - out_v
        s2 += in_v*in_v - out_v*out_v
        mu[i] = s / period
        var[i] = (s2/period) - mu[i]*mu[i]
    for i in range(period-1):
        mu[i] = np.nan
        var[i] = np.nan
    return mu, var


@njit(cache=False)
def sma_numba(x, period):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    if period <= 0 or period > n:
        for i in range(n):
            out[i] = np.nan
        return out
    s = 0.0
    for i in range(period):
        s += x[i]
    out[period-1] = s / period
    for i in range(period, n):
        s += x[i] - x[i-period]
        out[i] = s / period
    for i in range(period-1):
        out[i] = np.nan
    return out


@njit(cache=False)
def ema_numba(x, period):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    if period <= 0:
        for i in range(n):
            out[i] = np.nan
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = out[i-1] + alpha * (x[i] - out[i-1])
    return out


@njit(cache=False)
def ema_numba_safe(x, period):
    """Не распространяет NaN: если x[i] is NaN — просто копируем предыдущее значение."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    if period <= 0:
        for i in range(n):
            out[i] = np.nan
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = x[0]
    for i in range(1, n):
        xi = x[i]
        if np.isnan(xi):
            out[i] = out[i-1]
        else:
            out[i] = out[i-1] + alpha * (xi - out[i-1])
    return out


@njit(cache=False)
def adx_numba(high, low, close, period):
    n = close.shape[0]
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    tr_s = _wilder_smooth(tr, period)
    plus_dm_s = _wilder_smooth(plus_dm, period)
    minus_dm_s = _wilder_smooth(minus_dm, period)

    plus_di = np.empty(n, dtype=np.float64)
    minus_di = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(tr_s[i]) or tr_s[i] == 0.0:
            plus_di[i] = np.nan
            minus_di[i] = np.nan
        else:
            plus_di[i] = 100.0 * (plus_dm_s[i] / tr_s[i])
            minus_di[i] = 100.0 * (minus_dm_s[i] / tr_s[i])

    dx = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = plus_di[i] + minus_di[i]
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or s == 0.0:
            dx[i] = np.nan
        else:
            dx[i] = 100.0 * (abs(plus_di[i] - minus_di[i]) / s)

    adx = np.empty(n, dtype=np.float64)
    for i in range(n):
        adx[i] = np.nan
    if period <= 0 or period > n:
        return adx

    s = 0.0
    cnt = 0
    start = -1
    for i in range(n):
        if not np.isnan(dx[i]):
            s += dx[i]; cnt += 1
            if cnt == period:
                adx[i] = s / period
                start = i + 1
                break
    if start != -1:
        prev = adx[start-1]
        for i in range(start, n):
            if np.isnan(dx[i]):
                adx[i] = np.nan
            else:
                prev = ((prev * (period - 1)) + dx[i]) / period
                adx[i] = prev
    return adx


@njit(cache=False)
def atr_numba(high, low, close, period):
    n = close.shape[0]
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    atr = _wilder_smooth(tr, period)
    for i in range(n):
        if not np.isnan(atr[i]):
            atr[i] = atr[i] / period
    return atr


@njit(cache=False)
def rsi_numba(close, period=14):
    n = close.shape[0]
    if period <= 0 or n < 2:
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = np.nan
        return out
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        d = close[i] - close[i-1]
        gains[i] = d if d > 0.0 else 0.0
        losses[i] = -d if d < 0.0 else 0.0

    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]):
            out[i] = np.nan
        elif avg_loss[i] == 0.0:
            out[i] = 100.0
        else:
            rs = (avg_gain[i] / period) / (avg_loss[i] / period)
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


@njit(cache=False)
def macd_numba(close, fast=12, slow=26, signal=9):
    ema_fast = ema_numba(close, fast)
    ema_slow = ema_numba(close, slow)
    macd = ema_fast - ema_slow
    signal_line = ema_numba(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist


@njit(cache=False)
def stoch_numba(high, low, close, k_period=14, d_period=3):
    n = close.shape[0]
    k = np.empty(n, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)
    for i in range(n):
        k[i] = np.nan
        d[i] = np.nan

    if k_period <= 0 or d_period <= 0 or k_period > n:
        return k, d

    for i in range(k_period-1, n):
        hh = high[i]
        ll = low[i]
        for j in range(i - k_period + 1, i + 1):
            if high[j] > hh: hh = high[j]
            if low[j]  < ll: ll = low[j]
        denom = hh - ll
        if denom == 0.0:
            k[i] = 0.0
        else:
            k[i] = 100.0 * (close[i] - ll) / denom

    d_start = (k_period - 1) + (d_period - 1)
    if d_start < n:
        s = 0.0
        for j in range(d_start - d_period + 1, d_start + 1):
            s += k[j]
        d[d_start] = s / d_period
        for i in range(d_start + 1, n):
            s += k[i] - k[i - d_period]
            d[i] = s / d_period

    return k, d


@njit(cache=False)
def bollinger_numba(close, period=20, num_std=2.0):
    mu, var = _rolling_mean_var(close, period)
    n = close.shape[0]
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(mu[i]):
            upper[i] = np.nan
            lower[i] = np.nan
        else:
            std = np.sqrt(var[i]) if var[i] > 0.0 else 0.0
            upper[i] = mu[i] + num_std*std
            lower[i] = mu[i] - num_std*std
    return mu, upper, lower


@njit(cache=False)
def obv_numba(close, volume):
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i-1]:
            out[i] = out[i-1] + volume[i]
        elif close[i] < close[i-1]:
            out[i] = out[i-1] - volume[i]
        else:
            out[i] = out[i-1]
    return out


@njit(cache=False)
def cci_numba(high, low, close, period=20):
    n = close.shape[0]
    tp = (high + low + close) / 3.0
    sma_tp = sma_numba(tp, period)
    mad = np.empty(n, dtype=np.float64)
    for i in range(n):
        mad[i] = np.nan
    if period <= 0 or period > n:
        return mad

    for i in range(period-1, n):
        s = 0.0
        for j in range(i-period+1, i+1):
            s += abs(tp[j] - sma_tp[i])
        mad[i] = s / period

    cci = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(sma_tp[i]) or np.isnan(mad[i]) or mad[i] == 0.0:
            cci[i] = np.nan
        else:
            cci[i] = (tp[i] - sma_tp[i]) / (0.015 * mad[i])
    return cci


@njit(cache=False)
def williams_r_numba(high, low, close, period=14):
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    if period <= 0 or period > n:
        return out
    for i in range(period-1, n):
        hh = high[i]
        ll = low[i]
        for j in range(i-period+1, i+1):
            if high[j] > hh: hh = high[j]
            if low[j]  < ll: ll = low[j]
        denom = hh - ll
        if denom == 0.0:
            out[i] = 0.0
        else:
            out[i] = -100.0 * (hh - close[i]) / denom
    return out


@njit(cache=False)
def mfi_numba(high, low, close, volume, period=14):
    n = close.shape[0]
    tp = (high + low + close) / 3.0
    pmf = np.zeros(n, dtype=np.float64)
    nmf = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        mf = tp[i] * volume[i]
        if tp[i] > tp[i-1]:
            pmf[i] = mf
        elif tp[i] < tp[i-1]:
            nmf[i] = mf

    pmf_s = _wilder_smooth(pmf, period)
    nmf_s = _wilder_smooth(nmf, period)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(pmf_s[i]) or np.isnan(nmf_s[i]):
            out[i] = np.nan
        elif nmf_s[i] == 0.0:
            out[i] = 100.0
        else:
            mr = (pmf_s[i] / period) / (nmf_s[i] / period)
            out[i] = 100.0 - (100.0 / (1.0 + mr))
    return out


@njit(cache=False)
def roc_numba(close, period=12):
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            prev = close[i-period]
            if prev == 0.0:
                out[i] = np.nan
            else:
                out[i] = 100.0 * (close[i] - prev) / prev
    return out


@njit(cache=False)
def slope_numba(values, period):
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    if period <= 1 or period > n:
        return out

    period_f = float(period)
    sum_x = 0.5 * period_f * (period_f - 1.0)
    sum_x2 = (period_f - 1.0) * period_f * (2.0 * period_f - 1.0) / 6.0
    denom = period_f * sum_x2 - sum_x * sum_x

    sum_y = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    nan_count = 0
    for k in range(period):
        v = values[k]
        if np.isnan(v):
            nan_count += 1
        else:
            sum_y += v
            sum_y2 += v * v
            sum_xy += k * v

    if nan_count == 0:
        mean_y = sum_y / period_f
        var_y = (sum_y2 / period_f) - mean_y * mean_y
        if var_y > 0.0:
            std_y = np.sqrt(var_y)
            slope = (period_f * sum_xy - sum_x * sum_y) / denom
            out[period - 1] = slope / std_y
        else:
            out[period - 1] = np.nan
        needs_reset = False
    else:
        out[period - 1] = np.nan
        needs_reset = True

    for end in range(period, n):
        y_out = values[end - period]
        y_in = values[end]

        if np.isnan(y_out):
            nan_count -= 1
        if np.isnan(y_in):
            nan_count += 1

        if nan_count == 0:
            if needs_reset:
                sum_y = 0.0
                sum_y2 = 0.0
                sum_xy = 0.0
                start = end - period + 1
                for k in range(period):
                    v = values[start + k]
                    sum_y += v
                    sum_y2 += v * v
                    sum_xy += k * v
            else:
                sum_y_prev = sum_y
                sum_y = sum_y_prev - y_out + y_in
                sum_y2 = sum_y2 - y_out * y_out + y_in * y_in
                sum_xy = sum_xy - (sum_y_prev - y_out) + (period_f - 1.0) * y_in

            mean_y = sum_y / period_f
            var_y = (sum_y2 / period_f) - mean_y * mean_y
            if var_y > 0.0:
                std_y = np.sqrt(var_y)
                slope = (period_f * sum_xy - sum_x * sum_y) / denom
                out[end] = slope / std_y
            else:
                out[end] = np.nan
            needs_reset = False
        else:
            out[end] = np.nan
            needs_reset = True

    return out


@njit(cache=False)
def vwap_numba(high, low, close, volume, period=20):
    n = close.shape[0]
    tp = (high + low + close) / 3.0
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    if period <= 0 or period > n:
        return out
    s_pv = 0.0
    s_v = 0.0
    for i in range(period):
        s_pv += tp[i]*volume[i]
        s_v  += volume[i]
    out[period-1] = (s_pv / s_v) if s_v != 0.0 else np.nan
    for i in range(period, n):
        s_pv += tp[i]*volume[i] - tp[i-period]*volume[i-period]
        s_v  += volume[i] - volume[i-period]
        out[i] = (s_pv / s_v) if s_v != 0.0 else np.nan
    return out


PEAK   =  -1
VALLEY = 1


@njit
def _safe_window_bounds(t, w):
    start = t - w
    if start < 0:
        start = 0
    end = t + 1
    return start, end


@njit
def _local_max(a, start, end):
    mx = a[start]
    idx = start
    for i in range(start+1, end):
        if a[i] > mx:
            mx = a[i]
            idx = i
    return mx, idx


@njit
def _local_min(a, start, end):
    mn = a[start]
    idx = start
    for i in range(start+1, end):
        if a[i] < mn:
            mn = a[i]
            idx = i
    return mn, idx


@njit
def identify_initial_pivot_highlow(high, low, up_thresh, down_thresh):
    """
    Возвращает начальный пивот: PEAK (1) или VALLEY (-1).
    Логика: если быстрее выполняется условие роста от минимального low — стартуем как впадина,
    если падения от максимального high — стартуем как пик, иначе по направлению изменения суммы H+L.
    """
    max_x = high[0]
    min_x = low[0]
    max_t = 0
    min_t = 0

    for t in range(1, len(high)):
        h_t = high[t]
        l_t = low[t]
        if (h_t / min_x - 1.0) >= up_thresh:
            return VALLEY if min_t == 0 else PEAK
        if (l_t / max_x - 1.0) <= down_thresh:
            return PEAK if max_t == 0 else VALLEY
        if h_t > max_x:
            max_x = h_t
            max_t = t
        if l_t < min_x:
            min_x = l_t
            min_t = t

    t_n = len(high) - 1
    return VALLEY if (low[0] + high[0]) < (low[t_n] + high[t_n]) else PEAK


@njit
def zigzag_pivots_highlow_numba(high, low, close, up_thresh, down_thresh, min_bars=3, min_dist=1, eps=1e-12):
    n = len(high)
    piv = np.zeros(n, dtype=int8)

    if not (up_thresh > 0.0 and down_thresh < 0.0):
        if up_thresh <= 0.0:
            up_thresh = 0.01
        if down_thresh >= 0.0:
            down_thresh = -0.01

    init = identify_initial_pivot_highlow(high, low, up_thresh, down_thresh)
    trend = -init
    last_t = 0
    last_x = low[0] if trend == PEAK else high[0]

    last_pivot_t = 0
    last_pivot_x = last_x
    last_pivot_kind = -trend

    for t in range(1, n):
        h = high[t]; l = low[t]; c = close[t]

        if trend == PEAK:
            if h > last_x + eps or (h - last_x) > eps:
                last_x = h
                last_t = t
            if last_x > 0 and (l / last_x - 1.0) <= down_thresh:
                ws, we = _safe_window_bounds(last_t, min_bars)
                mx, mxi = _local_max(high, ws, we)
                if abs(mx - high[last_t]) <= 1e-12 or mxi == last_t:
                    if (t - last_pivot_t) >= min_dist:
                        piv[last_t] = PEAK
                        last_pivot_t = last_t
                        last_pivot_x = last_x
                        last_pivot_kind = PEAK
                        trend = VALLEY
                        last_t = t
                        last_x = l
                    else:
                        trend = VALLEY
                        last_t = t
                        last_x = l

        else:
            if l < last_x - eps or (last_x - l) > eps:
                last_x = l
                last_t = t
            if last_x > 0 and (c / last_x - 1.0) >= up_thresh:
                ws, we = _safe_window_bounds(last_t, min_bars)
                mn, mni = _local_min(low, ws, we)
                if abs(mn - low[last_t]) <= 1e-12 or mni == last_t:
                    if (t - last_pivot_t) >= min_dist:
                        piv[last_t] = VALLEY
                        last_pivot_t = last_t
                        last_pivot_x = last_x
                        last_pivot_kind = VALLEY
                        trend = PEAK
                        last_t = t
                        last_x = h
                    else:
                        trend = PEAK
                        last_t = t
                        last_x = h

    if piv[last_pivot_t] == 0 and 0 <= last_pivot_t < n:
        piv[last_pivot_t] = last_pivot_kind

    if 0 <= last_t < n and piv[last_t] == 0:
        piv[last_t] = trend

    return piv


def expand_pivots(pivots: np.ndarray, N: int) -> np.ndarray:
    """
    Расширяет метки пиков и впадин на N строк в обе стороны.

    :param pivots: Массив с метками пиков (1) и впадин (-1).
    :param N: Количество строк для расширения.
    :return: Массив с расширенными метками.
    """
    expanded_pivots = pivots.copy()

    peak_indices = np.where(pivots == PEAK)[0]
    valley_indices = np.where(pivots == VALLEY)[0]

    for idx in peak_indices:
        start = max(0, idx - 1)
        end = min(len(pivots), idx + N + 1)
        expanded_pivots[start:end] = PEAK

    for idx in valley_indices:
        start = max(0, idx - 1)
        end = min(len(pivots), idx + N + 1)
        expanded_pivots[start:end] = VALLEY

    return expanded_pivots
