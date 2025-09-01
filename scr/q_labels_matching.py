import numpy as np
import pandas as pd
from typing import Literal, Optional, Iterable

"""
1. Execution @ next_open: решения на баре t исполняются на Open[t+1]. Это обеспечивает корректную причинность и реалистичный бэктест.
2. One-side + действия: работаем отдельно long-only или short-only; действия — Open / Close / Hold / Wait.
Реверсы исключаем (реверс = Close → Open на следующем баре).
3. Абсолютные Q-метки (а не Sharpe-подобные): мы целимся в последующий PPO, где важно сохранить «масштаб выигрыша» действий.
Поэтому никаких делений на σ, √Δt или построчных нормировок — они ломают сопоставимость Close vs Hold/Open.
4. Единая временная опора для сравнения: в позиции сравниваем из t+1:
Close = 0 (после закрытия прирост дальше равен нулю),
Hold = Ret(t+1 → TGT). Это устраняет «ранний выход на полпути» из-за некорректных горизонтов и даёт стабильные решения.
5. Три режима разметки:
  mode='exit' — continuation до teacher-exit. Хорош для имитации конкретной стратегии выдержки/выхода.
  mode='horizon' — фиксированный горизонт H. Классика supervised в финансах; стационарная цель без лейбл-ликинга по exit.
  mode='tdlambda' — смесь n-step горизонтов (TF-style, TD-λ). Снижает дисперсию меток и согласуется с тем, как PPO/A2C обрабатывают возвраты.
6. MAE-штраф (опционально):
  если нужно «не сидеть в глубокой просадке», вводим штраф по Maximum Adverse Excursion на интервале удержания. 
  Он уменьшает Q_Hold (и/или Q_Open) в абсолютной шкале, не ломая сопоставимость.

7. Нормализация/клип:
  допускается только глобальный winsorize/клип/робаст-скейл по train (единый для всех действий), чтобы стабилизировать обучение и 
  подобрать разумную температуру τ для softmax. Не применять построчные/волатильностные нормировки.
8. Таргеты для SL → PPO:
  строим Y = softmax(Q/τ) с масками валидности (advantage-нормировка нежелательна на этапе SL, т.к. теряет глобальный контекст; 
  он нужен для калибровки PPO). Value-head для PPO можно инициализировать как Vτ(s) = τ * log Σ_a exp(Q_a/τ) (мягкий максимум) 
  или max_a Q_a.
"""

# ------------------------------------------------------------
# Вспомогательные утилиты (если у тебя уже есть версии — можно удалить эти)
# ------------------------------------------------------------


def simulate_position_one_side(
    open_px: np.ndarray,
    buy_sig: np.ndarray,
    sell_sig: np.ndarray,
    side_long: bool = True,
):
    """
    Односторонняя логика без реверсов (pos ∈ {0,+1} или {0,-1}).
    Вход/выход по сигналам, исполнение на next open.
    Возвращает:
      pos[t]       : текущая позиция на баре t
      entry_eff[t] : цена входа ТЕКУЩЕЙ позиции (NaN во flat)
    """
    n = len(open_px)
    pos = np.zeros(n, dtype=np.int8)
    entry_eff = np.full(n, np.nan, dtype=np.float64)
    pv = 1 if side_long else -1
    p = 0
    ee = np.nan
    for t in range(n - 1):
        pos[t] = p
        entry_eff[t] = ee
        enter_sig = buy_sig[t] if side_long else sell_sig[t]
        exit_sig = sell_sig[t] if side_long else buy_sig[t]
        if p == 0:
            if enter_sig:
                p = pv
                # вход исполняется на t+1 без комиссий (комиссии учитываем в Q)
                ee = open_px[t + 1]
        else:
            if exit_sig:
                p = 0
                ee = np.nan
    pos[n - 1] = p
    entry_eff[n - 1] = ee
    return pos, entry_eff


def next_exit_exec_arrays(
    open_px: np.ndarray,
    buy_sig: np.ndarray,
    sell_sig: np.ndarray,
    side_long: bool = True,
):
    """
    Для каждого t возвращает:
      exit_exec_idx[t] — индекс бара, на котором будет ИСПОЛНЁН выход (exec@next open),
                         если бы мы следовали teacher-логике; иначе -1
      exit_exec_px[t]  — Open[exit_exec_idx[t]] или NaN
    Логика: ищем ближайший future 'exit' сигнал; его исполнение = next open после сигнала.
    """
    n = len(open_px)
    exit_idx = np.full(n, -1, dtype=np.int64)
    exit_px = np.full(n, np.nan, dtype=np.float64)

    # какие сигналы считаем выходом?
    # long-only: выход = sell_sig; short-only: выход = buy_sig
    exit_sig = sell_sig if side_long else buy_sig
    next_exec = np.full(n, -1, dtype=np.int64)
    # для каждого t_exit, исполнение на (t_exit+1)
    for t in range(n - 1):
        if exit_sig[t]:
            next_exec[t] = t + 1

    # список ИНДЕКСОВ ИСПОЛНЕНИЯ (t_exit+1), отсортированный по времени
    exec_positions = next_exec[next_exec >= 0]
    ptr = 0
    for t in range(n):
        # продвинем ptr до первого exec >= t+2 (чтобы было минимум одно плечо PnL)
        while ptr < len(exec_positions) and exec_positions[ptr] < (t + 2):
            ptr += 1
        if ptr < len(exec_positions):
            e_exec = exec_positions[ptr]
            exit_idx[t] = e_exec
            exit_px[t] = open_px[e_exec]
    return exit_idx, exit_px


# ------------------------------------------------------------
# Метрики текущей позиции
# ------------------------------------------------------------


def calc_position_metrics(
    pos: np.ndarray,
    entry_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side_long: bool = True,
):
    """Вычисляет метрики по смоделированной позиции.

    Возвращает кортеж из четырёх массивов длины ``n``:
      * ``unreal``  – нереализованный PnL (в процентах);
      * ``flat_cnt`` – количество последовательных шагов вне позиции;
      * ``hold_cnt`` – количество последовательных шагов в позиции (удержание);
      * ``dd``       – текущая просадка открытой позиции (в процентах).
    """
    n = len(pos)
    unreal = np.zeros(n, dtype=np.float64)
    flat_cnt = np.zeros(n, dtype=np.int64)
    hold_cnt = np.zeros(n, dtype=np.int64)
    dd = np.zeros(n, dtype=np.float64)

    run_flat = 0
    run_hold = 0
    worst = np.nan
    entry = np.nan

    for t in range(n):
        if pos[t] == 0:
            run_flat += 1
            run_hold = 0
            flat_cnt[t] = run_flat
            hold_cnt[t] = 0
            unreal[t] = 0.0
            dd[t] = 0.0
        else:
            if t == 0 or pos[t - 1] == 0:
                run_hold = 1
                run_flat = 0
                entry = entry_px[t] if np.isfinite(entry_px[t]) else close[t]
                worst = low[t] if side_long else high[t]
            else:
                run_hold += 1
                run_flat = 0
                if side_long:
                    worst = min(worst, low[t])
                else:
                    worst = max(worst, high[t])
            hold_cnt[t] = run_hold
            flat_cnt[t] = 0
            if side_long:
                unreal[t] = close[t] / entry - 1.0
                dd[t] = worst / entry - 1.0
            else:
                unreal[t] = entry / close[t] - 1.0
                dd[t] = entry / worst - 1.0

    return unreal, flat_cnt, hold_cnt, dd


# ------------------------------------------------------------
# Основная разметка Q с mode ∈ {'exit','horizon','tdlambda'}
# ------------------------------------------------------------


def enrich_q_labels_trend_one_side(
    df: pd.DataFrame,
    mode: Literal["exit", "horizon", "tdlambda"] = "exit",
    side_long: bool = True,
    # для 'horizon'
    horizon: int = 60,
    # для 'tdlambda'
    H_max: int = 60,
    lam: float = 0.9,
    # комиссии и проскальзывание (доли, например 0.0002 = 2 bps)
    fee: float = 0.0002,
    slippage: float = 0.0001,
    # MAE-штраф (по желанию) — применяется ТОЛЬКО к Hold/Open (см. ниже)
    use_mae_penalty: bool = False,
    mae_lambda: float = 0.0,  # 0.0 = нет штрафа
    mae_apply_to: Literal["hold", "open", "both", "none"] = "hold",
) -> pd.DataFrame:
    """
    Размечает абсолютные Q для one-side (long-only или short-only) и действий: Open / Close / Hold / Wait.
    Всегда предполагается исполнение решений на next open (t -> t+1).

    mode='exit':
        Open = Ret(t+1 -> exit_exec), c обеими ногами издержек
        Close = 0
        Hold = Ret(t+1 -> exit_exec)
        Wait = 0

    mode='horizon':
        Open = Ret(t+1 -> t+H), с издержками входа и выхода
        Close = 0
        Hold = Ret(t+1 -> t+H)
        Wait = 0

    mode='tdlambda':
        Open = Σ_{n=1..H_max} w_n * Ret(t+1 -> t+n), с издержками входа/выхода
        Close = 0
        Hold = Σ_{n=1..H_max} w_n * Ret(t+1 -> t+n)
        Wait = 0
        где w_n = (1-λ) λ^{n-1} / нормировка.

    Все Q — АБСОЛЮТНЫЕ доходности (без деления на σ или √Δt).
    Маски Mask_* показывают исполнимость действия в моменте.

    Требуемые колонки df: 'Open','High','Low','Close','Signal_Rule'
      Signal_Rule: +1 — buy-сигнал, -1 — sell-сигнал (для teacher)
    """
    need = {"Open", "High", "Low", "Close", "Signal_Rule"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"нет колонок: {sorted(miss)}")

    out = df.copy()
    Open = out["Open"].to_numpy(np.float64)
    High = out["High"].to_numpy(np.float64)
    Low = out["Low"].to_numpy(np.float64)
    Close = out["Close"].to_numpy(np.float64)
    n = len(Open)

    # сигналы учителя (ПРОВЕРЬ соответствие под твой пайплайн!)
    buy_sig = out["Signal_Rule"].to_numpy() == 1
    sell_sig = out["Signal_Rule"].to_numpy() == -1
    buy_sig = buy_sig.astype(np.bool_)
    sell_sig = sell_sig.astype(np.bool_)

    # позиция учителя (для entry_eff/pos)
    pos, entry_eff = simulate_position_one_side(
        Open, buy_sig, sell_sig, side_long=side_long
    )
    out["Pos"] = pos

    # метрики текущей позиции
    unreal, flat_steps, hold_steps, drawdown = calc_position_metrics(
        pos, entry_eff, High, Low, Close, side_long=side_long
    )
    out["Unreal_PnL"] = unreal.astype(np.float32)
    out["Flat_Steps"] = (flat_steps / 1000).astype(np.float32)
    out["Hold_Steps"] = (hold_steps / 1000).astype(np.float32)
    out["Drawdown"] = drawdown.astype(np.float32)

    # исполнение "сейчас" — next open
    exec_next_open = np.full(n, np.nan)
    exec_next_open[:-1] = Open[1:]

    # комиссии
    c_open = fee + slippage
    c_close = fee + slippage

    # маски состояний
    pos_now = pos
    flat = pos_now == 0
    inpos = (pos_now != 0) & np.isfinite(entry_eff)
    has_next = np.arange(n) < n - 1

    # контейнеры Q и масок
    Q_Open = np.full(n, np.nan, np.float64)
    Q_Close = np.full(n, np.nan, np.float64)
    Q_Hold = np.full(n, np.nan, np.float64)
    Q_Wait = np.full(n, np.nan, np.float64)

    M_Open = np.zeros(n, np.int8)
    M_Close = np.zeros(n, np.int8)
    M_Hold = np.zeros(n, np.int8)
    M_Wait = np.zeros(n, np.int8)

    # ------------------------------
    # mode = 'exit' : продолжение до teacher-exit
    # ------------------------------
    if mode == "exit":
        # ближайший future exit (exec@next open)
        exit_idx, exit_px = next_exit_exec_arrays(
            Open, buy_sig, sell_sig, side_long=side_long
        )
        has_exit = exit_idx >= (
            np.arange(n) + 2
        )  # нужен хотя бы один бар PnL: t+1 -> exit

        # Open (flat): t+1 -> exit, с обеими ногами издержек
        m_open = flat & has_next & has_exit
        if side_long:
            entry_eff_open = exec_next_open * (1.0 + c_open)
            exit_eff_open = exit_px * (1.0 - c_close)
            Q_Open[m_open] = exit_eff_open[m_open] / entry_eff_open[m_open] - 1.0
        else:
            entry_eff_open = exec_next_open * (1.0 - c_open)  # sell
            exit_eff_open = exit_px * (1.0 + c_close)  # buy
            Q_Open[m_open] = entry_eff_open[m_open] / exit_eff_open[m_open] - 1.0
        M_Open[m_open] = 1

        # Close (inpos): в TD-конвенции — baseline "дальше 0"
        m_close = inpos & has_next
        Q_Close[m_close] = 0.0
        M_Close[m_close] = 1

        # Hold (inpos): t+1 -> exit (continuation)
        m_hold = inpos & has_next & has_exit
        if side_long:
            Q_Hold[m_hold] = (exit_px[m_hold] / exec_next_open[m_hold]) - 1.0
        else:
            Q_Hold[m_hold] = (exec_next_open[m_hold] / exit_px[m_hold]) - 1.0
        M_Hold[m_hold] = 1

        # Wait (flat): 0
        m_wait = flat
        Q_Wait[m_wait] = 0.0
        M_Wait[m_wait] = 1

        # опционально: MAE-штраф (уменьшает Q_Hold и/или Q_Open)
        if use_mae_penalty and mae_lambda > 0.0:
            # штраф считаем по пути t+1...exit-1 относительно ExecNow
            def mae_fwd_long(t):
                L, R = t + 1, exit_idx[t]
                if R <= L or not np.isfinite(exec_next_open[t]):
                    return 0.0
                worst = np.nanmin(Low[L:R])
                return min(worst / exec_next_open[t] - 1.0, 0.0)

            def mae_fwd_short(t):
                L, R = t + 1, exit_idx[t]
                if R <= L or not np.isfinite(exec_next_open[t]):
                    return 0.0
                worst = np.nanmax(High[L:R])
                return min(exec_next_open[t] / worst - 1.0, 0.0)

            # HOLD
            if mae_apply_to in ("hold", "both"):
                idxs = np.where(M_Hold == 1)[0]
                if side_long:
                    penalties = np.array([mae_fwd_long(t) for t in idxs])
                else:
                    penalties = np.array([mae_fwd_short(t) for t in idxs])
                Q_Hold[idxs] = Q_Hold[idxs] - mae_lambda * np.abs(penalties)

            # OPEN
            if mae_apply_to in ("open", "both"):
                idxs = np.where(M_Open == 1)[0]
                if side_long:
                    penalties = np.array([mae_fwd_long(t) for t in idxs])
                else:
                    penalties = np.array([mae_fwd_short(t) for t in idxs])
                Q_Open[idxs] = Q_Open[idxs] - mae_lambda * np.abs(penalties)

    # ------------------------------
    # mode = 'horizon' : фиксированный горизонт H
    # ------------------------------
    elif mode == "horizon":
        H = int(max(1, horizon))
        fut_idx = np.arange(n) + H
        has_fut = fut_idx < n

        fut_px = np.full(n, np.nan)
        fut_px[has_fut] = Open[fut_idx[has_fut]]

        # Open (flat): t+1 -> t+H, с обеими ногами издержек
        m_open = flat & has_next & has_fut
        if side_long:
            entry_eff_open = exec_next_open * (1.0 + c_open)
            exit_eff_open = fut_px * (1.0 - c_close)
            Q_Open[m_open] = exit_eff_open[m_open] / entry_eff_open[m_open] - 1.0
        else:
            entry_eff_open = exec_next_open * (1.0 - c_open)
            exit_eff_open = fut_px * (1.0 + c_close)
            Q_Open[m_open] = entry_eff_open[m_open] / exit_eff_open[m_open] - 1.0
        M_Open[m_open] = 1

        # Close: 0
        m_close = inpos & has_next
        Q_Close[m_close] = 0.0
        M_Close[m_close] = 1

        # Hold: t+1 -> t+H
        m_hold = inpos & has_next & has_fut
        if side_long:
            Q_Hold[m_hold] = (fut_px[m_hold] / exec_next_open[m_hold]) - 1.0
        else:
            Q_Hold[m_hold] = (exec_next_open[m_hold] / fut_px[m_hold]) - 1.0
        M_Hold[m_hold] = 1

        # Wait: 0
        m_wait = flat
        Q_Wait[m_wait] = 0.0
        M_Wait[m_wait] = 1

        # (опционально) MAE-штраф к Hold/Open — в горизонте считаем MAE на [t+1, t+H)
        if use_mae_penalty and mae_lambda > 0.0:

            def mae_h_long(t):
                L, R = t + 1, t + H
                if R <= L or not np.isfinite(exec_next_open[t]) or R > n:
                    return 0.0
                worst = np.nanmin(Low[L:R])
                return min(worst / exec_next_open[t] - 1.0, 0.0)

            def mae_h_short(t):
                L, R = t + 1, t + H
                if R <= L or not np.isfinite(exec_next_open[t]) or R > n:
                    return 0.0
                worst = np.nanmax(High[L:R])
                return min(exec_next_open[t] / worst - 1.0, 0.0)

            if mae_apply_to in ("hold", "both"):
                idxs = np.where(M_Hold == 1)[0]
                penalties = np.array(
                    [mae_h_long(t) if side_long else mae_h_short(t) for t in idxs]
                )
                Q_Hold[idxs] = Q_Hold[idxs] - mae_lambda * np.abs(penalties)

            if mae_apply_to in ("open", "both"):
                idxs = np.where(M_Open == 1)[0]
                penalties = np.array(
                    [mae_h_long(t) if side_long else mae_h_short(t) for t in idxs]
                )
                Q_Open[idxs] = Q_Open[idxs] - mae_lambda * np.abs(penalties)

    # ------------------------------
    # mode = 'tdlambda' : смесь n-step горизонтов (TF-style)
    # ------------------------------
    elif mode == "tdlambda":
        Hm = int(max(1, H_max))
        # подготовим будущие Open для n=1..Hm
        fut = np.full((n, Hm), np.nan, dtype=np.float64)
        for j in range(1, Hm + 1):
            idx = np.arange(n) + j
            ok = idx < n
            fut[ok, j - 1] = Open[idx[ok]]

        # веса TD(λ)
        w = np.array(
            [(1.0 - lam) * (lam ** (k - 1)) for k in range(1, Hm + 1)], dtype=np.float64
        )
        w = w / np.sum(w)

        # Open (flat): смесь n-step
        m_open = flat & has_next
        if np.any(m_open):
            if side_long:
                entry_eff = (exec_next_open[m_open] * (1.0 + c_open))[:, None]
                exit_eff = fut[m_open, :] * (1.0 - c_close)
                vals = exit_eff / entry_eff - 1.0
            else:
                entry_eff = (exec_next_open[m_open] * (1.0 - c_open))[:, None]  # sell
                exit_eff = fut[m_open, :] * (1.0 + c_close)  # buy
                vals = entry_eff / exit_eff - 1.0
            # валидные столбцы: где fut не NaN
            mask_cols = np.isfinite(vals)
            vals[~mask_cols] = np.nan
            # средневзвешенно по доступным n (нормировка по доступным весам)
            weights = np.broadcast_to(w, vals.shape).copy()
            weights[~mask_cols] = 0.0
            denom = weights.sum(axis=1, keepdims=True)
            valid_rows = denom.squeeze(1) > 0.0
            denom[denom == 0.0] = np.nan  # чтобы итог был NaN, а не 0
            q_vals = np.nansum(vals * weights, axis=1) / denom.squeeze(1)
            # назначаем только туда, где есть хотя бы один валидный горизонт
            idx_rows = np.where(m_open)[0]
            Q_Open[idx_rows[valid_rows]] = q_vals[valid_rows]
            M_Open[idx_rows[valid_rows]] = 1

        # Close: 0
        m_close = inpos & has_next
        Q_Close[m_close] = 0.0
        M_Close[m_close] = 1

        # Hold (inpos): смесь n-step
        m_hold = inpos & has_next
        if np.any(m_hold):
            if side_long:
                vals = (fut[m_hold, :] / exec_next_open[m_hold, None]) - 1.0
            else:
                vals = (exec_next_open[m_hold, None] / fut[m_hold, :]) - 1.0
            mask_cols = np.isfinite(vals)
            vals[~mask_cols] = np.nan
            weights = np.broadcast_to(w, vals.shape).copy()
            weights[~mask_cols] = 0.0
            denom = weights.sum(axis=1, keepdims=True)
            valid_rows = denom.squeeze(1) > 0.0
            denom[denom == 0.0] = np.nan
            q_vals = np.nansum(vals * weights, axis=1) / denom.squeeze(1)
            idx_rows = np.where(m_hold)[0]
            Q_Hold[idx_rows[valid_rows]] = q_vals[valid_rows]
            M_Hold[idx_rows[valid_rows]] = 1

        # Wait: 0
        m_wait = flat
        Q_Wait[m_wait] = 0.0
        M_Wait[m_wait] = 1

        # опциональный MAE-штраф (на интервалах [t+1, t+n)), применим эквивалентно к средневзвешенной MAE
        if use_mae_penalty and mae_lambda > 0.0:
            # Предвычислим MAE для каждого n (дороже, но прозрачно)
            N = n  # длина временного ряда

            def mae_n_long(t, n_step):
                L, R = t + 1, t + n_step  # R - правая граница (исключительная)
                if R <= L or R > N or not np.isfinite(exec_next_open[t]):
                    return 0.0
                worst = np.nanmin(Low[L:R])
                return min(worst / exec_next_open[t] - 1.0, 0.0)

            def mae_n_short(t, n_step):
                L, R = t + 1, t + n_step
                if R <= L or R > N or not np.isfinite(exec_next_open[t]):
                    return 0.0
                worst = np.nanmax(High[L:R])
                return min(exec_next_open[t] / worst - 1.0, 0.0)

            # HOLD
            if mae_apply_to in ("hold", "both"):
                idxs = np.where(M_Hold == 1)[0]
                penalties = []
                for t in idxs:
                    acc = 0.0
                    zw = 0.0
                    for j in range(1, Hm + 1):
                        R = t + j
                        if R < N:
                            pen = mae_n_long(t, j) if side_long else mae_n_short(t, j)
                            acc += w[j - 1] * np.abs(pen)
                            zw += w[j - 1]
                    penalties.append(acc / (zw + 1e-12))
                Q_Hold[idxs] = Q_Hold[idxs] - mae_lambda * np.asarray(penalties)

            # OPEN
            if mae_apply_to in ("open", "both"):
                idxs = np.where(M_Open == 1)[0]
                penalties = []
                for t in idxs:
                    acc = 0.0
                    zw = 0.0
                    for j in range(1, Hm + 1):
                        R = t + j
                        if R < N:
                            pen = mae_n_long(t, j) if side_long else mae_n_short(t, j)
                            acc += w[j - 1] * np.abs(pen)
                            zw += w[j - 1]
                    penalties.append(acc / (zw + 1e-12))
                Q_Open[idxs] = Q_Open[idxs] - mae_lambda * np.asarray(penalties)

    else:
        raise ValueError("mode ∈ {'exit','horizon','tdlambda'}")

    # итоговые колонки
    out["Q_Open"] = Q_Open.astype(np.float32)
    out["Q_Close"] = Q_Close.astype(np.float32)
    out["Q_Hold"] = Q_Hold.astype(np.float32)
    out["Q_Wait"] = Q_Wait.astype(np.float32)

    out["Mask_Open"] = M_Open
    out["Mask_Close"] = M_Close
    out["Mask_Hold"] = M_Hold
    out["Mask_Wait"] = M_Wait

    return out.reset_index(drop=True)


def soft_signal_labels_gaussian(
    df: pd.DataFrame,
    side_long: bool = True,
    blur_window: int = 3,
    blur_sigma: float = 1.0,
    mae_lambda: float = 0.0,
) -> pd.DataFrame:
    """Строит мягкие action-метки ``A_*`` из ``Signal_Rule``.
    * ``Signal_Rule``: +1 — вход, -1 — выход, 0 — Hold/Wait в зависимости от позиции.
    * Размытие гауссом действует ТОЛЬКО слева от сигнала: веса Open/Close
      монотонно растут к событию и достигают 0.9 в точке сигнала.
    * Hold/Wait дополняют вероятности до 1, убывая внутри окна.
    * В позиции базовая метка = Hold, однако MAE-штраф уменьшает вес Hold
      в пользу Close пропорционально глубине просадки.
    * В результате добавляется колонка ``Pos`` — смоделированная позиция
      (one-side) для дальнейшей визуализации.
    """
    need = {"Open", "High", "Low", "Close", "Signal_Rule"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"нет колонок: {sorted(miss)}")

    out = df.copy()
    Open = out["Open"].to_numpy(np.float64)
    High = out["High"].to_numpy(np.float64)
    Low = out["Low"].to_numpy(np.float64)
    Close = out["Close"].to_numpy(np.float64)
    sig = out["Signal_Rule"].to_numpy(np.int8)
    n = len(out)

    buy_sig = sig == 1
    sell_sig = sig == -1

    pos, entry_px = simulate_position_one_side(
        Open, buy_sig, sell_sig, side_long=side_long
    )
    inpos = pos != 0
    flat = ~inpos

    # метрики текущей позиции
    unreal, flat_steps, hold_steps, drawdown = calc_position_metrics(
        pos, entry_px, High, Low, Close, side_long=side_long
    )
    out["Unreal_PnL"] = unreal.astype(np.float32)
    out["Flat_Steps"] = (flat_steps / 1000).astype(np.float32)
    out["Hold_Steps"] = (hold_steps / 1000).astype(np.float32)
    out["Drawdown"] = drawdown.astype(np.float32)

    offsets = np.arange(blur_window + 1)
    kernel = np.exp(-0.5 * (offsets / blur_sigma) ** 2)
    kernel /= kernel[0]
    weights = 0.9 * kernel

    a_open = np.zeros(n, dtype=np.float64)
    a_close = np.zeros(n, dtype=np.float64)
    a_hold = inpos.astype(np.float64)
    a_wait = flat.astype(np.float64)

    open_idx = np.where(buy_sig & flat)[0]
    for i in open_idx:
        for k, w in enumerate(weights):
            j = i - k
            if j < 0 or not flat[j]:
                break
            if w > a_open[j]:
                a_open[j] = w
                a_wait[j] = 1.0 - w

    close_idx = np.where(sell_sig & inpos)[0]
    for i in close_idx:
        for k, w in enumerate(weights):
            j = i - k
            if j < 0 or not inpos[j]:
                break
            if w > a_close[j]:
                a_close[j] = w
                a_hold[j] = 1.0 - w

    if mae_lambda > 0.0:
        mae_h = np.zeros(n, dtype=np.float64)
        mae_w = np.zeros(n, dtype=np.float64)
        worst_h = np.nan
        worst_w = np.nan
        entry = np.nan
        for t in range(n):
            if inpos[t]:
                if t == 0 or not inpos[t - 1]:
                    entry = entry_px[t]
                    worst_h = Low[t] if side_long else High[t]
                else:
                    if side_long:
                        worst_h = min(worst_h, Low[t])
                    else:
                        worst_h = max(worst_h, High[t])
                if side_long:
                    mae_h[t] = worst_h / entry - 1.0
                else:
                    mae_h[t] = entry / worst_h - 1.0
            else:
                mae_h[t] = 0.0

            if flat[t]:
                if t == 0 or inpos[t - 1]:
                    entry = Open[t]
                    worst_w = High[t] if side_long else Low[t]
                else:
                    if side_long:
                        worst_w = max(worst_w, High[t])
                    else:
                        worst_w = min(worst_w, Low[t])
                if side_long:
                    mae_w[t] = worst_w / entry - 1.0
                else:
                    mae_w[t] = entry / worst_w - 1.0
            else:
                mae_w[t] = 0.0

        pen_h = mae_lambda * np.abs(mae_h)
        pen_w = mae_lambda * np.abs(mae_w)
        shift_h = np.minimum(a_hold, pen_h)
        shift_w = np.minimum(a_wait, pen_w)
        a_hold -= shift_h
        a_close += shift_h
        a_wait -= shift_w
        a_open += shift_w

    total = a_open + a_close + a_hold + a_wait
    mask = total > 0
    a_open[mask] /= total[mask]
    a_close[mask] /= total[mask]
    a_hold[mask] /= total[mask]
    a_wait[mask] /= total[mask]

    out["Pos"] = pos.astype(np.int8)
    out["A_Open"] = a_open.astype(np.float32)
    out["A_Close"] = a_close.astype(np.float32)
    out["A_Hold"] = a_hold.astype(np.float32)
    out["A_Wait"] = a_wait.astype(np.float32)

    return out.reset_index(drop=True)
