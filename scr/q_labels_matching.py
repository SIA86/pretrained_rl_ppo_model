import numpy as np
import pandas as pd
from .backtest_env import BacktestEnv, EnvConfig


"""
1. Execution @ next_open: решения на баре t исполняются на Open[t+1]. Это обеспечивает корректную причинность и реалистичный бэктест.
2. One-side + действия: работаем отдельно long-only или short-only; действия — Open / Close / Hold / Wait.
Реверсы исключаем (реверс = Close → Open на следующем баре).
3. Абсолютные Q-метки (а не Sharpe-подобные): мы целимся в последующий PPO, где важно сохранить «масштаб выигрыша» действий.
Поэтому никаких делений на σ, √Δt или построчных нормировок — они ломают сопоставимость Close vs Hold/Open.
4. Единая временная опора для сравнения: в позиции сравниваем из t+1:
Close = 0 (после закрытия прирост дальше равен нулю),
Hold = Ret(t+1 → TGT). Это устраняет «ранний выход на полпути» из-за некорректных горизонтов и даёт стабильные решения.
5. Разметка TD-λ: смесь n-step горизонтов (TF-style, TD-λ). Снижает дисперсию меток и согласуется с тем, как PPO/A2C обрабатывают возвраты.
6. Нормализация/клип:
  допускается только глобальный winsorize/клип/робаст-скейл по train (единый для всех действий), чтобы стабилизировать обучение и
  подобрать разумную температуру τ для softmax. Не применять построчные/волатильностные нормировки.
7. Таргеты для SL → PPO:
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
# Симуляция позиции через BacktestEnv
# ------------------------------------------------------------


def _simulate_positions_via_env(
    df: pd.DataFrame,
    side_long: bool,
    fee: float,
    slippage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Прогоняет действия с максимальным Q через ``BacktestEnv``.

    Возвращает кортеж массивов ``(pos, entry_price)`` длины ``len(df)``
    со состоянием позиции *перед* выполнением каждого шага.
    """
    cfg = EnvConfig(
        mode=1 if side_long else -1,
        fee=fee,
        spread=slippage,
        leverage=1.0,
        max_steps=len(df) - 1,
        reward_scale=1.0,
        use_log_reward=False,
        time_penalty=0.0,
        hold_penalty=0.0,
    )
    env = BacktestEnv(df[["Open", "High", "Low", "Close"]], price_col="Open", cfg=cfg)
    env.reset()

    positions = []
    entries = []
    q_mat = df[["Q_Open", "Q_Close", "Q_Hold", "Q_Wait"]].to_numpy(np.float64)
    for row in q_mat[:-1]:
        positions.append(env.position)
        entries.append(env.entry_price if env.position != 0 else np.nan)
        mask = env.action_mask().astype(bool)
        logits = np.where(np.isfinite(row), row, -1e9)
        priority = np.array([2e-12, 3e-12, 4e-12, 1e-12])
        logits = np.where(mask, logits + priority, -1e9)
        action = int(np.argmax(logits))
        _, _, done, _ = env.step(action)
        if done:
            break

    positions.append(env.position)
    entries.append(env.entry_price if env.position != 0 else np.nan)
    return np.array(positions, dtype=np.int8), np.array(entries, dtype=np.float64)


# ------------------------------------------------------------
# Основная разметка Q по TD-λ смеси n-step горизонтов
# ------------------------------------------------------------


def enrich_q_labels_trend_one_side(
    df: pd.DataFrame,
    side_long: bool = True,
    H_max: int = 60,
    lam: float = 0.9,
    fee: float = 0.0002,
    slippage: float = 0.0001,
) -> pd.DataFrame:
    """TD-λ разметка абсолютных Q для действий Open/Close/Hold/Wait.

    Позиция и маски валидности определяются через симуляцию на
    ``BacktestEnv`` с выбором действия максимального Q на каждом шаге.
    """
    need = {"Open", "High", "Low", "Close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"нет колонок: {sorted(miss)}")

    out = df.copy()
    Open = out["Open"].to_numpy(np.float64)
    High = out["High"].to_numpy(np.float64)
    Low = out["Low"].to_numpy(np.float64)
    Close = out["Close"].to_numpy(np.float64)
    n = len(Open)

    # исполнение "сейчас" — next open
    exec_next_open = np.full(n, np.nan)
    exec_next_open[:-1] = Open[1:]

    # комиссии
    c_open = fee + slippage
    c_close = fee + slippage

    has_next = np.arange(n) < n - 1

    # контейнеры Q
    Q_Open = np.full(n, np.nan, np.float64)
    Q_Close = np.full(n, np.nan, np.float64)
    Q_Hold = np.full(n, np.nan, np.float64)
    Q_Wait = np.zeros(n, np.float64)

    # TD(λ): смесь n-step горизонтов (TF-style)
    Hm = int(max(1, H_max))
    fut = np.full((n, Hm), np.nan, dtype=np.float64)
    for j in range(1, Hm + 1):
        idx = np.arange(n) + j
        ok = idx < n
        fut[ok, j - 1] = Open[idx[ok]]
    w = np.array(
        [(1.0 - lam) * (lam ** (k - 1)) for k in range(1, Hm + 1)], dtype=np.float64
    )
    w = w / np.sum(w)
    # Open: смесь n-step, если есть будущее
    m_open = has_next
    if np.any(m_open):
        if side_long:
            entry_eff = (exec_next_open[m_open] * (1.0 + c_open))[:, None]
            exit_eff = fut[m_open, :] * (1.0 - c_close)
            vals = exit_eff / entry_eff - 1.0
        else:
            entry_eff = (exec_next_open[m_open] * (1.0 - c_open))[:, None]
            exit_eff = fut[m_open, :] * (1.0 + c_close)
            vals = entry_eff / exit_eff - 1.0
        mask_cols = np.isfinite(vals)
        vals[~mask_cols] = np.nan
        weights = np.broadcast_to(w, vals.shape).copy()
        weights[~mask_cols] = 0.0
        denom = weights.sum(axis=1, keepdims=True)
        valid_rows = denom.squeeze(1) > 0.0
        denom[denom == 0.0] = np.nan
        q_vals = np.nansum(vals * weights, axis=1) / denom.squeeze(1)
        idx_rows = np.where(m_open)[0]
        Q_Open[idx_rows[valid_rows]] = q_vals[valid_rows]

    # Close: 0, если есть следующий бар
    Q_Close[has_next] = 0.0

    # Hold: смесь n-step, если есть будущее
    m_hold = has_next
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

    # Итоговые Q-колонки (масштабируем для стабильности)
    out["Q_Open"] = Q_Open.astype(np.float32) / 2e-3
    out["Q_Close"] = Q_Close.astype(np.float32) / 2e-3
    out["Q_Hold"] = Q_Hold.astype(np.float32) / 2e-3
    out["Q_Wait"] = Q_Wait.astype(np.float32) / 2e-3

    # Симуляция позиции по максимальному Q
    pos, entry_eff = _simulate_positions_via_env(out, side_long, fee, slippage)

    # Метрики текущей позиции
    unreal, flat_steps, hold_steps, drawdown = calc_position_metrics(
        pos, entry_eff, High, Low, Close, side_long=side_long
    )
    out["Pos"] = pos
    out["Unreal_PnL"] = unreal.astype(np.float32)
    out["Flat_Steps"] = (flat_steps / 1000).astype(np.float32)
    out["Hold_Steps"] = (hold_steps / 1000).astype(np.float32)
    out["Drawdown"] = drawdown.astype(np.float32)

    # Маски валидности на основе симулированной позиции
    flat = pos == 0
    inpos = pos != 0
    valid_open = np.isfinite(Q_Open)
    valid_hold = np.isfinite(Q_Hold)
    valid_close = np.isfinite(Q_Close)

    M_Open = (flat & has_next & valid_open).astype(np.int8)
    M_Close = (inpos & has_next & valid_close).astype(np.int8)
    M_Hold = (inpos & has_next & valid_hold).astype(np.int8)
    M_Wait = flat.astype(np.int8)

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
