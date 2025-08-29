"""Utilities for visualising strategy signals and Q‑labelled actions."""

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def _candles(
    ax,
    t: Iterable[int],
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    width: float = 0.6,
    up: str = "#26a69a",
    dn: str = "#ef5350",
    alpha: float = 1.0,
) -> None:
    """Render candlesticks on ``ax``."""
    for i in range(len(t)):
        color = up if c[i] >= o[i] else dn
        ax.vlines(t[i], l[i], h[i], linewidth=1.0, color=color, alpha=alpha)
        y = min(o[i], c[i])
        height = abs(c[i] - o[i])
        if height == 0:
            ax.hlines(
                y,
                t[i] - width / 2,
                t[i] + width / 2,
                colors=color,
                linewidth=2.0,
                alpha=alpha,
            )
        else:
            ax.add_patch(
                Rectangle(
                    (t[i] - width / 2, y),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=alpha,
                )
            )


def actions_from_pos_one_side(
    pos: np.ndarray, assume_exec_next_bar: bool = True
) -> Tuple[list[int], list[int]]:
    """Return indices of open/close signals for a single‑side position array."""
    n = len(pos)
    shift = 1 if assume_exec_next_bar else 0
    open_ix, close_ix = [], []
    for t in range(n - 1):
        a, b = pos[t], pos[t + 1]
        ix = t + shift
        if ix < 0 or ix >= n:
            continue
        if a == 0 and b != 0:
            open_ix.append(ix)
        elif a != 0 and b == 0:
            close_ix.append(ix)
        elif a != 0 and b != 0 and np.sign(a) != np.sign(b):
            close_ix.append(ix)
            open_ix.append(ix)
    return open_ix, close_ix


def _resolve_series(
    df: pd.DataFrame,
    src,
    N: int,
    label: str,
    align_mode: str = "right",
) -> np.ndarray:
    """Align an indicator series to length ``N`` for plotting."""
    if isinstance(src, str):
        if src not in df.columns:
            raise ValueError(f"indicator '{label}': column '{src}' not found")
        y = df[src].to_numpy(float)
        return y

    if isinstance(src, pd.Series):
        s = src.astype(float).reindex(df.index)
        return s.to_numpy()

    y = np.asarray(src, dtype=float).ravel()
    m = y.shape[0]

    if m == N:
        return y

    if align_mode not in ("right", "left"):
        raise ValueError("align_mode must be 'right' or 'left'")

    if m > N:
        return y[-N:] if align_mode == "right" else y[:N]

    pad = np.full(N - m, np.nan, dtype=float)
    return (
        np.concatenate([pad, y]) if align_mode == "right" else np.concatenate([y, pad])
    )


def plot_enriched_actions_one_side(
    enriched_df: pd.DataFrame,
    *,
    side_long: bool = True,
    start: int = 0,
    end: int | None = None,
    title: str = "Q-разметка (one-side)",
    show_reference: bool = True,
    q_threshold: float | None = None,
    indicators_price: dict | None = None,
    indicators_panels: dict | None = None,
    assume_exec_next_bar: bool = True,
) -> None:
    """Plot price with strategy signals and best actions based on Q labels.

    ``enriched_df`` must contain columns:
    ['Open','High','Low','Close','Pos','Q_Open','Q_Close','Q_Hold','Q_Wait'].
    """
    need = {
        "Open",
        "High",
        "Low",
        "Close",
        "Pos",
        "Q_Open",
        "Q_Close",
        "Q_Hold",
        "Q_Wait",
    }
    miss = need - set(enriched_df.columns)
    if miss:
        raise ValueError(f"missing columns: {sorted(miss)}")

    if end is None:
        end = len(enriched_df)
    assert 0 <= start < end <= len(enriched_df)

    df = enriched_df.iloc[start:end].copy()
    N = len(df)
    t = np.arange(N)
    o, h, l, c = [df[col].to_numpy(float) for col in ("Open", "High", "Low", "Close")]

    n_extra = len(indicators_panels) if indicators_panels else 0
    n_rows = 2 + n_extra
    height_ratios = [2.0, 1.6] + [1.2] * n_extra

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(16, 6 + 2.2 * n_rows),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = [axes]
    ax_price = axes[0]
    ax_actions = axes[1] if n_rows >= 2 else None
    ax_extras = axes[2:] if n_extra > 0 else []

    _candles(ax_price, t, o, h, l, c, alpha=0.9)

    if indicators_price:
        for label, src in indicators_price.items():
            y = _resolve_series(df, src, N, label, align_mode="right")
            ax_price.plot(t, y, linewidth=1.3, label=label)

    action_handles = []
    if show_reference:
        pos = df["Pos"].to_numpy(np.int8)
        open_ix, close_ix = actions_from_pos_one_side(
            pos, assume_exec_next_bar=assume_exec_next_bar
        )

        if side_long:
            if open_ix:
                ax_price.scatter(
                    open_ix, l[open_ix] * 0.998, marker="^", s=70, c="g", zorder=3
                )
                action_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="w",
                        markerfacecolor="g",
                        markersize=8,
                        label="Ref: Open (Long)",
                    )
                )
            if close_ix:
                mid = (o[close_ix] + c[close_ix]) * 0.5
                ax_price.scatter(close_ix, mid, marker="v", s=70, c="k", zorder=3)
                action_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="v",
                        color="k",
                        markersize=8,
                        label="Ref: Close (Long)",
                    )
                )
        else:
            if open_ix:
                ax_price.scatter(
                    open_ix, h[open_ix] * 1.002, marker="v", s=70, c="r", zorder=3
                )
                action_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="v",
                        color="w",
                        markerfacecolor="r",
                        markersize=8,
                        label="Ref: Open (Short)",
                    )
                )
            if close_ix:
                mid = (o[close_ix] + c[close_ix]) * 0.5
                ax_price.scatter(close_ix, mid, marker="^", s=70, c="k", zorder=3)
                action_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="k",
                        markersize=8,
                        label="Ref: Close (Short)",
                    )
                )

        ax1_twin = ax_price.twinx()
        ax1_twin.set_ylim(-1.2, 1.2)
        ax1_twin.plot(t, pos, alpha=0.18, linewidth=1.1, color="tab:purple")
        ax1_twin.set_yticks([-1, 0, 1])
        ax1_twin.set_yticklabels(["-1", "0", "1"])
        ax1_twin.grid(False)

    ax_price.set_title(title)
    if (indicators_price and len(indicators_price)) or action_handles:
        ax_price.legend(loc="upper left")
        if action_handles:
            leg = ax_price.get_legend()
            for hndl in action_handles:
                leg._legend_box._children[0]._children.append(hndl)
    ax_price.grid(alpha=0.25)

    if ax_actions is not None:
        _candles(ax_actions, t, o, h, l, c, alpha=0.35)
        ax_actions.grid(alpha=0.25)

        q_open = df["Q_Open"].to_numpy(float)
        q_close = df["Q_Close"].to_numpy(float)
        q_hold = df["Q_Hold"].to_numpy(float)
        q_wait = df["Q_Wait"].to_numpy(float)

        qs = np.vstack(
            [
                np.where(np.isnan(q_open), -np.inf, q_open),
                np.where(np.isnan(q_close), -np.inf, q_close),
                np.where(np.isnan(q_hold), -np.inf, q_hold),
                np.where(np.isnan(q_wait), -np.inf, q_wait),
            ]
        )

        best_idx = np.argmax(qs, axis=0)
        best_q = np.take_along_axis(qs, best_idx[None, :], axis=0).ravel()
        mask_thr = (
            best_q >= q_threshold
            if (q_threshold is not None)
            else np.ones_like(best_q, dtype=bool)
        )

        ix_open = np.where((best_idx == 0) & mask_thr)[0]
        ix_close = np.where((best_idx == 1) & mask_thr)[0]
        ix_hold = np.where((best_idx == 2) & mask_thr)[0]
        ix_wait = np.where((best_idx == 3) & mask_thr)[0]

        if side_long:
            ax_actions.scatter(
                ix_open, l[ix_open] * 0.998, marker="^", s=55, label="OPEN"
            )
            ax_actions.scatter(
                ix_close,
                (o[ix_close] + c[ix_close]) * 0.5,
                marker="v",
                s=55,
                label="CLOSE",
            )
        else:
            ax_actions.scatter(
                ix_open, h[ix_open] * 1.002, marker="v", s=55, label="OPEN"
            )
            ax_actions.scatter(
                ix_close,
                (o[ix_close] + c[ix_close]) * 0.5,
                marker="^",
                s=55,
                label="CLOSE",
            )

        ax_actions.scatter(ix_hold, h[ix_hold] * 1.002, marker="o", s=45, label="HOLD")
        ax_actions.scatter(ix_wait, l[ix_wait] * 0.998, marker="o", s=45, label="WAIT")
        ax_actions.legend(loc="upper left")

    if indicators_panels:
        for ax, (panel_name, series_dict) in zip(ax_extras, indicators_panels.items()):
            is_bar = len(series_dict) == 1 and next(iter(series_dict)).lower() in (
                "bar",
                "hist",
                "volume",
                "vol",
            )
            if is_bar:
                label, src = next(iter(series_dict.items()))
                y = _resolve_series(df, src, N, label, align_mode="right")
                ax.bar(t, y, width=0.8, alpha=0.7, label=label)
            else:
                for label, src in series_dict.items():
                    y = _resolve_series(df, src, N, label, align_mode="right")
                    ax.plot(t, y, linewidth=1.2, label=label)
            ax.set_ylabel(panel_name)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


__all__ = [
    "_candles",
    "actions_from_pos_one_side",
    "_resolve_series",
    "plot_enriched_actions_one_side",
]
