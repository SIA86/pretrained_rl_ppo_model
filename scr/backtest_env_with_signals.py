from __future__ import annotations

from typing import Optional, Sequence

from .backtest_env import BacktestEnv, DEFAULT_CONFIG, EnvConfig


class BacktestEnvWithSignals(BacktestEnv):
    """Расширение ``BacktestEnv`` с управлением торговли по сигналам."""

    def __init__(
        self,
        df,
        feature_cols: Optional[Sequence[str]] = None,
        price_col: str = "close",
        cfg: EnvConfig = DEFAULT_CONFIG,
        ppo_true: bool = False,
        record_history: Optional[bool] = None,
        signals: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            df,
            feature_cols=feature_cols,
            price_col=price_col,
            cfg=cfg,
            ppo_true=ppo_true,
            record_history=record_history,
            signals=signals,
        )
        if self.signals is None:
            raise ValueError("signals array is required for BacktestEnvWithSignals")
        if self.cfg.n < 0:
            raise ValueError("cfg.n must be non-negative")
        self._countdown_remaining = 0
        self.can_trade = False

    def reset(self):
        obs = super().reset()
        self._countdown_remaining = 0
        self.can_trade = False
        return obs

    def _pre_step_signal(self) -> tuple[int, Optional[int]]:
        current_signal = int(self.signals[self.t])
        forced_action: Optional[int] = None

        if current_signal == -1:
            self.can_trade = False
            self._countdown_remaining = 0
            forced_action = 1 if self.position != 0 else 3
        elif current_signal == 1 and not self.can_trade:
            if self.cfg.n <= 0:
                self.can_trade = True
            elif self._countdown_remaining == 0:
                self._countdown_remaining = int(self.cfg.n)
        return current_signal, forced_action

    def _finalize_countdown(self, current_signal: int) -> None:
        if current_signal == -1:
            return
        if self.cfg.n > 0 and not self.can_trade and self._countdown_remaining > 0:
            self._countdown_remaining -= 1
            if self._countdown_remaining == 0:
                self.can_trade = True

    def step(self, action, q_threshold: float | None = None):
        current_signal, forced_action = self._pre_step_signal()

        if forced_action is not None:
            effective_action = forced_action
            forced = forced_action
        elif self.can_trade:
            effective_action = action
            forced = None
        else:
            forced = 2 if self.position != 0 else 3
            effective_action = forced

        obs, reward, done, info = super().step(effective_action, q_threshold=q_threshold)

        self._finalize_countdown(current_signal)

        if not self.can_trade:
            obs = dict(obs)
            obs["state"] = self._zero_state.copy()

        info = dict(info)
        info.update(
            signal=current_signal,
            can_trade=self.can_trade,
            countdown=self._countdown_remaining,
            forced_action=forced,
            effective_action=None if forced is None else effective_action,
        )
        return obs, reward, done, info

