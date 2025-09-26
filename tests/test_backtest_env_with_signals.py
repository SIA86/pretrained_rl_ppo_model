import numpy as np
import pandas as pd
import pytest

from scr.backtest_env import BacktestEnv, EnvConfig
from scr.backtest_env_with_signals import BacktestEnvWithSignals


def make_df(length: int) -> pd.DataFrame:
    prices = np.linspace(1.0, 1.0 + 0.1 * (length - 1), length)
    return pd.DataFrame({"close": prices})


def make_cfg(**overrides) -> EnvConfig:
    base = dict(
        mode=1,
        fee=0.0,
        spread=0.0,
        leverage=1.0,
        max_steps=100,
        reward_scale=1.0,
        valid_time=0,
        time_penalty=0.0,
        hold_penalty=0.0,
        terminal_reward=False,
        terminal_reward_coef=0.0,
        close_deal_if_done=False,
        only_positive_tr_reward=False,
        n=overrides.get("n", 0),
    )
    base.update(overrides)
    return EnvConfig(**base)


def test_signals_validation_length_mismatch():
    df = make_df(4)
    cfg = make_cfg()
    with pytest.raises(ValueError):
        BacktestEnv(df, cfg=cfg, signals=[1, -1, 1])


def test_signals_validation_value_range():
    df = make_df(4)
    cfg = make_cfg()
    with pytest.raises(ValueError):
        BacktestEnv(df, cfg=cfg, signals=[1, 2, -1, 1])


def test_signals_validation_rejects_zero():
    df = make_df(3)
    cfg = make_cfg()
    with pytest.raises(ValueError):
        BacktestEnv(df, cfg=cfg, signals=[1, 0, -1])


def test_countdown_enables_trading():
    df = make_df(6)
    cfg = make_cfg(n=2)
    signals = [1, 1, 1, 1, 1, 1]
    env = BacktestEnvWithSignals(df, cfg=cfg, signals=signals)
    env.reset()

    # Step 0: countdown starts, trading still disabled
    _, _, _, info = env.step(0)
    assert env.position == 0
    assert info["forced_action"] == 3
    assert info["can_trade"] is False
    assert info["countdown"] == 1

    # Step 1: countdown finishes, trading becomes available on next step
    _, _, _, info = env.step(0)
    assert info["forced_action"] == 3
    assert info["can_trade"] is True
    assert info["countdown"] == 0

    # Step 2: trading allowed, open position
    _, _, _, info = env.step(0)
    assert env.position == 1
    assert info["forced_action"] is None
    assert info["can_trade"] is True


def test_signal_minus_one_forces_closure():
    df = make_df(6)
    cfg = make_cfg(n=1)
    signals = [1, 1, -1, -1, -1, -1]
    env = BacktestEnvWithSignals(df, cfg=cfg, signals=signals)
    env.reset()

    # Step 0: countdown start
    env.step(0)
    # Step 1: countdown expires, position can be opened
    _, _, _, info = env.step(0)
    assert env.can_trade is True
    assert env.position == 1
    assert info["forced_action"] is None

    # Step 2 (signal -1): force close
    _, _, _, info = env.step(2)
    assert env.position == 0
    assert info["forced_action"] == 1
    assert info["can_trade"] is False
    assert info["countdown"] == 0

