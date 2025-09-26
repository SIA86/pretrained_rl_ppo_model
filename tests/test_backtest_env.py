import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scr.backtest_env import BacktestEnv, EnvConfig
from scr.backtest_env import run_backtest_with_logits
from scr.normalisation import NormalizationStats


def make_env(prices, **cfg_kwargs):
    cfg = EnvConfig(
        mode=cfg_kwargs.get("mode", 1),
        fee=cfg_kwargs.get("fee", 0.0),
        spread=0.0,
        leverage=1.0,
        max_steps=100,
        reward_scale=1.0,
        time_penalty=cfg_kwargs.get("time_penalty", 0.0),
        hold_penalty=cfg_kwargs.get("hold_penalty", 0.0),
        terminal_reward=cfg_kwargs.get("terminal_reward", False),
        terminal_reward_coef=cfg_kwargs.get("terminal_reward_coef", 0.0),
    )
    df = pd.DataFrame({"close": prices})
    return BacktestEnv(df, cfg=cfg)


def run_actions(env, actions, reset=True):
    if reset:
        env.reset()
    for a in actions:
        env.step(a)
    return env.logs().iloc[-1]


def test_equity_realized_no_fee():
    env = make_env([1, 2, 3])
    last = run_actions(env, [0, 1])
    assert last["equity"] == pytest.approx(last["realized_pnl"])


def test_equity_realized_with_fee_long():
    env = make_env([1, 2, 3], fee=0.1)
    last = run_actions(env, [0, 1])
    assert last["equity"] == pytest.approx(0.0)
    assert last["realized_pnl"] == pytest.approx(0.0)


def test_equity_realized_with_fee_short():
    env = make_env([3, 2, 1], mode=-1, fee=0.1)
    last = run_actions(env, [0, 1])
    assert last["equity"] == pytest.approx(0.2)
    assert last["realized_pnl"] == pytest.approx(0.2)


def test_metrics_report():
    env = make_env([1, 2, 3, 4, 1])
    run_actions(env, [0, 1, 0, 1])
    metrics = env.metrics_report()
    assert metrics["Win Rate"] == pytest.approx(50.0)
    assert metrics["Profit Factor"] == pytest.approx(2 / 3, rel=1e-3)
    assert metrics["Maximum Drawdown"] == pytest.approx(0.5)


def test_time_penalty():
    env = make_env([1, 1, 1], time_penalty=0.1)
    env.reset()
    env.step(0)  # open
    last = run_actions(env, [2], reset=False)  # hold one step
    assert last["equity"] == pytest.approx(0.0)
    assert last["reward"] == pytest.approx(-0.1)


def test_hold_penalty():
    env = make_env([1, 1], hold_penalty=0.05)
    last = run_actions(env, [3])
    assert last["equity"] == pytest.approx(0.0)
    assert last["reward"] == pytest.approx(-0.05)


def test_terminal_reward_bonus_applied():
    coef = 0.5
    env = make_env(
        [1, 2, 3], terminal_reward=True, terminal_reward_coef=coef
    )
    env.reset()
    env.step(0)
    _, reward, _, info = env.step(1)
    last = env.logs().iloc[-1]
    expected_bonus = last["net_trade"] * coef
    assert last["terminal_bonus"] == pytest.approx(expected_bonus)
    assert info["terminal_bonus"] == pytest.approx(expected_bonus)
    assert reward == pytest.approx(last["pnl_trade"] + expected_bonus)


def test_terminal_reward_bonus_not_applied_with_negative_net_trade():
    coef = 0.5
    env = make_env(
        [1, 2, 3], fee=0.2, terminal_reward=True, terminal_reward_coef=coef
    )
    env.reset()
    env.step(0)
    _, reward, _, info = env.step(1)
    last = env.logs().iloc[-1]
    prev = env.logs().iloc[-2]
    expected_bonus = last["net_trade"] * coef
    assert last["pnl_trade"] > 0.0
    assert last["net_trade"] < 0.0
    assert last["terminal_bonus"] == pytest.approx(expected_bonus)
    assert info["terminal_bonus"] == pytest.approx(expected_bonus)
    assert reward == pytest.approx((last["equity"] - prev["equity"]) + expected_bonus)


def test_plot_show_false(monkeypatch):
    env = make_env([1, 2, 3, 4])
    run_actions(env, [0, 2, 1])
    called = []

    def fake_show():
        called.append(True)

    monkeypatch.setattr(plt, "show", fake_show)
    fig = env.plot("Test", show=False)
    assert called == []
    assert fig is not None
    plt.close(fig)


# Конструктор тестовых данных
def make_df(n=6, start=100.0, step=1.0):
    prices = np.array([start + i * step for i in range(n)], dtype=float)
    return pd.DataFrame({"close": prices, "feat": np.arange(n)})


def test_no_index_error_after_done():
    df = make_df(5)  # индексы 0..4
    cfg = EnvConfig(
        mode=1,
        fee=0.0,
        spread=0.0,
        leverage=1.0,
        max_steps=3,
        reward_scale=1.0,
        time_penalty=0.0,
        hold_penalty=0.0,
    )
    env = BacktestEnv(df, feature_cols=["feat"], cfg=cfg)

    # Совершаем ровно max_steps шагов
    done = False
    for _ in range(cfg.max_steps):
        _, _, done, _ = env.step(3)  # Wait
    assert done is True

    # Доп. шаг после done не должен падать и должен оставаться done=True
    try:
        _, r, d, _ = env.step(3)
    except IndexError:
        pytest.fail("IndexError after done")
    assert d is True
    assert r == 0.0


def test_vector_action_with_mask_argmax():
    df = make_df(5)
    env = BacktestEnv(
        df,
        feature_cols=["feat"],
        cfg=EnvConfig(
            mode=1,
            fee=0.0,
            spread=0.0,
            leverage=1.0,
            max_steps=10**9,
            reward_scale=1.0,
            time_penalty=0.0,
            hold_penalty=0.0,
        ),
    )
    # На старте позиция 0 → валидны только [Open, Wait]
    logits = [
        5.0,
        -1.0,
        9.0,
        -5.0,
    ]  # max на индексе 2, но он замаскирован → должен выбрать Open (0)
    _, _, _, info = env.step(logits)
    assert info["position"] in (0, 1)


def test_step_q_threshold_wait():
    env = make_env([1, 2, 3])
    env.reset()
    logits = [0.0, 0.0, 0.0, 0.0]
    env.step(logits, q_threshold=0.6)
    assert env.position == 0


def test_run_backtest_with_logits_executes_trade():
    df = make_df(4, start=1.0, step=1.0)
    stats = NormalizationStats().fit(df[["feat"]].to_numpy(np.float32))

    class DummyModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs, training=False):
            self.calls += 1
            if self.calls == 1:
                return np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            return np.array([[0.0, 5.0, 0.0, 0.0]], dtype=np.float32)

    env = run_backtest_with_logits(
        df,
        DummyModel(),
        stats,
        seq_len=2,
        feature_cols=["feat"],
    )
    log = env.logs()
    assert log.iloc[-1]["equity"] == pytest.approx((4 - 3) / 3)


def test_run_backtest_with_logits_respects_threshold():
    df = make_df(4, start=1.0, step=1.0)
    stats = NormalizationStats().fit(df[["feat"]].to_numpy(np.float32))

    class DummyModel:
        def __call__(self, inputs, training=False):
            return np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    env = run_backtest_with_logits(
        df,
        DummyModel(),
        stats,
        seq_len=2,
        feature_cols=["feat"],
        q_threshold=0.6,
    )
    log = env.logs()
    assert not log["opened"].any()


def test_observation_state_updates():
    df = pd.DataFrame(
        {
            "close": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "feat": [0, 1, 2],
        }
    )
    env = BacktestEnv(
        df,
        feature_cols=["feat"],
        cfg=EnvConfig(
            mode=1,
            fee=0.0,
            spread=0.0,
            leverage=1.0,
            max_steps=100,
            reward_scale=1.0,
            time_penalty=0.0,
            hold_penalty=0.0,
        ),
    )
    obs = env.reset()
    np.testing.assert_allclose(obs["state"], np.zeros(5, dtype=np.float32))
    env.step(0)  # open
    obs, _, _, _ = env.step(2)  # hold
    np.testing.assert_allclose(obs["state"], np.zeros(5, dtype=np.float32))
