import joblib

from scr.check_data import prepare_time_series
from scr.q_labels_matching import enrich_q_labels_trend_one_side
from scr.dataset_builder import DatasetBuilderForYourColumns, NUM_CLASSES
from scr.residual_lstm import build_backbone, build_head
from scr.train_eval import fit_model
from scr.calibrate import calibrate_model
from scr.backtest_env import run_backtest_with_logits, DEFAULT_CONFIG
from scr.ppo_training import train as ppo_train


def main() -> None:
    df = joblib.load("dummy_df.joblib")
    df = prepare_time_series(df, "timestamp", tz="UTC")
    df = enrich_q_labels_trend_one_side(df, H_max=3)
    df = df.rename(
        columns={
            "Unreal_PnL": "un_pnl",
            "Flat_Steps": "flat_steps",
            "Hold_Steps": "hold_steps",
            "Drawdown": "drawdown",
        }
    )

    builder = DatasetBuilderForYourColumns(
        seq_len=5,
        feature_cols=["Open", "High", "Low", "Close", "Volume"],
        account_cols=["Pos", "un_pnl", "flat_steps", "hold_steps", "drawdown"],
        norm="none",
        splits=(0.6, 0.2, 0.2),
        batch_size=32,
    )
    splits = builder.fit_transform(df, return_indices=True)
    ds_tr, ds_va, _ = builder.as_tf_datasets(splits)

    backbone = build_backbone(
        seq_len=5,
        feature_dim=len(builder.feature_names) + len(builder.account_names),
        units_per_layer=(8, 8),
    )
    model = build_head(backbone, NUM_CLASSES)
    fit_model(
        model,
        ds_tr,
        ds_va,
        epochs=1,
        steps_per_epoch=1,
        lr=1e-3,
        best_path="sl_weights/best_lstm.weights.h5",
    )
    backbone.save_weights("sl_weights/best_backbone.weights.h5")

    calibrate_model(model, ds_va, plot=False, batch_size=32, save_dir=".")

    idx = splits["test"][-1]
    start = int(idx[0]) if len(idx) else builder.seq_len
    env = run_backtest_with_logits(
        df,
        model,
        feature_stats=builder.stats_features,
        seq_len=5,
        start=start,
        feature_cols=builder.feature_names,
        price_col="Close",
        cfg=DEFAULT_CONFIG,
        show_progress=False,
    )
    print(env.metrics_report())

    cfg = DEFAULT_CONFIG._replace(max_steps=10)
    df_ppo = df.drop(columns=["Pos", "un_pnl", "flat_steps", "hold_steps", "drawdown"])
    ppo_train(
        df_ppo,
        cfg,
        seq_len=5,
        teacher_weights="sl_weights/best_lstm.weights.h5",
        backbone_weights="sl_weights/best_backbone.weights.h5",
        save_path="ppo_weights",
        num_actions=4,
        units_per_layer=[8, 8],
        dropout=0.2,
        updates=1,
        n_env=1,
        rollout=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.01,
        epochs=1,
        batch_size=1,
        teacher_kl=0.1,
        kl_decay=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        val_interval=1,
    )
    print("PPO training completed")


if __name__ == "__main__":
    main()
