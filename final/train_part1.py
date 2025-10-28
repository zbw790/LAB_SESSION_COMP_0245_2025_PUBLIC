# train_mlp_from_rollouts.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取器：确保你把 rollout_loader.py 放在 final/ 目录下（与本脚本同仓库）
from final.rollout_loader import load_rollouts


def per_joint_metrics(y: np.ndarray, yp: np.ndarray):
    """Return (mse, mae, r2) per joint with shape [7]."""
    jmse = ((y - yp) ** 2).mean(axis=0)
    jmae = np.abs(y - yp).mean(axis=0)
    # r2_score 支持 multioutput="raw_values"
    jr2 = r2_score(y, yp, multioutput="raw_values")
    return jmse, jmae, jr2


def build_model(random_state: int = 42) -> TransformedTargetRegressor:
    base = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=256,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=False,
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline([("xsc", StandardScaler()), ("mlp", base)]),
        transformer=StandardScaler(),
    )
    return model


def load_xy_from_rollouts(count: int | None,
                          indices: list[int] | None,
                          directory: Path | None):
    rollouts = load_rollouts(count=count, indices=indices, directory=directory)

    X_list, Y_list = [], []
    for r in rollouts:
        q_mes_all = np.asarray(r.q_mes_all, dtype=float)    # [T,7]
        qd_mes_all = np.asarray(r.qd_mes_all, dtype=float)  # [T,7]
        q_des_all = np.asarray(r.q_d_all, dtype=float)      # [T,7]
        qd_des_all = np.asarray(r.qd_d_all, dtype=float)    # [T,7]
        tau_all = np.asarray(r.tau_cmd_all, dtype=float)    # [T,7]

        err = q_des_all - q_mes_all
        derr = qd_des_all - qd_mes_all
        X = np.concatenate([err, derr], axis=1)  # [T,14]

        X_list.append(X)
        Y_list.append(tau_all)

        print(f"[Load] idx={r.idx} | T={X.shape[0]} | feat_dim={X.shape[1]} | path={r.path}")

    X = np.vstack(X_list)  # [N,14]
    Y = np.vstack(Y_list)  # [N,7]
    print(f"[Data] X={X.shape}, Y={Y.shape}")
    return X, Y


def evaluate_split(name: str, model, Xs, Ys):
    Yp = model.predict(Xs)
    mse = mean_squared_error(Ys, Yp)
    mae = mean_absolute_error(Ys, Yp)
    r2 = r2_score(Ys, Yp)
    jmse, jmae, jr2 = per_joint_metrics(Ys, Yp)

    print(f"[{name}] MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
    print("        per-joint R2: " + "  ".join([f"J{i+1}:{r:.3f}" for i, r in enumerate(jr2)]))
    return {"mse": mse, "mae": mae, "r2": r2, "jmse": jmse, "jmae": jmae, "jr2": jr2}


def main():
    parser = argparse.ArgumentParser(description="Train MLP torque regressor from rollout PKLs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--count", type=int, default=4, help="Load indices 0..count-1 (默认 4 个：0,1,2,3)")
    group.add_argument("--indices", type=int, nargs="+", help="指定索引，如: --indices 0 1 2 3")

    parser.add_argument("--directory", type=str, default=None,
                        help="pkl 所在目录（默认：final 包所在目录）")
    parser.add_argument("--model-out", type=str, default="models/model_part1.joblib",
                        help="保存模型路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    dir_path = Path(args.directory) if args.directory else None
    model_out = Path(args.model_out)

    print("========== Loading ==========")
    X, Y = load_xy_from_rollouts(count=args.count, indices=args.indices, directory=dir_path)

    print("========== Split ==========")
    # 70 / 15 / 15
    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X, Y, test_size=0.30, random_state=args.seed, shuffle=True)
    X_va, X_te, Y_va, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=args.seed, shuffle=True)
    print(f"[Split] train={X_tr.shape[0]} | val={X_va.shape[0]} | test={X_te.shape[0]}")

    print("========== Training ==========")
    model = build_model(random_state=args.seed)
    model.fit(X_tr, Y_tr)

    print("========== Evaluation ==========")
    evaluate_split("Train", model, X_tr, Y_tr)
    evaluate_split("Val", model, X_va, Y_va)
    evaluate_split("Test", model, X_te, Y_te)

    print("========== Saving ==========")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"[OK] saved model -> {model_out.resolve()}")


if __name__ == "__main__":
    main()
