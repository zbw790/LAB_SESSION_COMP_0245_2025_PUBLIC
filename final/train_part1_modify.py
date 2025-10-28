# train_cv4_from_rollouts.py
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

from final.rollout_loader import load_rollouts


def per_joint_metrics(y: np.ndarray, yp: np.ndarray):
    jmse = ((y - yp) ** 2).mean(axis=0)
    jmae = np.abs(y - yp).mean(axis=0)
    jr2 = r2_score(y, yp, multioutput="raw_values")
    return jmse, jmae, jr2


def build_model(seed: int = 42) -> TransformedTargetRegressor:
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
        random_state=seed,
        verbose=False,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("xsc", StandardScaler()), ("mlp", base)]),
        transformer=StandardScaler(),
    )


def to_xy(rollouts):
    Xs, Ys = [], []
    for r in rollouts:
        q = np.asarray(r.q_mes_all, dtype=float)
        qd = np.asarray(r.qd_mes_all, dtype=float)
        qd_des = np.asarray(r.qd_d_all, dtype=float)
        q_des = np.asarray(r.q_d_all, dtype=float)
        tau = np.asarray(r.tau_mes_all, dtype=float)

        err = q_des - q
        derr = qd_des - qd
        X = np.concatenate([err, derr], axis=1)  # [T,14]
        Xs.append(X); Ys.append(tau)
    return np.vstack(Xs), np.vstack(Ys)


def evaluate(name: str, model, X, Y):
    Yp = model.predict(X)
    mse = mean_squared_error(Y, Yp)
    mae = mean_absolute_error(Y, Yp)
    r2 = r2_score(Y, Yp)
    jmse, jmae, jr2 = per_joint_metrics(Y, Yp)
    print(f"[{name}] MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
    print("        per-joint R2: " + "  ".join([f"J{i+1}:{r:.3f}" for i, r in enumerate(jr2)]))
    return dict(mse=mse, mae=mae, r2=r2, jmse=jmse, jmae=jmae, jr2=jr2)


def run_fold(all_rollouts, test_idx: int, seed: int):
    test_ro = [r for r in all_rollouts if r.idx == test_idx]
    train_ro = [r for r in all_rollouts if r.idx != test_idx]
    assert len(test_ro) == 1 and len(train_ro) >= 1

    X_tr_all, Y_tr_all = to_xy(train_ro)
    X_te, Y_te = to_xy(test_ro)

    # 从训练里再切一部分做验证（保持早停稳定）；不动测试集
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tr_all, Y_tr_all, test_size=0.15, random_state=seed, shuffle=True
    )

    model = build_model(seed)
    model.fit(X_tr, Y_tr)

    print(f"\n===== Fold test_idx={test_idx} =====")
    evaluate("Train(subset)", model, X_tr, Y_tr)
    evaluate("Val(from train)", model, X_va, Y_va)
    te_metrics = evaluate("Test(Held-out file)", model, X_te, Y_te)
    return te_metrics


def main():
    ap = argparse.ArgumentParser("4-fold CV where each pkl is a fold (leave-one-file-out).")
    ap.add_argument("--directory", type=str, default=None, help="pkl 所在目录（默认：final 包目录）")
    ap.add_argument("--indices", type=int, nargs="+", default=[0,1,2,3],
                    help="作为折的文件索引，例如 0 1 2 3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-final-model", action="store_true",
                    help="在全部数据上再训练一个最终模型并保存")
    ap.add_argument("--model-out", type=str, default="models/model_part1.joblib")
    args = ap.parse_args()

    dir_path = Path(args.directory) if args.directory else None

    # 读取所有折
    rollouts = load_rollouts(indices=args.indices, directory=dir_path, strict_missing=True)
    print("Loaded rollouts:", [r.idx for r in rollouts])

    # 逐折评估
    fold_results = []
    for k in args.indices:
        res = run_fold(rollouts, test_idx=k, seed=args.seed)
        fold_results.append(res)

    # 汇总
    mse_list = [r["mse"] for r in fold_results]
    mae_list = [r["mae"] for r in fold_results]
    r2_list  = [r["r2"]  for r in fold_results]
    print("\n========== CV Summary ==========")
    print(f"MSE: mean={np.mean(mse_list):.6f}  std={np.std(mse_list):.6f}")
    print(f"MAE: mean={np.mean(mae_list):.6f}  std={np.std(mae_list):.6f}")
    print(f" R2: mean={np.mean(r2_list):.4f}   std={np.std(r2_list):.4f}")

    # 可选：在全部数据上再训一个最终模型并保存
    if args.save_final_model:
        print("\n========== Train on ALL & Save ==========")
        X_all, Y_all = to_xy(rollouts)
        model = build_model(seed=args.seed)
        # 这里直接用全部数据拟合，以便推理时使用；早停依靠内置的训练划分
        X_tr, X_va, Y_tr, Y_va = train_test_split(
            X_all, Y_all, test_size=0.15, random_state=args.seed, shuffle=True
        )
        model.fit(X_tr, Y_tr)
        evaluate("Train(all-subset)", model, X_tr, Y_tr)
        evaluate("Val(all-subset)", model, X_va, Y_va)

        out = Path(args.model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
        print(f"[OK] saved final model -> {out.resolve()}")


if __name__ == "__main__":
    main()
