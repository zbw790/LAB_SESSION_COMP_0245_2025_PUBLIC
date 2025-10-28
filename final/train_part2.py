# train_cartdiffkin_models.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from final.rollout_loader import load_rollouts


# ---------------- utils ----------------
def stack_xy(indices: list[int] | None, directory: Path | None):
    """
    X = [q_mes(7), target_xyz(3)]
    Y = [q_d(7), qd_d(7)]
    """
    rls = load_rollouts(indices=indices, directory=directory, strict_missing=True)

    Xs, Ys, groups = [], [], []
    for r in rls:
        q   = np.asarray(r.q_mes_all, float)     # [T,7]
        q_d = np.asarray(r.q_d_all, float)       # [T,7]
        qd_d = np.asarray(r.qd_d_all, float)     # [T,7]

        # 该段的目标点（用末帧末端位置近似 desired_cart_pos）
        target = np.asarray(r.cart_pos_all[-1], float)[:3]  # [3]
        target_rep = np.repeat(target[None, :], q.shape[0], axis=0)  # [T,3]

        X = np.concatenate([q, target_rep], axis=1)   # [T,10]
        Y = np.concatenate([q_d, qd_d], axis=1)       # [T,14]

        Xs.append(X); Ys.append(Y)
        groups += [r.idx] * len(X)

        print(f"[Load] idx={r.idx} | T={len(X)} | Xdim={X.shape[1]} | Ydim={Y.shape[1]} | path={r.path}")

    X = np.vstack(Xs); Y = np.vstack(Ys); groups = np.asarray(groups)
    print(f"[Data] X={X.shape}, Y={Y.shape}, groups(unique)={sorted(set(groups.tolist()))}")
    return X, Y, groups


def per_block_metrics(Y, Yp, block_names=("q_d", "qd_d")):
    """分别对前7维(q_d)与后7维(qd_d)给出指标"""
    res = {}
    for b, name in enumerate(block_names):
        s, e = b * 7, (b + 1) * 7
        r2j = r2_score(Y[:, s:e], Yp[:, s:e], multioutput="raw_values")
        res[name] = dict(
            R2=np.mean(r2j),
            R2_per_joint=r2j,
            MAE=np.mean(np.abs(Y[:, s:e] - Yp[:, s:e])),
            MSE=np.mean((Y[:, s:e] - Yp[:, s:e]) ** 2),
        )
    return res


def pretty_print_scores(tag, Y, Yp):
    mse = mean_squared_error(Y, Yp)
    mae = mean_absolute_error(Y, Yp)
    r2  = r2_score(Y, Yp)
    print(f"[{tag}] overall: MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")

    blk = per_block_metrics(Y, Yp)
    for name, m in blk.items():
        r2_list = "  ".join([f"J{i+1}:{v:.3f}" for i, v in enumerate(m["R2_per_joint"])])
        print(f"    {name:>5}: R2={m['R2']:.4f}  MAE={m['MAE']:.5f}  MSE={m['MSE']:.6f}")
        print(f"          per-joint R2 -> {r2_list}")


# --------------- models ----------------
def build_mlp(seed=42):
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
    # 输入标准化；目标（14维）也标准化再学习
    return TransformedTargetRegressor(
        regressor=Pipeline([("xsc", StandardScaler()), ("mlp", base)]),
        transformer=StandardScaler(),
    )


def build_rf(seed=42):
    # 随机森林支持多输出回归：Y shape = [n, 14]
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )


# --------------- training entry ----------------
def run_random_split(X, Y, seed, model_name):
    # 70 / 15 / 15
    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X, Y, test_size=0.30, random_state=seed, shuffle=True)
    X_va, X_te, Y_va, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=seed, shuffle=True)

    model = build_mlp(seed) if model_name == "mlp" else build_rf(seed)
    model.fit(X_tr, Y_tr)

    print("\n== Random split results ==")
    pretty_print_scores(f"{model_name.upper()}-Train", Y_tr, model.predict(X_tr))
    pretty_print_scores(f"{model_name.upper()}-Val",   Y_va, model.predict(X_va))
    pretty_print_scores(f"{model_name.upper()}-Test",  Y_te, model.predict(X_te))
    return model


def run_group_kfold(X, Y, groups, seed, model_name, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    print("\n== GroupKFold by file ==")
    for k, (tr, te) in enumerate(gkf.split(X, Y, groups=groups), start=1):
        model = build_mlp(seed + k) if model_name == "mlp" else build_rf(seed + k)
        model.fit(X[tr], Y[tr])
        print(f"\n-- Fold {k} --")
        pretty_print_scores(f"{model_name.upper()}-Train", Y[tr], model.predict(X[tr]))
        pretty_print_scores(f"{model_name.upper()}-Test ", Y[te], model.predict(X[te]))

        r2 = r2_score(Y[te], model.predict(X[te]))
        fold_metrics.append(r2)
    print(f"\n[CV {model_name.upper()}] R2 mean={np.mean(fold_metrics):.4f}  std={np.std(fold_metrics):.4f}")


def main():
    ap = argparse.ArgumentParser("Approximate CartesianDiffKin: (q, target_xyz) -> (q_d, qd_d)")
    ap.add_argument("--directory", type=str, default=None, help="pkl 目录（默认 final 包目录）")
    ap.add_argument("--indices", type=int, nargs="+", default=[0,1,2,3], help="哪些文件作为数据源")
    ap.add_argument("--model", choices=["mlp", "rf", "both"], default="both", help="训练哪个模型")
    ap.add_argument("--split", choices=["random", "group"], default="group",
                    help="random=随机切分；group=按文件分组（更公平）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true", help="是否保存已训练模型")
    ap.add_argument("--outdir", type=str, default="models", help="模型保存目录")
    args = ap.parse_args()

    dir_path = Path(args.directory) if args.directory else None
    X, Y, groups = stack_xy(indices=args.indices, directory=dir_path)

    models_to_run = ["mlp", "rf"] if args.model == "both" else [args.model]
    trained = {}

    for m in models_to_run:
        if args.split == "random":
            model = run_random_split(X, Y, args.seed, model_name=m)
            trained[m] = model
        else:
            # 按文件分组进行K折（折数=文件数）
            run_group_kfold(X, Y, groups, args.seed, model_name=m, n_splits=len(set(groups)))

            # 可选：在全部数据上拟合一个用于部署的终模型（更稳）
            model = build_mlp(args.seed) if m == "mlp" else build_rf(args.seed)
            model.fit(X, Y)
            trained[m] = model

    if args.save:
        outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
        for name, model in trained.items():
            path = outdir / f"cartdiffkin_{name}.joblib"
            joblib.dump(model, path)
            print(f"[OK] saved -> {path.resolve()}")


if __name__ == "__main__":
    main()
