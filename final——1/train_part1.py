# final/train_part1.py
import os
import glob
import pickle
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 来自你已有环境（与 data_generator 一致）
from simulation_and_control import pb, PinWrapper, CartesianDiffKin


# ----------------- 固定为你的目录结构 -----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
CONFIG_PATH = ROOT / "configs" / "pandaconfig.json"
MODEL_OUT = ROOT / "models" / "model_part1.joblib"

CONF_FILE = "pandaconfig.json"
CONTROLLED_FRAME = "panda_link8"

DEFAULT_TARGETS = [
    [0.5, 0.0, 0.1],
    [0.4, 0.2, 0.1],
    [0.4,-0.2, 0.1],
    [0.5, 0.0, 0.1],
]
DEFAULT_ORI = [0.0, 0.0, 0.0, 1.0]


KP_POS = 100
KP_ORI = 0
# -----------------------------------------------------


def debug_paths():
    print("[DEBUG] CWD:", Path().resolve())
    print("[DEBUG] Script ROOT:", ROOT)
    print("[DEBUG] DATA_DIR:", DATA_DIR)
    print("[DEBUG] CONFIG_PATH:", CONFIG_PATH)
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")


def list_pkls():
    files = sorted(
        [*glob.glob(str(DATA_DIR / "data_*.pkl"))] +
        [*glob.glob(str(DATA_DIR / "[0-9]*.pkl"))]
    )
    if not files:
        raise FileNotFoundError(f"No .pkl files under {DATA_DIR}")
    print(f"[Info] Found {len(files)} pkl files")
    return [Path(f) for f in files]


def infer_index_from_name(p: Path) -> int | None:
    name = p.stem  # data_0 或 0
    try:
        if name.startswith("data_"):
            return int(name.split("_")[1])
        return int(name)
    except Exception:
        return None


def build_dyn_and_sim():
    sim = pb.SimInterface(CONF_FILE, conf_file_path_ext=str(ROOT))
    time_step = sim.GetTimeStep()
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    dyn = PinWrapper(CONF_FILE, "pybullet", ext_names, ["pybullet"], False, 0, str(ROOT))
    joint_vel_limits = np.array(sim.GetBotJointsVelLimit(), dtype=float)
    return sim, dyn, time_step, joint_vel_limits


def reconstruct_qdes_for_rollout(dyn, time_step, joint_vel_limits, q_mes_all, desired_cart_pos, desired_cart_ori):
    q_des_list, qd_des_list = [], []
    pd_d = np.zeros(3); ori_d_des = np.zeros(3)
    for q_mes in q_mes_all:
        q_des, qd_des = CartesianDiffKin(
            dyn, CONTROLLED_FRAME, np.array(q_mes, dtype=float),
            np.array(desired_cart_pos, dtype=float), pd_d,
            np.array(desired_cart_ori, dtype=float), ori_d_des,
            time_step, "pos", KP_POS, KP_ORI, joint_vel_limits
        )
        q_des_list.append(q_des); qd_des_list.append(qd_des)
    return np.asarray(q_des_list), np.asarray(qd_des_list)


def per_joint_metrics(y, yp):
    return (mean_squared_error(y, yp, multioutput="raw_values"),
            mean_absolute_error(y, yp, multioutput="raw_values"),
            r2_score(y, yp, multioutput="raw_values"))


def main():
    debug_paths()
    files = list_pkls()
    _, dyn, time_step, joint_vel_limits = build_dyn_and_sim()

    X_list, Y_list = [], []

    for p in files:
        with p.open("rb") as f:
            d = pickle.load(f)

        q_mes_all = np.asarray(d["q_mes_all"], dtype=float)   # [T,7]
        tau_all   = np.asarray(d["tau_mes_all"], dtype=float) # [T,7]
        qd_mes_all = np.asarray(d.get("qd_mes_all", []), dtype=float) if "qd_mes_all" in d else None
        cart_pos_all = np.asarray(d["cart_pos_all"], dtype=float)

        idx = infer_index_from_name(p)
        if idx is not None and idx < len(DEFAULT_TARGETS):
            desired_cart_pos = DEFAULT_TARGETS[idx]
            desired_cart_ori = DEFAULT_ORI
        else:
            desired_cart_pos = cart_pos_all[-1].tolist()
            desired_cart_ori = DEFAULT_ORI

        q_des_all, qd_des_all = reconstruct_qdes_for_rollout(
            dyn, time_step, joint_vel_limits, q_mes_all, desired_cart_pos, desired_cart_ori
        )

        err = q_des_all - q_mes_all
        if qd_mes_all is not None and qd_mes_all.size > 0:
            derr = qd_des_all - qd_mes_all
            X = np.concatenate([err, derr], axis=1)  # [T,14]
        else:
            X = err  # [T,7]

        X_list.append(X)
        Y_list.append(tau_all)
        print(f"[Load] {p.name}: T={X.shape[0]}, feat_dim={X.shape[1]}")

    X = np.vstack(X_list)   # [N, D]
    Y = np.vstack(Y_list)   # [N, 7]
    print(f"[Data] X={X.shape}, Y={Y.shape}")

    # 划分数据：70/15/15
    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X, Y, test_size=0.30, random_state=42, shuffle=True)
    X_va, X_te, Y_va, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=42, shuffle=True)
    print(f"[Split] train={X_tr.shape[0]} | val={X_va.shape[0]} | test={X_te.shape[0]}")

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
        random_state=42,
        verbose=False
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline([("xsc", StandardScaler()), ("mlp", base)]),
        transformer=StandardScaler()
    )

    model.fit(X_tr, Y_tr)

    for name, Xs, Ys in [("Train", X_tr, Y_tr), ("Val", X_va, Y_va), ("Test", X_te, Y_te)]:
        Yp = model.predict(Xs)
        mse = mean_squared_error(Ys, Yp)
        mae = mean_absolute_error(Ys, Yp)
        r2  = r2_score(Ys, Yp)
        jmse, jmae, jr2 = per_joint_metrics(Ys, Yp)
        print(f"[{name}] MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
        print("        per-joint R2:", "  ".join([f"J{i+1}:{r:.3f}" for i,r in enumerate(jr2)]))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"[OK] saved model -> {MODEL_OUT.resolve()}")


if __name__ == "__main__":
    main()
