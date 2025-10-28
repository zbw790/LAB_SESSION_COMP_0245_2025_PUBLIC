# final/evaluate_model.py
import os, pickle, numpy as np, joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ====== 路径 ======
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model_part1.joblib"
DATA_DIR = ROOT
CONF_FILE = "pandaconfig.json"          # 与训练一致
CONTROLLED_FRAME = "panda_link8"
KP_POS, KP_ORI = 100, 0                 # 与训练一致

assert MODEL_PATH.is_file(), f"❌ Model not found: {MODEL_PATH}"

# ====== 来自你项目的动力学/差分运动学 ======
from simulation_and_control import pb, PinWrapper, CartesianDiffKin

def build_dyn_and_sim():
    sim = pb.SimInterface(CONF_FILE, conf_file_path_ext=str(ROOT))
    time_step = sim.GetTimeStep()
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    dyn = PinWrapper(CONF_FILE, "pybullet", ext_names, ["pybullet"], False, 0, str(ROOT))
    joint_vel_limits = np.array(sim.GetBotJointsVelLimit(), dtype=float)
    return sim, dyn, time_step, joint_vel_limits

# 与 train_part1 保持一致：按文件 index 推断目标位姿（或用轨迹末端位置兜底）
DEFAULT_TARGETS = [
    [0.5, 0.0, 0.1],
    [0.4, 0.2, 0.1],
    [0.4,-0.2, 0.1],
    [0.5, 0.0, 0.1],
]
DEFAULT_ORI = [0.0, 0.0, 0.0, 1.0]

def infer_index_from_name(p: Path):
    s = p.stem  # data_0 或 0
    try:
        return int(s.split("_")[1]) if s.startswith("data_") else int(s)
    except Exception:
        return None

def reconstruct_feats_for_rollout(dyn, time_step, joint_vel_limits, d: dict, p: Path):
    q_mes_all = np.asarray(d["q_mes_all"], dtype=float)        # [T,7]
    qd_mes_all = np.asarray(d.get("qd_mes_all", []), dtype=float) if "qd_mes_all" in d else None
    cart_pos_all = np.asarray(d["cart_pos_all"], dtype=float)

    idx = infer_index_from_name(p)
    if idx is not None and idx < len(DEFAULT_TARGETS):
        desired_cart_pos = np.array(DEFAULT_TARGETS[idx], dtype=float)
    else:
        desired_cart_pos = cart_pos_all[-1]  # 兜底
    desired_cart_ori = np.array(DEFAULT_ORI, dtype=float)

    # 重建 q_des, qd_des 与训练一致
    q_des_list, qd_des_list = [], []
    pd_d = np.zeros(3); ori_d_des = np.zeros(3)
    for q_mes in q_mes_all:
        q_des, qd_des = CartesianDiffKin(
            dyn, CONTROLLED_FRAME, np.array(q_mes, dtype=float),
            desired_cart_pos, pd_d, desired_cart_ori, ori_d_des,
            time_step, "pos", KP_POS, KP_ORI, joint_vel_limits
        )
        q_des_list.append(q_des); qd_des_list.append(qd_des)
    q_des_all = np.asarray(q_des_list)         # [T,7]
    qd_des_all = np.asarray(qd_des_list)       # [T,7]

    err  = q_des_all - q_mes_all               # [T,7]
    if qd_mes_all is not None and qd_mes_all.size > 0:
        derr = qd_des_all - qd_mes_all         # [T,7]
        X = np.concatenate([err, derr], axis=1)  # [T,14] —— 训练所用特征
    else:
        # 极少数情况下缺少速度测量；为了与训练一致仍需14维，可用零向量代替
        derr = np.zeros_like(err)
        X = np.concatenate([err, derr], axis=1)
    Y = np.asarray(d["tau_mes_all"], dtype=float)  # 监督信号 [T,7]
    return X, Y

# ====== 主流程 ======
print(f"[INFO] Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

sim, dyn, time_step, joint_vel_limits = build_dyn_and_sim()

data_files = sorted(list(DATA_DIR.glob("data_*.pkl")) + list(DATA_DIR.glob("[0-9]*.pkl")))
assert data_files, f"❌ No rollout .pkl found under {DATA_DIR}"

all_r2, all_mae, all_mse = [], [], []

for p in data_files:
    with p.open("rb") as f:
        d = pickle.load(f)
    X, Y_true = reconstruct_feats_for_rollout(dyn, time_step, joint_vel_limits, d, p)
    Y_pred = model.predict(X)

    mse = mean_squared_error(Y_true, Y_pred)
    mae = mean_absolute_error(Y_true, Y_pred)
    r2  = r2_score(Y_true, Y_pred)
    per_joint_r2 = r2_score(Y_true, Y_pred, multioutput="raw_values")

    all_mse.append(mse); all_mae.append(mae); all_r2.append(r2)
    print(f"[{p.name}] MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
    print("    per-joint R2:", "  ".join([f"J{i+1}:{v:.3f}" for i,v in enumerate(per_joint_r2)]))

print("\n=== Summary ===")
print(f"Avg MSE={np.mean(all_mse):.6f}  Avg MAE={np.mean(all_mae):.6f}  Avg R2={np.mean(all_r2):.4f}")

# —— 可选：画一条关节1的真实 vs 预测 —— #
try:
    sample = data_files[0]
    with sample.open("rb") as f:
        d0 = pickle.load(f)
    X0, Yt = reconstruct_feats_for_rollout(dyn, time_step, joint_vel_limits, d0, sample)
    Yp = model.predict(X0)
    Tshow = min(400, len(Yt))

    plt.figure(figsize=(8,4.5))
    plt.plot(Yt[:Tshow,0], label="True τ_joint1")
    plt.plot(Yp[:Tshow,0], "--", label="Pred τ_joint1")
    plt.title(f"True vs Predicted Torque — {sample.name} (Joint 1)")
    plt.xlabel("Timestep"); plt.ylabel("Torque (Nm)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[WARN] plot skipped: {e}")
