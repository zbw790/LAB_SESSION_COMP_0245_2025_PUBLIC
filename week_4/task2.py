import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt, numpy as np, pandas as pd, os

# === 环境设置 ===
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 数据集准备 ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_data, val_data = random_split(dataset, [train_len, val_len])

# === Deep-2L 模型 ===
class Deep2L(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 30)
        self.fc2 = nn.Linear(30, 20)
        self.out = nn.Linear(20, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return torch.log_softmax(x, dim=1)

criterion = nn.NLLLoss()
epochs = 20

def run_experiment(lr, batch):
    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch)
    model = Deep2L().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_loss, val_loss = [], []

    for ep in range(epochs):
        model.train()
        total = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        train_loss.append(total / len(train_loader.dataset))

        # 验证
        model.eval()
        total = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                total += criterion(model(x), y).item() * x.size(0)
        val_loss.append(total / len(val_loader.dataset))
        print(f"[lr={lr}, bs={batch}] Epoch {ep+1}/{epochs} | Train {train_loss[-1]:.3f} | Val {val_loss[-1]:.3f}")

    return model, train_loss, val_loss

# === 超参数网格 ===
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
records = []

# === 运行实验 ===
for lr in learning_rates:
    for bs in batch_sizes:
        model, tr_loss, val_loss = run_experiment(lr, bs)
        plt.plot(val_loss, label=f"lr={lr}, bs={bs}")
        records.append((lr, bs, val_loss[-1]))

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Task 2 — Validation Loss under different hyperparameters")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("figures/task2_val_loss_grid.png", dpi=200)

# === 选择最佳模型 (最小验证损失) ===
best_lr, best_bs, _ = sorted(records, key=lambda x: x[2])[0]
print(f"\n✅ Best hyperparameters: lr={best_lr}, batch_size={best_bs}")

# === 在测试集上评估 ===
best_model, _, _ = run_experiment(best_lr, best_bs)
test_loader = DataLoader(test_dataset, batch_size=best_bs)
best_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        pred = best_model(x).argmax(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
acc = np.mean(np.array(all_preds)==np.array(all_labels))
print(f"Test Accuracy: {acc*100:.2f}%")

# === 混淆矩阵 ===
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(cmap="Blues")
plt.title(f"Task 2 — Confusion Matrix (lr={best_lr}, bs={best_bs})")
plt.savefig("figures/task2_confusion_matrix.png", dpi=200)

# === 错误样例可视化 ===
wrong_idx = np.where(np.array(all_preds)!=np.array(all_labels))[0][:16]
fig, axes = plt.subplots(4,4,figsize=(5,5))
for ax,i in zip(axes.flat, wrong_idx):
    img = test_dataset[i][0][0].numpy()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"T:{all_labels[i]},P:{all_preds[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("figures/task2_wrong_examples.png", dpi=200)

# === 保存表格 ===
df = pd.DataFrame(records, columns=["LearningRate","BatchSize","ValLoss"])
df.to_csv("figures/task2_results.csv", index=False)
print("✅ Saved all figures and results to ./figures/")
