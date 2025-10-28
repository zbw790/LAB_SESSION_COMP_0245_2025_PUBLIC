import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 固定保存到脚本所在目录 /week_4，并确保 figures/ 存在
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

# 1. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 2. Model Construction
class ShallowMLP(nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


class DeepMLP(nn.Module):
    # 2 层隐藏层：784 -> 30 -> 20 -> 10（与浅网参数量接近）
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 30)
        self.hidden2 = nn.Linear(30, 20)
        self.relu = nn.ReLU()
        self.output = nn.Linear(20, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


class Deep3MLP(nn.Module):
    # 3 层隐藏层：784 -> 30 -> 20 -> 12 -> 10
    # 近似 24,552 个可训练参数（略低于 shallow 的 ~25,450）
    def __init__(self):
        super(Deep3MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 15)
        self.out = nn.Linear(15, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return torch.log_softmax(x, dim=1)


# 3. Training / Evaluation setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

criterion = nn.NLLLoss()
epochs = 20
lr = 0.01

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 通用训练函数
def train_model(model, title):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_losses, train_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * correct / len(train_loader.dataset)
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        print(f"[{title}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

    return train_losses, train_accs

@torch.no_grad()
def test_model(model, title):
    model.eval()
    total_loss, correct = 0.0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"[{title}] Test Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    return avg_loss, accuracy


# 4. Run all three models
shallow_model = ShallowMLP().to(device)
deep_model = DeepMLP().to(device)
deep3_model = Deep3MLP().to(device)

print(f"\n[Params] Shallow: {count_params(shallow_model):,} | Deep(2L): {count_params(deep_model):,} | Deep(3L): {count_params(deep3_model):,}\n")

print("Training Shallow MLP...")
shallow_loss, shallow_acc = train_model(shallow_model, "Shallow")
shallow_test_loss, shallow_test_acc = test_model(shallow_model, "Shallow")

print("\nTraining Deep (2-layer) MLP...")
deep_loss, deep_acc = train_model(deep_model, "Deep-2L")
deep_test_loss, deep_test_acc = test_model(deep_model, "Deep-2L")

print("\nTraining Deep (3-layer) MLP...")
deep3_loss, deep3_acc = train_model(deep3_model, "Deep-3L")
deep3_test_loss, deep3_test_acc = test_model(deep3_model, "Deep-3L")

# 5. Visualization
epochs_range = np.arange(1, epochs+1)

# Accuracy
plt.figure(figsize=(7,4))
plt.plot(epochs_range, shallow_acc, label=f"Shallow (Test {shallow_test_acc:.2f}%)")
plt.plot(epochs_range, deep_acc,    label=f"Deep-2L (Test {deep_test_acc:.2f}%)")
plt.plot(epochs_range, deep3_acc,   label=f"Deep-3L (Test {deep3_test_acc:.2f}%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Task 1 — Accuracy vs Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/task1_acc_curves.png", dpi=200)

# Loss
plt.figure(figsize=(7,4))
plt.plot(epochs_range, shallow_loss, label="Shallow")
plt.plot(epochs_range, deep_loss,    label="Deep-2L")
plt.plot(epochs_range, deep3_loss,   label="Deep-3L")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Task 1 — Training Loss vs Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/task1_loss_curves.png", dpi=200)

print("\n✅ Figures saved to ./figures/: task1_acc_curves.png, task1_loss_curves.png")


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_layer(ax, x, y, label):
    box = FancyBboxPatch((x-0.4, y-0.2), 0.8, 0.4,
                         boxstyle="round,pad=0.3",
                         facecolor="#DCE6F2", edgecolor="black")
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=10)

def draw_network(ax, title, layers):
    ax.set_xlim(0, len(layers))
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")

    for i, (name, num) in enumerate(layers):
        draw_layer(ax, i, 0, f"{name}\n({num})")
        if i < len(layers) - 1:
            ax.arrow(i+0.4, 0, 0.2, 0, head_width=0.05, head_length=0.1, fc='gray', ec='gray')
    ax.set_title(title, fontsize=12, pad=10)

# --- Create 3 subplots horizontally ---
fig, axs = plt.subplots(1, 3, figsize=(12, 2.5))

draw_network(axs[0], "Shallow MLP", [("Input", 784), ("Hidden", 32), ("Output", 10)])
draw_network(axs[1], "Deep-2L MLP", [("Input", 784), ("Hidden1", 30), ("Hidden2", 20), ("Output", 10)])
draw_network(axs[2], "Deep-3L MLP", [("Input", 784), ("Hidden1", 30), ("Hidden2", 15), ("Hidden3", 10), ("Output", 10)])

plt.tight_layout()
plt.savefig("figures/task1_structure_comparison.png", dpi=200)
print("✅ Saved combined structure diagram: figures/task1_structure_comparison.png")
