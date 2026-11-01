"""
AlexNet Architecture from Scratch on Resized CIFAR-10 (224x224)
Part of the '100 PyTorch Projects in 10 Days' Challenge (Day 3 - Project 22)
Author: Gideon Mwiti

Demonstrates:
- Canonical 8-layer AlexNet architecture (Krizhevsky, Sutskever, Hinton, 2012)
- Re-implementation of Local Response Normalization (LRN) & Overlapping Max Pooling
- Input resizing of CIFAR-10 to ImageNet standard dimensions (224x224)
- 11x11 Conv1 filter bank visualization and feature map activation extraction
"""

import sys
import types
import io
import os
import random
import time
from typing import Tuple, List, Dict, Any

# Windows Python 3.11+ bz2/lzma compatibility patch
sys.modules['bz2'] = types.SimpleNamespace(open=io.open, BZ2File=io.BytesIO)
sys.modules['lzma'] = types.SimpleNamespace(
    open=io.open,
    LZMAFile=io.BytesIO,
    FORMAT_ALONE=1,
    FORMAT_XZ=2,
    CHECK_NONE=0,
    CHECK_CRC32=1,
    CHECK_CRC64=4,
    CHECK_SHA256=10
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------------
# 1. Reproducibility & Device Configuration
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Execution Device: {DEVICE}", flush=True)

CIFAR10_CLASSES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# 2. Canonical AlexNet Architecture (2012)
# -----------------------------------------------------------------------------
class AlexNetFromScratch(nn.Module):
    """
    Canonical 8-layer AlexNet architecture (5 Conv + 3 Dense) from scratch.
    Features:
    - Layer 1: Conv2d(3, 64, 11x11, s=4, p=2) -> ReLU -> LRN -> MaxPool2d(3x3, s=2)
    - Layer 2: Conv2d(64, 192, 5x5, p=2) -> ReLU -> LRN -> MaxPool2d(3x3, s=2)
    - Layer 3: Conv2d(192, 384, 3x3, p=1) -> ReLU
    - Layer 4: Conv2d(384, 256, 3x3, p=1) -> ReLU
    - Layer 5: Conv2d(256, 256, 3x3, p=1) -> ReLU -> MaxPool2d(3x3, s=2)
    - Classifier: AdaptiveAvgPool2d((6, 6)) -> Dropout(0.5) -> Linear(9216, 4096)
                  -> ReLU -> Dropout(0.5) -> Linear(4096, 4096) -> ReLU -> Linear(4096, 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_conv1_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts C1 feature maps for visual inspection."""
        return self.features[0](x)


# -----------------------------------------------------------------------------
# 3. Model Training & Evaluation Routines
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, List[torch.Tensor]]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true_list, y_pred_list, y_prob_list = [], [], []
    raw_images = []

    for images, targets in loader:
        images_dev = images.to(DEVICE)
        outputs = model(images_dev)
        loss = criterion(outputs, targets.to(DEVICE))

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        correct += (preds == targets.numpy()).sum()
        total += targets.size(0)

        y_true_list.extend(targets.numpy())
        y_pred_list.extend(preds)
        y_prob_list.extend(probs)
        raw_images.extend([img for img in images])

    return (
        total_loss / total,
        (correct / total) * 100.0,
        np.array(y_true_list),
        np.array(y_pred_list),
        np.array(y_prob_list),
        raw_images
    )


# -----------------------------------------------------------------------------
# 4. Publication-Grade Visualization Dashboard
# -----------------------------------------------------------------------------
def plot_alexnet_dashboard(
    history: Dict[str, List[float]],
    conv1_weights: torch.Tensor,
    sample_img: torch.Tensor,
    conv1_acts: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    raw_images: List[torch.Tensor],
    output_path: str = "alexnet_cifar10_results.png"
) -> None:
    print("[*] Generating publication-grade AlexNet diagnostic dashboard...", flush=True)
    fig = plt.figure(figsize=(20, 13), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], wspace=0.25, hspace=0.35)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2980b9", linewidth=2.5, marker="o")
    ax1.plot(epochs, history["test_loss"], label="Val Loss", color="#e74c3c", linewidth=2.5, marker="s", linestyle="--")
    ax1.set_title("Cross-Entropy Loss Dynamics (224x224)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9.5, loc="upper right")

    # 2. Accuracy Trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#27ae60", linewidth=2.5, marker="o")
    ax2.plot(epochs, history["test_acc"], label="Val Accuracy", color="#8e44ad", linewidth=2.5, marker="^", linestyle="--")
    ax2.set_title(f"Accuracy Trajectory (Peak Val: {max(history['test_acc']):.2f}%)",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9.5, loc="lower right")

    # 3. Conv1 11x11 Learned Filter Bank (8x8 grid of RGB weights)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    sub_gs_filt = gridspec.GridSpecFromSubplotSpec(8, 8, subplot_spec=gs[0, 2], wspace=0.08, hspace=0.08)
    # Normalize weights to [0, 1] for visualization
    w_min, w_max = conv1_weights.min(), conv1_weights.max()
    norm_weights = (conv1_weights - w_min) / (w_max - w_min + 1e-8)
    norm_weights = norm_weights.permute(0, 2, 3, 1).cpu().numpy()

    for i in range(64):
        sub_ax = fig.add_subplot(sub_gs_filt[i // 8, i % 8])
        sub_ax.imshow(norm_weights[i])
        sub_ax.axis("off")

    ax3_title = fig.add_subplot(gs[0, 2])
    ax3_title.axis("off")
    ax3_title.set_title("First Layer Learned Convolutional Filters (64 Filters of 11x11x3)",
                        fontsize=12, fontweight="bold", pad=12)

    # 4. Conv1 Feature Map Activations
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    sub_gs_act = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[1, 0], wspace=0.12, hspace=0.2)
    # Display sample input image
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std = [1.0 / s for s in IMAGENET_STD]
    denorm = T.Compose([
        T.Normalize(mean=[0.0, 0.0, 0.0], std=inv_std),
        T.Normalize(mean=inv_mean, std=[1.0, 1.0, 1.0]),
    ])
    disp_sample = torch.clamp(denorm(sample_img.squeeze(0)), 0.0, 1.0).permute(1, 2, 0).cpu().numpy()

    sub_ax_in = fig.add_subplot(sub_gs_act[0, 0])
    sub_ax_in.imshow(disp_sample)
    sub_ax_in.set_title("Input (224x224)", fontsize=7.5, fontweight="bold")
    sub_ax_in.axis("off")

    acts = conv1_acts[0].detach().cpu().numpy() # [64, 55, 55]
    for k in range(11):
        r = (k + 1) // 4
        c = (k + 1) % 4
        sub_ax = fig.add_subplot(sub_gs_act[r, c])
        sub_ax.imshow(acts[k], cmap="viridis")
        sub_ax.set_title(f"Act #{k+1}", fontsize=7)
        sub_ax.axis("off")

    ax4_title = fig.add_subplot(gs[1, 0])
    ax4_title.axis("off")
    ax4_title.set_title("Conv1 Feature Map Activations (55x55 Feature Resolution)",
                        fontsize=12, fontweight="bold", pad=14)

    # 5. Model Architecture & Complexity Summary
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    table_data = [
        ["Layer", "Type", "Kernel / Stride", "Output Shape", "Parameters"],
        ["Conv 1", "Conv2d + LRN", "11x11, s=4, p=2", "64 × 55 × 55", "23,296"],
        ["Pool 1", "MaxPool2d", "3x3, s=2 (Overlap)", "64 × 27 × 27", "0"],
        ["Conv 2", "Conv2d + LRN", "5x5, s=1, p=2", "192 × 27 × 27", "307,392"],
        ["Pool 2", "MaxPool2d", "3x3, s=2 (Overlap)", "192 × 13 × 13", "0"],
        ["Conv 3", "Conv2d", "3x3, s=1, p=1", "384 × 13 × 13", "663,936"],
        ["Conv 4", "Conv2d", "3x3, s=1, p=1", "256 × 13 × 13", "884,992"],
        ["Conv 5", "Conv2d", "3x3, s=1, p=1", "256 × 13 × 13", "590,080"],
        ["Pool 5", "MaxPool2d", "3x3, s=2 (Overlap)", "256 × 6 × 6", "0"],
        ["FC 6", "Dense + Dropout", "9216 -> 4096", "4096", "37,752,832"],
        ["FC 7", "Dense + Dropout", "4096 -> 4096", "4096", "16,781,312"],
        ["FC 8", "Dense Output", "4096 -> 10", "10", "40,970"],
        ["Total", "AlexNet-Full", "5 Conv + 3 Dense", "10 Classes", "57,044,810"]
    ]
    table = ax5.table(cellText=table_data, loc="center", cellLoc="center",
                      colWidths=[0.18, 0.26, 0.26, 0.20, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.45)
    for col in range(5):
        cell = table[(0, col)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
    for col in range(5):
        cell = table[(12, col)]
        cell.set_facecolor("#e8f8f5")
        cell.set_text_props(weight="bold")
    ax5.set_title("AlexNet Architectural Parameter Derivation", fontsize=12, fontweight="bold", pad=12)

    # 6. Qualitative Prediction Gallery (Sample 224x224 Test Images)
    sub_gs_pred = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, 2], wspace=0.25, hspace=0.35)
    for idx in range(8):
        sub_ax = fig.add_subplot(sub_gs_pred[idx // 4, idx % 4])
        img_tensor = raw_images[idx]
        disp_img = torch.clamp(denorm(img_tensor), 0.0, 1.0).permute(1, 2, 0).numpy()

        sub_ax.imshow(disp_img)
        t_cls = CIFAR10_CLASSES[y_true[idx]]
        p_cls = CIFAR10_CLASSES[y_pred[idx]]
        conf = y_prob[idx, y_pred[idx]] * 100.0

        is_correct = (y_true[idx] == y_pred[idx])
        color = "#27ae60" if is_correct else "#c0392b"
        sub_ax.set_title(f"T:{t_cls[:4]}\nP:{p_cls[:4]} ({conf:.0f}%)",
                         fontsize=7.5, fontweight="bold", color=color)
        sub_ax.axis("off")
        for spine in sub_ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)

    ax6_title = fig.add_subplot(gs[1, 2])
    ax6_title.axis("off")
    ax6_title.set_title("Qualitative Resized Test Predictions (224x224)",
                        fontsize=12, fontweight="bold", pad=14)

    plt.suptitle("AlexNet (Krizhevsky et al., 2012) Implemented from Scratch on CIFAR-10 (224x224)",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Diagnostic dashboard saved successfully to {output_path}", flush=True)


# -----------------------------------------------------------------------------
# 5. Main Routine
# -----------------------------------------------------------------------------
def main() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    print("================================================================================")
    print("      AlexNet Architecture from Scratch on Resized CIFAR-10 (224x224)           ")
    print("================================================================================")

    # 1. Dataset Preparation: Resizing CIFAR-10 (32x32) to ImageNet resolution (224x224)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=transform)

    # Use a structured representative subset for CPU benchmarking
    train_subset = Subset(train_dataset, list(range(1200)))
    test_subset = Subset(test_dataset, list(range(300)))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)
    print(f"[*] Train Subset: {len(train_subset)} samples | Test Subset: {len(test_subset)} samples (Resized to 224x224).", flush=True)

    # 2. Model Instantiation & Parameter Verification
    set_seed(42)
    model = AlexNetFromScratch(num_classes=10).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] AlexNet Instantiated: Total Trainable Parameters = {total_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    num_epochs = 3

    # 3. Training Loop
    print(f"\n[*] Training AlexNet for {num_epochs} Epochs on 224x224 images...", flush=True)
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _, _, _ = evaluate(model, test_loader, criterion)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["test_loss"].append(v_loss)
        history["test_acc"].append(v_acc)

        print(f"    Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:6.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:6.2f}%", flush=True)

    elapsed = time.time() - start_time
    print(f"[+] Training completed in {elapsed:.2f} seconds.", flush=True)

    # 4. Final Evaluation
    test_loss, test_acc, y_true, y_pred, y_prob, raw_images = evaluate(model, test_loader, criterion)

    # 5. Extract Feature Maps & Conv1 Weights
    conv1_weights = model.features[0].weight.data
    sample_img = raw_images[0].unsqueeze(0).to(DEVICE)
    conv1_acts = model.extract_conv1_activations(sample_img)

    # 6. Generate Visualization Dashboard
    plot_alexnet_dashboard(
        history=history,
        conv1_weights=conv1_weights,
        sample_img=sample_img,
        conv1_acts=conv1_acts,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        raw_images=raw_images,
        output_path="alexnet_cifar10_results.png"
    )

    # 7. Summary
    print("\n" + "=" * 80)
    print("                    PER-CLASS CLASSIFICATION REPORT                             ")
    print("=" * 80)
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, output_dict=True)
    for cls in CIFAR10_CLASSES:
        p = report[cls]["precision"] * 100.0
        r = report[cls]["recall"] * 100.0
        f1 = report[cls]["f1-score"] * 100.0
        sup = report[cls]["support"]
        print(f"{cls:<12} | Precision: {p:6.2f}% | Recall: {r:6.2f}% | F1: {f1:6.2f}% | Support: {sup}")
    print("=" * 80)
    print(f"Final Validation Accuracy: {test_acc:6.2f}%")
    print("=" * 80)
    print("[✓] Execution complete. Dashboard saved to 'alexnet_cifar10_results.png'.\n", flush=True)


if __name__ == "__main__":
    main()

