"""
train.py - Baseline CNN Model for Cat vs Dog Classification
============================================================
MLOps Assignment 2 - Model Building + Experiment Tracking (MLflow)
Uses PyTorch with CUDA acceleration.
Saves trained model in .pt format.
Logs parameters, metrics, and artifacts to MLflow.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, classification_report

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join("data", "PetImages")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cat_dog_cnn.pt")
PLOTS_DIR = "plots"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 128
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

EXPERIMENT_NAME = "CatDog-CNN-Baseline"

# ─── Device Setup ────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─── Fix Corrupted Images ───────────────────────────────────────────────────

def safe_loader(path):
    """Load image, raise error on corrupted files so they get skipped."""
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        return img.convert("RGB")
    except Exception:
        raise OSError(f"Corrupted image: {path}")


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips corrupted images."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, loader=safe_loader)
        valid_samples = []
        print("Validating images (this may take a moment)...")
        skipped = 0
        for path, label in self.samples:
            try:
                if os.path.getsize(path) > 0:
                    valid_samples.append((path, label))
                else:
                    skipped += 1
            except OSError:
                skipped += 1
        print(f"Skipped {skipped} invalid files. Using {len(valid_samples)} images.")
        self.samples = valid_samples
        self.imgs = valid_samples


# ─── Data Transforms ────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Baseline CNN Model ─────────────────────────────────────────────────────

class CatDogCNN(nn.Module):
    """
    Simple baseline CNN for binary classification (Cat vs Dog).
    Architecture: 3 conv blocks + 2 FC layers.
    """

    def __init__(self):
        super(CatDogCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─── Training Function ──────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ─── Evaluation Function ────────────────────────────────────────────────────

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ─── Plotting Functions ─────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Plot and save training/validation loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-o', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_accuracy_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved loss/accuracy curves to '{path}'")
    return path


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Plot and save confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved confusion matrix to '{path}'")
    return path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Cat vs Dog CNN - Training Pipeline with MLflow Tracking")
    print("=" * 60)

    # ── MLflow Setup ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n[MLflow] Experiment: '{EXPERIMENT_NAME}'")

    with mlflow.start_run(run_name=f"baseline-cnn-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"[MLflow] Run ID: {run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "model_type": "CatDogCNN",
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "train_split": TRAIN_SPLIT,
            "optimizer": "Adam",
            "scheduler": "StepLR(step=3, gamma=0.5)",
            "device": str(device),
            "random_seed": RANDOM_SEED,
        })

        # 1. Load dataset
        print(f"\n[1/6] Loading dataset from '{DATA_DIR}'...")
        full_dataset = SafeImageFolder(DATA_DIR, transform=train_transform)

        class_names = full_dataset.classes
        print(f"Classes: {class_names}")
        print(f"Total images: {len(full_dataset)}")

        mlflow.log_params({
            "dataset_path": DATA_DIR,
            "num_classes": len(class_names),
            "total_images": len(full_dataset),
        })

        # 2. Split into train / val
        print(f"\n[2/6] Splitting dataset (train: {TRAIN_SPLIT*100:.0f}%, val: {(1-TRAIN_SPLIT)*100:.0f}%)...")
        torch.manual_seed(RANDOM_SEED)
        train_size = int(TRAIN_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Override transform for validation set
        val_dataset.dataset = SafeImageFolder(DATA_DIR, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2, pin_memory=True)

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")

        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        })

        # 3. Initialize model
        print(f"\n[3/6] Initializing CNN model on {device}...")
        model = CatDogCNN().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        mlflow.log_param("total_parameters", total_params)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        # 4. Training loop
        print(f"\n[4/6] Training for {NUM_EPOCHS} epochs...")
        print("-" * 60)

        best_val_acc = 0.0
        start_time = time.time()

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )

            scheduler.step()
            epoch_time = time.time() - epoch_start

            # Store for plots
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Log metrics to MLflow (per epoch)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch_time_seconds": epoch_time,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            print(f"Epoch [{epoch}/{NUM_EPOCHS}] ({epoch_time:.1f}s) | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_preds = val_preds
                best_val_labels = val_labels
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'class_names': class_names,
                    'image_size': IMAGE_SIZE,
                }, MODEL_PATH)
                print(f"  -> Best model saved! (Val Acc: {val_acc:.2f}%)")

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        # Log final metrics
        mlflow.log_metrics({
            "best_val_accuracy": best_val_acc,
            "total_training_time_seconds": total_time,
        })

        # 5. Generate and log plots/artifacts
        print(f"\n[5/6] Generating plots and logging artifacts to MLflow...")

        # Loss & Accuracy curves
        loss_curve_path = plot_loss_curves(
            train_losses, val_losses, train_accs, val_accs, PLOTS_DIR
        )
        mlflow.log_artifact(loss_curve_path, "plots")

        # Confusion matrix
        cm_path = plot_confusion_matrix(
            best_val_labels, best_val_preds, class_names, PLOTS_DIR
        )
        mlflow.log_artifact(cm_path, "plots")

        # Classification report
        report = classification_report(
            best_val_labels, best_val_preds,
            target_names=class_names, output_dict=True
        )
        report_path = os.path.join(PLOTS_DIR, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, "reports")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            best_val_labels, best_val_preds, target_names=class_names
        ))

        # Log precision, recall, f1 per class
        for cls_name in class_names:
            mlflow.log_metrics({
                f"{cls_name}_precision": report[cls_name]["precision"],
                f"{cls_name}_recall": report[cls_name]["recall"],
                f"{cls_name}_f1_score": report[cls_name]["f1-score"],
            })

        # Log the trained model artifact
        mlflow.log_artifact(MODEL_PATH, "model")

        # Log model with MLflow's PyTorch integration
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        mlflow.pytorch.log_model(model, "pytorch_model", input_example=dummy_input)

        # 6. Summary
        print(f"\n[6/6] Model saved to '{MODEL_PATH}'")
        print(f"Model format: PyTorch (.pt)")
        print(f"File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        print(f"\n[MLflow] All metrics, params, and artifacts logged!")
        print(f"[MLflow] Run ID: {run_id}")
        print(f"[MLflow] View UI: run 'mlflow ui' in terminal")
        print("=" * 60)


if __name__ == "__main__":
    main()
