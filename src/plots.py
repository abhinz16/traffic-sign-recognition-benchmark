import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import project_path


def save_history(history):
    path = project_path("outputs", "training_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training(history):
    fig_dir = project_path("figures")
    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    plt.plot(epochs, [h["val_acc"] for h in history], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_curve.png", dpi=180)
    plt.close()


def plot_confusion(cm):
    plt.figure(figsize=(9, 8))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.imshow(cm_norm, aspect="auto")
    plt.colorbar(label="row-normalized count")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(project_path("figures", "confusion_matrix.png"), dpi=180)
    plt.close()


def save_classification_report(text):
    path = project_path("outputs", "classification_report.txt")
    path.write_text(text, encoding="utf-8")


def save_robustness(rows):
    path = project_path("outputs", "robustness_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "level", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for r in rows:
        grouped.setdefault(r["test"], []).append(r)

    plt.figure(figsize=(8, 5))
    for name, vals in grouped.items():
        vals = sorted(vals, key=lambda x: float(x["level"]))
        plt.plot([float(v["level"]) for v in vals], [v["accuracy"] for v in vals], marker="o", label=name)
    plt.xlabel("Perturbation level")
    plt.ylabel("Accuracy")
    plt.title("Robustness Check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(project_path("figures", "robustness_comparison.png"), dpi=180)
    plt.close()
