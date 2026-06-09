import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.utils import project_path
from src.labels import GTSRB_CLASSES


def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="testing", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "loss": total_loss / len(y_true),
        "acc": float((y_true == y_pred).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(43))),
        "report": classification_report(y_true, y_pred, labels=list(range(43)), target_names=GTSRB_CLASSES, zero_division=0),
    }


def denormalize(x):
    mean = torch.tensor([0.3403, 0.3121, 0.3214], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.2724, 0.2608, 0.2669], device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def save_failure_cases(model, loader, device, max_cases=36):
    out = project_path("outputs", "failure_cases")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    model.eval()
    saved = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="saving failures", leave=False):
            x_dev = x.to(device, non_blocking=True)
            logits = model(x_dev)
            pred = logits.argmax(1).cpu()
            for i in range(x.size(0)):
                if int(pred[i]) == int(y[i]):
                    continue
                img = denormalize(x[i]).permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                fname = f"case_{saved:03d}_true_{int(y[i]):02d}_pred_{int(pred[i]):02d}.png"
                img.save(out / fname)
                saved += 1
                if saved >= max_cases:
                    return saved
    return saved
