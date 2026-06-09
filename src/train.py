import torch
import torch.nn as nn
from tqdm import tqdm


def _accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def run_one_epoch(model, loader, optimizer, device, scaler=None, label_smoothing=0.0, train=True):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing if train else 0.0)
    model.train(train)
    total_loss, total_correct, total_seen = 0.0, 0, 0
    desc = "training" if train else "validation"

    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        n = y.size(0)
        total_loss += loss.item() * n
        total_correct += (logits.argmax(1) == y).sum().item()
        total_seen += n

    return {"loss": total_loss / total_seen, "acc": total_correct / total_seen}


def train_model(model, loaders, cfg, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.EPOCHS))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.USE_AMP))
    if not scaler.is_enabled():
        scaler = None

    history = []
    best_val = -1.0
    best_state = None

    for epoch in tqdm(range(1, cfg.EPOCHS + 1), desc="epochs"):
        train_stats = run_one_epoch(model, loaders["train"], optimizer, device, scaler, cfg.LABEL_SMOOTHING, train=True)
        val_stats = run_one_epoch(model, loaders["val"], optimizer, device, None, 0.0, train=False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"], "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"], "val_acc": val_stats["acc"],
        }
        history.append(row)
        print(f"epoch {epoch:02d}: train acc {train_stats['acc']:.3f}, val acc {val_stats['acc']:.3f}")
        if val_stats["acc"] > best_val:
            best_val = val_stats["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
