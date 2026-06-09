import copy

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import eval_transforms, worker_count


class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base.dataset[self.base.indices[idx]] if hasattr(self.base, "indices") else self.base[idx]
        if not hasattr(img, "mode"):
            return img, y
        return self.transform(img), y


def add_noise(x, sigma):
    if sigma <= 0:
        return x
    return (x + sigma * torch.randn_like(x)).clamp(0, 1)


class NoiseTensor:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x):
        return add_noise(x, self.sigma)


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="robustness", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def run_robustness(model, test_set, cfg, device):
    rows = []
    nw = worker_count(cfg.NUM_WORKERS)
    pin = device.type == "cuda"

    def make_loader(transform):
        ds = copy.copy(test_set)
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "transform"):
            ds.dataset.transform = transform
        elif hasattr(ds, "transform"):
            ds.transform = transform
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin)

    for sigma in tqdm(cfg.NOISE_LEVELS, desc="noise tests"):
        tfm = transforms.Compose([
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            NoiseTensor(sigma),
            transforms.Normalize(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669)),
        ])
        rows.append({"test": "gaussian_noise", "level": sigma, "accuracy": evaluate_accuracy(model, make_loader(tfm), device)})

    for blur in tqdm(cfg.BLUR_LEVELS, desc="blur tests"):
        ops = [transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))]
        if blur > 0:
            ops.append(transforms.GaussianBlur(kernel_size=blur))
        ops += [transforms.ToTensor(), transforms.Normalize(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669))]
        rows.append({"test": "gaussian_blur", "level": blur, "accuracy": evaluate_accuracy(model, make_loader(transforms.Compose(ops)), device)})

    for b in tqdm(cfg.BRIGHTNESS_LEVELS, desc="brightness tests"):
        tfm = transforms.Compose([
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ColorJitter(brightness=(b, b)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669)),
        ])
        rows.append({"test": "brightness", "level": b, "accuracy": evaluate_accuracy(model, make_loader(tfm), device)})
    return rows
