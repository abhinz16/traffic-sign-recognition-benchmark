from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils import project_path, worker_count
from src.multi_dataset import find_external_training_sets


def train_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669)),
    ])


def eval_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669)),
    ])


class TinyTrafficDemo(torch.utils.data.Dataset):
    """Small generated dataset used only for a quick pipeline check."""

    def __init__(self, n=1200, image_size=64, num_classes=43, seed=0, transform=None):
        self.n = n
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.gen = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (n,), generator=self.gen)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        from PIL import Image, ImageDraw
        import numpy as np

        rng = np.random.default_rng(int(idx) + 1337)
        label = int(self.labels[idx])
        bg = rng.integers(15, 90, size=(self.image_size, self.image_size, 3), dtype=np.uint8)
        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)
        margin = 10 + (label % 5)
        color = tuple(int(x) for x in rng.integers(140, 255, size=3))
        if label % 3 == 0:
            draw.ellipse([margin, margin, self.image_size - margin, self.image_size - margin], outline=color, width=5)
        elif label % 3 == 1:
            draw.rectangle([margin, margin, self.image_size - margin, self.image_size - margin], outline=color, width=5)
        else:
            draw.polygon([(self.image_size // 2, margin), (self.image_size - margin, self.image_size - margin), (margin, self.image_size - margin)], outline=color, width=5)
        draw.text((self.image_size // 2 - 5, self.image_size // 2 - 7), str(label % 10), fill=(255, 255, 255))
        if self.transform:
            img = self.transform(img)
        return img, label


def _split_train_val(dataset, valid_split, seed):
    n_val = int(len(dataset) * valid_split)
    n_train = len(dataset) - n_val
    return random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))


def _loader(dataset, batch_size, nw, pin, shuffle=False):
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=nw, pin_memory=pin)
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def make_loaders(cfg, device):
    data_dir = project_path(cfg.DATA_DIR)
    nw = worker_count(cfg.NUM_WORKERS, cfg.CPU_CORES_TO_LEAVE_FREE)
    pin = device.type == "cuda"
    mode = getattr(cfg, "DATASET_MODE", "gtsrb").lower()

    source_summary = []

    if mode == "demo":
        full_train = TinyTrafficDemo(cfg.DEMO_SAMPLES, cfg.IMAGE_SIZE, cfg.NUM_CLASSES, cfg.SEED,
                                     transform=train_transforms(cfg.IMAGE_SIZE))
        test_set = TinyTrafficDemo(max(300, cfg.DEMO_SAMPLES // 5), cfg.IMAGE_SIZE, cfg.NUM_CLASSES, cfg.SEED + 1,
                                   transform=eval_transforms(cfg.IMAGE_SIZE))
        train_set, val_set = _split_train_val(full_train, cfg.VALID_SPLIT, cfg.SEED)
        source_summary.append({"name": "demo", "train_images": len(train_set), "validation_images": len(val_set)})
    else:
        print("Loading GTSRB. If this is the first run, torchvision will download the full dataset.")
        gtsrb_train_aug = datasets.GTSRB(root=str(data_dir), split="train", download=True,
                                         transform=train_transforms(cfg.IMAGE_SIZE))
        gtsrb_train_eval = datasets.GTSRB(root=str(data_dir), split="train", download=False,
                                          transform=eval_transforms(cfg.IMAGE_SIZE))
        train_idx, val_idx = _split_train_val(range(len(gtsrb_train_aug)), cfg.VALID_SPLIT, cfg.SEED)
        gtsrb_train = torch.utils.data.Subset(gtsrb_train_aug, list(train_idx))
        val_set = torch.utils.data.Subset(gtsrb_train_eval, list(val_idx))
        test_set = datasets.GTSRB(root=str(data_dir), split="test", download=True,
                                  transform=eval_transforms(cfg.IMAGE_SIZE))

        train_parts = [gtsrb_train]
        source_summary.append({"name": "GTSRB", "train_images": len(gtsrb_train), "validation_images": len(val_set)})

        if mode == "combined" and getattr(cfg, "USE_EXTERNAL_DATASETS_FOR_TRAINING", True):
            extras, summaries = find_external_training_sets(
                data_dir=data_dir,
                transform=train_transforms(cfg.IMAGE_SIZE),
                min_images=getattr(cfg, "MIN_EXTERNAL_IMAGES_PER_DATASET", 1),
            )
            train_parts.extend(extras)
            for row in summaries:
                source_summary.append({
                    "name": row.name,
                    "train_images": row.usable_images,
                    "total_images_found": row.total_images,
                    "mapped_classes": row.mapped_classes,
                    "note": row.skipped_reason,
                })

        train_set = torch.utils.data.ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]

    loaders = {
        "train": _loader(train_set, cfg.BATCH_SIZE, nw, pin, shuffle=True),
        "val": _loader(val_set, cfg.BATCH_SIZE, nw, pin, shuffle=False),
        "test": _loader(test_set, cfg.BATCH_SIZE, nw, pin, shuffle=False),
    }
    loaders["source_summary"] = source_summary
    return loaders


def find_imagefolder_datasets(cfg):
    """Return external datasets that are already in an ImageFolder-style layout."""
    transform = eval_transforms(cfg.IMAGE_SIZE)
    candidates = {
        "belgium": project_path(cfg.DATA_DIR, "belgium"),
        "mapillary": project_path(cfg.DATA_DIR, "mapillary"),
    }
    out = {}
    for name, root in candidates.items():
        if not root.exists():
            continue
        try:
            ds = datasets.ImageFolder(str(root), transform=transform)
            if len(ds.classes) > 1 and len(ds) > 0:
                out[name] = ds
        except Exception:
            pass
    return out
