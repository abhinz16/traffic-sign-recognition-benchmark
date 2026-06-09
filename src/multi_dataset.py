"""Utilities for mixed-source traffic sign training.

The project trains in the GTSRB 43-class label space. Extra datasets can only be
combined with GTSRB after their labels are mapped into that same class system.
This file keeps that mapping logic in one place rather than hiding it inside the
training loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import datasets

from src.labels import GTSRB_CLASSES


@dataclass
class SourceSummary:
    name: str
    path: str
    total_images: int
    usable_images: int
    mapped_classes: int
    skipped_reason: str = ""


def _clean_name(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _class_aliases() -> Dict[str, int]:
    aliases: Dict[str, int] = {}
    for idx, name in enumerate(GTSRB_CLASSES):
        clean = _clean_name(name)
        aliases[clean] = idx

        # A few common dataset naming variants.
        aliases[clean.replace("speed limit ", "speedlimit ")] = idx
        aliases[clean.replace(" ", "")] = idx
        aliases[clean.replace("/", " ")] = idx

    manual = {
        "speed limit 20": 0,
        "speed limit 30": 1,
        "speed limit 50": 2,
        "speed limit 60": 3,
        "speed limit 70": 4,
        "speed limit 80": 5,
        "end speed limit 80": 6,
        "speed limit 100": 7,
        "speed limit 120": 8,
        "no passing": 9,
        "no passing trucks": 10,
        "right of way": 11,
        "priority road": 12,
        "yield": 13,
        "stop": 14,
        "no vehicles": 15,
        "trucks prohibited": 16,
        "no entry": 17,
        "general caution": 18,
        "dangerous curve left": 19,
        "dangerous curve right": 20,
        "double curve": 21,
        "bumpy road": 22,
        "slippery road": 23,
        "road narrows right": 24,
        "road work": 25,
        "traffic signals": 26,
        "pedestrians": 27,
        "children crossing": 28,
        "bicycles crossing": 29,
        "beware ice snow": 30,
        "wild animals crossing": 31,
        "end speed and passing limits": 32,
        "turn right ahead": 33,
        "turn left ahead": 34,
        "ahead only": 35,
        "go straight or right": 36,
        "go straight or left": 37,
        "keep right": 38,
        "keep left": 39,
        "roundabout mandatory": 40,
        "end no passing": 41,
        "end no passing trucks": 42,
    }
    aliases.update({_clean_name(k): v for k, v in manual.items()})
    return aliases


ALIASES = _class_aliases()


def map_class_name(name: str) -> Optional[int]:
    """Map an external dataset class folder to a GTSRB class id.

    Numeric folder names such as 00014 or 14 are treated as direct GTSRB class
    IDs. Text folder names are matched against normalized GTSRB class names.
    """

    raw = name.strip()
    if raw.isdigit():
        value = int(raw)
        return value if 0 <= value < len(GTSRB_CLASSES) else None

    # Folder names sometimes begin with a numeric class id: "14_stop".
    m = re.match(r"^0*(\d{1,2})(?:\D|$)", raw)
    if m:
        value = int(m.group(1))
        if 0 <= value < len(GTSRB_CLASSES):
            return value

    clean = _clean_name(raw)
    return ALIASES.get(clean)


class RemappedImageFolder(torch.utils.data.Dataset):
    """ImageFolder wrapper that remaps folder labels to GTSRB class IDs."""

    def __init__(self, root: Path, transform: Callable, min_images: int = 1):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_map: Dict[str, int] = {}

        base = datasets.ImageFolder(str(self.root))
        for folder_name, local_idx in base.class_to_idx.items():
            mapped = map_class_name(folder_name)
            if mapped is not None:
                self.class_map[folder_name] = mapped

        for path, local_idx in base.samples:
            folder_name = base.classes[local_idx]
            mapped = self.class_map.get(folder_name)
            if mapped is not None:
                self.samples.append((path, mapped))

        if len(self.samples) < min_images:
            raise ValueError(f"Only {len(self.samples)} mapped images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def find_external_training_sets(data_dir: Path, transform: Callable, min_images: int) -> Tuple[List[torch.utils.data.Dataset], List[SourceSummary]]:
    """Find BelgiumTS and Mapillary folders that can be used for training."""

    candidates = {
        "BelgiumTS": data_dir / "belgium",
        "Mapillary": data_dir / "mapillary",
    }
    datasets_out: List[torch.utils.data.Dataset] = []
    summaries: List[SourceSummary] = []

    for name, root in candidates.items():
        if not root.exists():
            summaries.append(SourceSummary(name, str(root), 0, 0, 0, "folder not found"))
            continue
        try:
            raw = datasets.ImageFolder(str(root))
            ds = RemappedImageFolder(root, transform=transform, min_images=min_images)
            datasets_out.append(ds)
            summaries.append(SourceSummary(
                name=name,
                path=str(root),
                total_images=len(raw.samples),
                usable_images=len(ds),
                mapped_classes=len(set(label for _, label in ds.samples)),
            ))
        except Exception as exc:
            total = 0
            try:
                total = len(datasets.ImageFolder(str(root)).samples)
            except Exception:
                pass
            summaries.append(SourceSummary(name, str(root), total, 0, 0, str(exc)))

    return datasets_out, summaries
