"""Download and register the datasets used by the benchmark.

GTSRB is downloaded through torchvision. BelgiumTS and Mapillary are attempted through
public Kaggle mirrors because the original project pages do not provide a single unauthenticated
Python download endpoint for every file. Mapillary's official dataset page requires a user login,
so this downloader uses the Kaggle mirror when credentials are available.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tqdm import tqdm
from torchvision import datasets

from src.utils import project_path


def _write_status(status):
    path = project_path("outputs", "dataset_status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in src.rglob("*") if p.is_file()]
    for p in tqdm(files, desc=f"Copying {dst.name}", unit="file"):
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(p, target)


def _download_kaggle_dataset(slug: str, target_dir: Path):
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed. Run: pip install kagglehub") from exc

    cached = Path(kagglehub.dataset_download(slug))
    _copy_tree(cached, target_dir)
    return target_dir


def download_all_datasets(cfg):
    data_dir = project_path(cfg.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    status = {}

    print("Checking GTSRB...")
    datasets.GTSRB(root=str(data_dir), split="train", download=True)
    datasets.GTSRB(root=str(data_dir), split="test", download=True)
    status["gtsrb"] = {
        "status": "available",
        "path": str(data_dir / "gtsrb"),
        "method": "torchvision.datasets.GTSRB",
    }

    jobs = [
        ("belgium", cfg.BELGIUM_KAGGLE_DATASET, data_dir / "belgium"),
        ("mapillary", cfg.MAPILLARY_KAGGLE_DATASET, data_dir / "mapillary"),
    ]

    for name, slug, target in jobs:
        if target.exists() and any(target.rglob("*")):
            status[name] = {"status": "available", "path": str(target), "method": "existing files"}
            continue

        print(f"Checking {name} dataset...")
        try:
            _download_kaggle_dataset(slug, target)
            status[name] = {"status": "available", "path": str(target), "method": f"kagglehub: {slug}"}
        except Exception as exc:
            status[name] = {
                "status": "not downloaded",
                "path": str(target),
                "method": f"kagglehub: {slug}",
                "reason": str(exc),
            }
            print(f"Could not download {name}: {exc}")

    _write_status(status)
    return status


def dataset_summary_text(status):
    lines = ["Dataset availability", "--------------------"]
    for name, row in status.items():
        lines.append(f"{name}: {row.get('status')} ({row.get('method')})")
        if row.get("reason"):
            lines.append(f"  reason: {row['reason']}")
    return "\n".join(lines)
