"""Download the datasets used by the traffic-sign benchmark."""

import config as cfg
from src.datasets_extra import dataset_summary_text, download_all_datasets
from src.utils import make_dirs


if __name__ == "__main__":
    make_dirs()
    status = download_all_datasets(cfg)
    print("\n" + dataset_summary_text(status))
