"""Run the traffic-sign recognition benchmark from one entry point."""

import torch

import config as cfg
from src.data import find_imagefolder_datasets, make_loaders
from src.datasets_extra import dataset_summary_text, download_all_datasets
from src.evaluate import evaluate_model, save_failure_cases
from src.models import build_model
from src.plots import (
    plot_confusion,
    plot_training,
    save_classification_report,
    save_history,
    save_robustness,
)
from src.robustness import run_robustness
from src.train import train_model
from src.utils import configure_cpu_threads, get_device, make_dirs, project_path, set_seed, worker_count


def main():
    make_dirs()
    set_seed(cfg.SEED)

    device = get_device()
    cpu_threads = configure_cpu_threads(cfg.CPU_CORES_TO_LEAVE_FREE)
    data_workers = worker_count(cfg.NUM_WORKERS, cfg.CPU_CORES_TO_LEAVE_FREE)

    print("Traffic Sign Recognition and Robustness Benchmark")
    print("-------------------------------------------------")
    print(f"dataset mode:    {cfg.DATASET_MODE}")
    print(f"model:           {cfg.MODEL_NAME}")
    print(f"device:          {device}")
    if device.type == "cuda":
        print(f"gpu:             {torch.cuda.get_device_name(0)}")
        print("mixed precision: enabled" if cfg.USE_AMP else "mixed precision: disabled")
    else:
        print(f"cpu threads:     {cpu_threads}")
    print(f"data workers:    {data_workers}")

    if cfg.DOWNLOAD_ALL_DATASETS:
        status = download_all_datasets(cfg)
        print("\n" + dataset_summary_text(status) + "\n")

    loaders = make_loaders(cfg, device)
    print("\nTraining sources")
    print("----------------")
    for row in loaders.get("source_summary", []):
        note = row.get("note") or ""
        details = ", ".join(f"{k}={v}" for k, v in row.items() if k not in {"name", "note"})
        print(f"{row.get('name')}: {details}" + (f" | {note}" if note else ""))

    # Keep a simple source summary for the README/results folder.
    import csv
    with open(project_path("outputs", "training_sources.csv"), "w", newline="") as f:
        fieldnames = sorted({key for row in loaders.get("source_summary", []) for key in row.keys()}) or ["name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loaders.get("source_summary", []))

    model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES).to(device)

    history = train_model(model, loaders, cfg, device)
    save_history(history)
    plot_training(history)

    results = evaluate_model(model, loaders["test"], device)
    save_classification_report(results["report"])
    plot_confusion(results["confusion"])

    print("\nGTSRB test results")
    print("------------------")
    print(f"test loss: {results['loss']:.4f}")
    print(f"test acc:  {results['acc']:.4f}")

    torch.save({
        "model_name": cfg.MODEL_NAME,
        "num_classes": cfg.NUM_CLASSES,
        "image_size": cfg.IMAGE_SIZE,
        "state_dict": model.state_dict(),
    }, project_path("models", f"{cfg.MODEL_NAME}_traffic_sign.pt"))

    if cfg.RUN_EXTERNAL_DATASET_EVAL:
        extra = find_imagefolder_datasets(cfg)
        if extra:
            print("\nExternal dataset checks")
            print("-----------------------")
            for name, dataset in extra.items():
                print(f"{name}: found {len(dataset)} images across {len(dataset.classes)} folders")
        else:
            print("\nNo ImageFolder-compatible external test set was found for BelgiumTS or Mapillary.")
            print("The full downloads may still be present, but their native annotation formats need mapping before classification evaluation.")

    if cfg.SAVE_FAILURE_CASES:
        n_saved = save_failure_cases(model, loaders["test"], device, cfg.MAX_FAILURE_CASES)
        print(f"saved failure cases: {n_saved}")

    if cfg.RUN_ROBUSTNESS_TESTS:
        rows = run_robustness(model, loaders["test"].dataset, cfg, device)
        save_robustness(rows)

    print("\nSaved outputs")
    print("-------------")
    for p in [
        "figures/graphical_abstract.png",
        "figures/accuracy_curve.png",
        "figures/loss_curve.png",
        "figures/confusion_matrix.png",
        "figures/robustness_comparison.png",
        "outputs/dataset_status.json",
        "outputs/training_sources.csv",
        "outputs/training_history.csv",
        "outputs/classification_report.txt",
        "outputs/robustness_results.csv",
        f"models/{cfg.MODEL_NAME}_traffic_sign.pt",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
