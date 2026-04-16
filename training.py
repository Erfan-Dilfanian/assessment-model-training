import os
import time
import json
import shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from ultralytics import settings
settings.update({"datasets_dir": "."})

# =========================
# User config
# =========================
DATA_YAML = "./garbage-container-detection.v4i.yolov8/data.yaml"
MODEL_NAME = "yolov8n.pt"          # change to yolov8s.pt / yolov8m.pt if needed
RUN_NAME = "bin_detector_vanilla"
PROJECT_DIR = "runs/train"

EPOCHS = 60
IMGSZ = 640
BATCH = 16
WORKERS = 4
PATIENCE = 20

DEVICE = 0 if torch.cuda.is_available() else "cpu"
SAVE_PLOTS = True
OVERWRITE_RUN = False              # set True if you want to delete old run folder


# =========================
# Helpers
# =========================
def print_system_info():
    print("\n========== System Info ==========")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print("=================================\n")


def prepare_run_dir(project_dir: str, run_name: str, overwrite: bool = False) -> Path:
    run_dir = Path(project_dir) / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, train_args: dict):
    metadata = {
        "train_args": train_args,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def plot_results_csv(results_csv: Path, output_dir: Path):
    if not results_csv.exists():
        print(f"Warning: results.csv not found at {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    if "epoch" not in df.columns:
        print("Warning: 'epoch' column not found in results.csv")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_groups = [
        ["train/box_loss", "val/box_loss"],
        ["train/cls_loss", "val/cls_loss"],
        ["train/dfl_loss", "val/dfl_loss"],
        ["metrics/precision(B)", "metrics/recall(B)"],
        ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
    ]

    for group in metric_groups:
        available = [col for col in group if col in df.columns]
        if not available:
            continue

        plt.figure(figsize=(8, 5))
        for col in available:
            plt.plot(df["epoch"], df[col], marker="o", linewidth=1.5, label=col)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(" / ".join(available))
        plt.grid(True)
        plt.legend()
        filename = available[0].replace("/", "_").replace("(", "").replace(")", "") + ".png"
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    print(f"Plots saved to: {output_dir}")


def save_metrics_summary(run_dir: Path, train_metrics, val_metrics, test_metrics=None):
    summary = {
        "train_best_fitness": getattr(train_metrics, "fitness", None),
        "val_map50": getattr(val_metrics.box, "map50", None) if hasattr(val_metrics, "box") else None,
        "val_map50_95": getattr(val_metrics.box, "map", None) if hasattr(val_metrics, "box") else None,
        "val_precision": getattr(val_metrics.box, "mp", None) if hasattr(val_metrics, "box") else None,
        "val_recall": getattr(val_metrics.box, "mr", None) if hasattr(val_metrics, "box") else None,
    }

    if test_metrics is not None and hasattr(test_metrics, "box"):
        summary.update({
            "test_map50": test_metrics.box.map50,
            "test_map50_95": test_metrics.box.map,
            "test_precision": test_metrics.box.mp,
            "test_recall": test_metrics.box.mr,
        })

    pd.DataFrame([summary]).to_csv(run_dir / "summary_metrics.csv", index=False)
    print(f"Summary metrics saved to: {run_dir / 'summary_metrics.csv'}")

def print_best_epoch_from_results(results_csv: Path):
    if not results_csv.exists():
        print(f"Warning: results.csv not found at {results_csv}")
        return None

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    target_col = "metrics/mAP50-95(B)"
    required = ["epoch", target_col, "metrics/precision(B)", "metrics/recall(B)"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: missing columns in results.csv: {missing}")
        return None

    best_idx = df[target_col].idxmax()
    best_row = df.loc[best_idx]

    best_epoch = int(best_row["epoch"])
    best_map = float(best_row[target_col])
    best_precision = float(best_row["metrics/precision(B)"])
    best_recall = float(best_row["metrics/recall(B)"])

    print("\nBest epoch from training history:")
    print(f"Epoch      : {best_epoch}")
    print(f"mAP50-95   : {best_map:.4f}")
    print(f"Precision  : {best_precision:.4f}")
    print(f"Recall     : {best_recall:.4f}")

    return {
        "best_epoch": best_epoch,
        "best_map50_95": best_map,
        "best_precision": best_precision,
        "best_recall": best_recall,
    }

# =========================
# Main
# =========================
def main():
    print_system_info()

    run_dir = prepare_run_dir(PROJECT_DIR, RUN_NAME, overwrite=OVERWRITE_RUN)

    train_args = {
        "data": DATA_YAML,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "workers": WORKERS,
        "patience": PATIENCE,
        "device": DEVICE,
        "project": PROJECT_DIR,
        "name": RUN_NAME,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "plots": SAVE_PLOTS,
        "save": True,
        "cache": False,
    }

    save_run_metadata(run_dir, train_args)

    print("Loading model...")
    model = YOLO(MODEL_NAME)

    print("Starting training...")
    t0 = time.time()

    train_results = model.train(**train_args)

    train_time_sec = time.time() - t0
    print(f"\nTraining finished in {train_time_sec:.2f} seconds")

    with open(run_dir / "training_time.txt", "w") as f:
        f.write(f"Training time (seconds): {train_time_sec:.2f}\n")

    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    print(f"Best weights: {best_weights}")
    print(f"Last weights: {last_weights}")

    if not best_weights.exists():
        raise FileNotFoundError(f"Could not find best weights at: {best_weights}")

    print("\nLoading best model for validation...")
    best_model = YOLO(str(best_weights))

    print("Running validation on val split...")
    val_results = best_model.val(
        data=DATA_YAML,
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_val",
        exist_ok=True,
    )

    print("\nValidation results:")
    print(f"mAP50      : {val_results.box.map50:.4f}")
    print(f"mAP50-95   : {val_results.box.map:.4f}")
    print(f"Precision  : {val_results.box.mp:.4f}")
    print(f"Recall     : {val_results.box.mr:.4f}")

    test_results = None
    try:
        print("\nRunning evaluation on test split...")
        test_results = best_model.val(
            data=DATA_YAML,
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT_DIR,
            name=f"{RUN_NAME}_test",
            exist_ok=True,
        )

        print("\nTest results:")
        print(f"mAP50      : {test_results.box.map50:.4f}")
        print(f"mAP50-95   : {test_results.box.map:.4f}")
        print(f"Precision  : {test_results.box.mp:.4f}")
        print(f"Recall     : {test_results.box.mr:.4f}")

    except Exception as e:
        print(f"\nNo usable test split found in data.yaml, or test evaluation failed: {e}")

    results_csv = run_dir / "results.csv"
    best_epoch_info = print_best_epoch_from_results(results_csv)
    plots_dir = run_dir / "custom_plots"
    plot_results_csv(results_csv, plots_dir)

    save_metrics_summary(run_dir, train_results, val_results, test_results)

    print("\nDone.")
    print(f"Run directory: {run_dir}")
    print(f"Best model path: {best_weights}")


if __name__ == "__main__":
    main()