"""
Data Preprocessing Script for Weather Classification MLOps Pipeline
Generates processed datasets (train/val/test) and metadata for training.
- Loads images from data directory with class subfolders
- Resizes to target size
- Splits into train/val/test
- Saves X.npy/y.npy per split
- Saves label_encoder.pkl and metadata.json
- Logs to MLflow and exposes run_id for GitHub Actions via GITHUB_OUTPUT
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASSES = ["cloudy", "foggy", "rainy", "snowy", "sunny"]


def load_images_from_dir(data_dir: Path, target_size: Tuple[int, int]) -> Tuple[np.ndarray, List[str]]:
    """Load and resize images from class subfolders under data_dir.
    Returns X (N,H,W,3) and y (class names).
    """
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return np.empty((0, *target_size, 3)), []

    classes = [
        d.name for d in data_dir.iterdir() if d.is_dir() and d.name in DEFAULT_CLASSES
    ]
    if not classes:
        logger.warning("No class subfolders found. Expected one of: %s", DEFAULT_CLASSES)
        return np.empty((0, *target_size, 3)), []

    for cls in classes:
        class_dir = data_dir / cls
        for f in class_dir.iterdir():
            if f.suffix.lower() in SUPPORTED_FORMATS:
                try:
                    img = Image.open(f).convert("RGB")
                    img = img.resize(target_size)
                    X_list.append(np.array(img, dtype=np.uint8))
                    y_list.append(cls)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
                    continue

    if not X_list:
        return np.empty((0, *target_size, 3)), []

    X = np.stack(X_list, axis=0)
    return X, y_list


def create_synthetic_dataset(target_size: Tuple[int, int], per_class: int = 10) -> Tuple[np.ndarray, List[str]]:
    """Create a tiny synthetic dataset when real data is absent.
    Generates simple color/texture images for DEFAULT_CLASSES.
    """
    rng = np.random.default_rng(42)
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for cls in DEFAULT_CLASSES:
        for i in range(per_class):
            base = rng.integers(0, 255, size=(target_size[1], target_size[0], 3), dtype=np.uint8)
            # Add class-specific tint
            if cls == "sunny":
                base[..., 0] = np.clip(base[..., 0] + 40, 0, 255)
            elif cls == "cloudy":
                base = np.clip(base // 2 + 120, 0, 255)
            elif cls == "rainy":
                base[..., 2] = np.clip(base[..., 2] + 60, 0, 255)
            elif cls == "foggy":
                base = np.clip(base // 3 + 180, 0, 255)
            elif cls == "snowy":
                base = np.clip(base // 4 + 200, 0, 255)

            X_list.append(base)
            y_list.append(cls)

    X = np.stack(X_list, axis=0)
    return X, y_list


def save_split(output_dir: Path, split_name: str, X: np.ndarray, y: np.ndarray) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "X.npy", X)
    np.save(split_dir / "y.npy", y)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data Preprocessing for Weather Classification")
    parser.add_argument("--data_path", type=str, default="../../data", help="Path to raw data")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../artifacts/processed_data",
        help="Output path for processed data",
    )
    parser.add_argument("--target_size", nargs=2, type=int, default=[128, 128], help="Target image size W H")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test split ratio")

    args = parser.parse_args()
    data_dir = Path(args.data_path)
    output_dir = Path(args.output_path)
    target_size = (args.target_size[0], args.target_size[1])

    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow experiment
    mlflow.set_experiment("Weather Classification - Data Preprocessing")

    with mlflow.start_run(run_name="data_preprocessing") as run:
        run_id = run.info.run_id
        logger.info(f"Starting preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_param("data_path", str(data_dir))
        mlflow.log_param("target_size", target_size)
        mlflow.log_param("val_split", args.val_split)
        mlflow.log_param("test_split", args.test_split)

        # Load data or create synthetic if missing
        X, y_names = load_images_from_dir(data_dir, target_size)
        if X.shape[0] == 0:
            logger.warning("No images found; creating synthetic dataset for CI/example use.")
            X, y_names = create_synthetic_dataset(target_size, per_class=10)

        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(DEFAULT_CLASSES)
        y = label_encoder.transform(y_names)

        # Train/Val/Test split
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=args.val_split + args.test_split, random_state=42, stratify=y
        )
        relative_test = (
            args.test_split / (args.val_split + args.test_split)
            if (args.val_split + args.test_split) > 0
            else 0.0
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=relative_test, random_state=42, stratify=y_tmp
        )

        # Save splits
        save_split(output_dir, "train", X_train, y_train)
        save_split(output_dir, "val", X_val, y_val)
        save_split(output_dir, "test", X_test, y_test)

        # Save label encoder
        with open(output_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        # Save metadata
        metadata = {
            "classes": DEFAULT_CLASSES,
            "target_size": list(target_size),
            "num_classes": len(DEFAULT_CLASSES),
            "total_images": int(X.shape[0]),
            "train_count": int(X_train.shape[0]),
            "val_count": int(X_val.shape[0]),
            "test_count": int(X_test.shape[0]),
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # MLflow logging
        mlflow.log_metric("total_images", int(X.shape[0]))
        mlflow.log_metric("train_count", int(X_train.shape[0]))
        mlflow.log_metric("val_count", int(X_val.shape[0]))
        mlflow.log_metric("test_count", int(X_test.shape[0]))
        mlflow.log_artifact(str(output_dir / "metadata.json"))

        # Expose run_id to GitHub Actions step outputs if available
        gh_out = os.environ.get("GITHUB_OUTPUT")
        if gh_out:
            with open(gh_out, "a") as f:
                f.write(f"run_id={run_id}\n")

        logger.success("Data preprocessing completed!")
        print(f"Preprocessing run_id: {run_id}")


if __name__ == "__main__":
    main()