"""
Data Validation Script for Weather Classification MLOps Pipeline
This script validates the quality and integrity of image data.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mlflow
import numpy as np
import pandas as pd
from loguru import logger


class DataValidator:
    def __init__(self, data_path: str, output_path: str = "artifacts"):
        """
        Initialize Data Validator

        Args:
            data_path: Path to the data directory
            output_path: Path to save validation artifacts
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        # Expected classes for weather classification
        self.expected_classes = ["cloudy", "foggy", "rainy", "snowy", "sunny"]

        # Image validation criteria
        self.min_image_size = (32, 32)  # Minimum width, height
        self.max_image_size = (4096, 4096)  # Maximum width, height
        self.supported_formats = [".jpg", ".jpeg", ".png"]

        # Setup logging
        logger.add("logs/data_validation.log", rotation="10 MB")

    def validate_directory_structure(self) -> Dict:
        """Validate the directory structure and class distribution"""
        logger.info("Validating directory structure...")

        validation_results = {
            "directory_exists": self.data_path.exists(),
            "expected_classes_found": [],
            "unexpected_directories": [],
            "class_counts": {},
            "total_images": 0,
        }

        if not validation_results["directory_exists"]:
            logger.error(f"Data directory {self.data_path} does not exist")
            return validation_results

        # Check for expected classes
        for class_name in self.expected_classes:
            class_path = self.data_path / class_name
            if class_path.exists() and class_path.is_dir():
                validation_results["expected_classes_found"].append(class_name)

                # Count images in each class
                image_files = [
                    f
                    for f in class_path.iterdir()
                    if f.suffix.lower() in self.supported_formats
                ]
                validation_results["class_counts"][class_name] = len(image_files)
                validation_results["total_images"] += len(image_files)
            else:
                logger.warning(f"Expected class directory '{class_name}' not found")

        # Check for unexpected directories
        for item in self.data_path.iterdir():
            if item.is_dir() and item.name not in self.expected_classes:
                validation_results["unexpected_directories"].append(item.name)

        logger.info(
            f"Found {len(validation_results['expected_classes_found'])} expected classes"
        )
        logger.info(f"Total images: {validation_results['total_images']}")

        return validation_results

    def validate_images(self) -> Dict:
        """Validate individual images for corruption, size, and format"""
        logger.info("Validating individual images...")

        validation_results = {
            "valid_images": 0,
            "corrupted_images": [],
            "invalid_size_images": [],
            "unsupported_format_images": [],
            "image_statistics": {
                "width_stats": [],
                "height_stats": [],
                "channel_stats": [],
            },
        }

        for class_name in self.expected_classes:
            class_path = self.data_path / class_name
            if not class_path.exists():
                continue

            logger.info(f"Validating images in class: {class_name}")

            for image_file in class_path.iterdir():
                if image_file.suffix.lower() not in self.supported_formats:
                    validation_results["unsupported_format_images"].append(
                        str(image_file)
                    )
                    continue

                try:
                    # Try to load the image
                    image = cv2.imread(str(image_file))

                    if image is None:
                        validation_results["corrupted_images"].append(str(image_file))
                        continue

                    height, width = image.shape[:2]
                    channels = image.shape[2] if len(image.shape) == 3 else 1

                    # Check image size
                    if (
                        width < self.min_image_size[0]
                        or height < self.min_image_size[1]
                        or width > self.max_image_size[0]
                        or height > self.max_image_size[1]
                    ):
                        validation_results["invalid_size_images"].append(
                            {"file": str(image_file), "size": (width, height)}
                        )
                        continue

                    # Collect statistics
                    validation_results["image_statistics"]["width_stats"].append(width)
                    validation_results["image_statistics"]["height_stats"].append(
                        height
                    )
                    validation_results["image_statistics"]["channel_stats"].append(
                        channels
                    )

                    validation_results["valid_images"] += 1

                except Exception as e:
                    logger.error(f"Error processing {image_file}: {str(e)}")
                    validation_results["corrupted_images"].append(str(image_file))

        # Calculate statistics
        if validation_results["image_statistics"]["width_stats"]:
            width_stats = validation_results["image_statistics"]["width_stats"]
            height_stats = validation_results["image_statistics"]["height_stats"]

            validation_results["image_statistics"] = {
                "width": {
                    "mean": np.mean(width_stats),
                    "std": np.std(width_stats),
                    "min": np.min(width_stats),
                    "max": np.max(width_stats),
                },
                "height": {
                    "mean": np.mean(height_stats),
                    "std": np.std(height_stats),
                    "min": np.min(height_stats),
                    "max": np.max(height_stats),
                },
                "channels": {
                    "unique": list(
                        set(validation_results["image_statistics"]["channel_stats"])
                    )
                },
            }

        logger.info(f"Valid images: {validation_results['valid_images']}")
        logger.info(f"Corrupted images: {len(validation_results['corrupted_images'])}")

        return validation_results

    def check_class_balance(self, class_counts: Dict) -> Dict:
        """Check for class imbalance issues"""
        logger.info("Checking class balance...")

        if not class_counts:
            return {"balanced": False, "imbalance_ratio": None, "recommendations": []}

        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)

        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        # Consider dataset balanced if ratio is less than 3:1
        balanced = imbalance_ratio <= 3.0

        recommendations = []
        if not balanced:
            recommendations.append(
                f"Dataset is imbalanced (ratio: {imbalance_ratio:.2f}:1)"
            )
            recommendations.append(
                "Consider data augmentation for underrepresented classes"
            )
            recommendations.append("Or use stratified sampling during train/test split")

        return {
            "balanced": balanced,
            "imbalance_ratio": imbalance_ratio,
            "class_counts": class_counts,
            "recommendations": recommendations,
        }

    def generate_validation_report(self, results: Dict) -> str:
        """Generate a comprehensive validation report"""
        report_path = self.output_path / "data_validation_report.json"

        # Save detailed results
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate summary report
        summary = []
        summary.append("=== DATA VALIDATION REPORT ===")
        summary.append(f"Total Images: {results['structure']['total_images']}")
        summary.append(f"Valid Images: {results['images']['valid_images']}")
        summary.append(
            f"Classes Found: {len(results['structure']['expected_classes_found'])}"
        )
        summary.append(f"Dataset Balanced: {results['balance']['balanced']}")

        if results["images"]["corrupted_images"]:
            summary.append(
                f"⚠️  Corrupted Images: {len(results['images']['corrupted_images'])}"
            )

        if results["images"]["invalid_size_images"]:
            summary.append(
                f"⚠️  Invalid Size Images: {len(results['images']['invalid_size_images'])}"
            )

        if results["balance"]["recommendations"]:
            summary.append("\n📋 Recommendations:")
            for rec in results["balance"]["recommendations"]:
                summary.append(f"  - {rec}")

        summary_text = "\n".join(summary)

        # Save summary
        summary_path = self.output_path / "validation_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        logger.info(f"Validation report saved to {report_path}")
        logger.info(f"Validation summary saved to {summary_path}")

        return summary_text

    def run_validation(self) -> Dict:
        """Run complete data validation pipeline"""
        logger.info("Starting data validation pipeline...")

        # End any existing MLflow run
        mlflow.end_run()

        # MLflow tracking
        with mlflow.start_run(run_name="data_validation"):
            # Validate directory structure
            structure_results = self.validate_directory_structure()

            # Validate images
            image_results = self.validate_images()

            # Check class balance
            balance_results = self.check_class_balance(
                structure_results["class_counts"]
            )

            # Combine results
            complete_results = {
                "structure": structure_results,
                "images": image_results,
                "balance": balance_results,
                "validation_passed": (
                    structure_results["directory_exists"]
                    and len(structure_results["expected_classes_found"]) >= 3
                    and image_results["valid_images"] > 0
                    and len(image_results["corrupted_images"]) == 0
                ),
            }

            # Log metrics to MLflow
            mlflow.log_metric("total_images", structure_results["total_images"])
            mlflow.log_metric("valid_images", image_results["valid_images"])
            mlflow.log_metric(
                "corrupted_images", len(image_results["corrupted_images"])
            )
            mlflow.log_metric(
                "classes_found", len(structure_results["expected_classes_found"])
            )
            mlflow.log_metric(
                "imbalance_ratio", balance_results.get("imbalance_ratio", 0)
            )
            mlflow.log_param("data_path", str(self.data_path))

            # Generate and log report
            summary = self.generate_validation_report(complete_results)
            mlflow.log_artifact(str(self.output_path / "data_validation_report.json"))
            mlflow.log_artifact(str(self.output_path / "validation_summary.txt"))

            logger.info("Data validation completed!")
            print(summary)

            return complete_results


def main():
    """Main function to run data validation with MLflow tracking"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Validation for Weather Classification"
    )
    parser.add_argument(
        "--data_path", type=str, default="../../data", help="Path to the data directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../artifacts",
        help="Path to save validation artifacts",
    )

    args = parser.parse_args()

    # Set MLflow experiment
    mlflow.set_experiment("Weather Classification - Data Validation")

    with mlflow.start_run(run_name="weather_data_validation"):
        logger.info("Starting weather data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # Initialize and run validator
        validator = DataValidator(args.data_path, args.output_path)
        results = validator.run_validation()

        # Log parameters
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("expected_classes", len(validator.expected_classes))
        mlflow.log_param("min_image_size", validator.min_image_size)
        mlflow.log_param("supported_formats", validator.supported_formats)

        # Log metrics - with safe access to results
        mlflow.log_metric("total_images", results.get("total_images", 0))
        mlflow.log_metric("usable_images", results.get("usable_images", 0))
        mlflow.log_metric("corrupted_images", results.get("corrupted_images", 0))
        mlflow.log_metric("invalid_size_images", results.get("invalid_size_images", 0))
        mlflow.log_metric("num_classes", results.get("num_classes", 0))
        mlflow.log_metric(
            "class_imbalance_ratio", results.get("class_imbalance_ratio", 0)
        )

        # Log validation status
        validation_status = (
            "Success" if results.get("validation_passed", False) else "Failed"
        )
        mlflow.log_param("validation_status", validation_status)

        logger.info(f"Data validation run finished with status: {validation_status}")

        # Continue with warning if validation has issues but don't exit with error
        if not results.get("validation_passed", False):
            logger.warning(
                "Data validation found issues but continuing with pipeline..."
            )
        else:
            logger.success("Data validation passed!")


if __name__ == "__main__":
    main()
