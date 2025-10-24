"""
Model Training, Evaluation, and Registration Script for Weather Classification MLOps Pipeline
This script handles model training, evaluation, and registration with MLflow.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras.applications import VGG16, EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class WeatherClassificationTrainer:
    def __init__(
        self,
        processed_data_path: str,
        models_path: str = "models",
        artifacts_path: str = "artifacts",
        experiment_name: str = "weather_classification",
    ):
        """
        Initialize Weather Classification Trainer

        Args:
            processed_data_path: Path to processed data
            models_path: Path to save trained models
            artifacts_path: Path to save training artifacts
            experiment_name: MLflow experiment name
        """
        self.processed_data_path = Path(processed_data_path)
        self.models_path = Path(models_path)
        self.artifacts_path = Path(artifacts_path)

        # Create directories
        self.models_path.mkdir(exist_ok=True)
        self.artifacts_path.mkdir(exist_ok=True)

        # MLflow setup
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

        # Load metadata
        with open(self.processed_data_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.target_size = tuple(self.metadata["target_size"])
        self.classes = self.metadata["classes"]
        self.num_classes = len(self.classes)

        # Load label encoder
        with open(self.processed_data_path / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        # Setup logging
        logger.add("logs/model_training.log", rotation="10 MB")

    def load_processed_data(self) -> Dict:
        """Load processed data splits"""
        logger.info("Loading processed data...")

        splits = {}
        for split_name in ["train", "val", "test"]:
            split_dir = self.processed_data_path / split_name

            X = np.load(split_dir / "X.npy")
            y = np.load(split_dir / "y.npy")

            # Convert labels to categorical
            y_categorical = tf.keras.utils.to_categorical(
                y, num_classes=self.num_classes
            )

            splits[split_name] = {"X": X, "y": y, "y_categorical": y_categorical}

            logger.info(f"{split_name.upper()} set: {len(X)} samples")

        return splits

    def create_data_generators(self, splits: Dict, batch_size: int = 32) -> Dict:
        """Create data generators with augmentation"""
        logger.info("Creating data generators...")

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode="nearest",
        )

        # Validation and test generators (no augmentation)
        val_test_datagen = ImageDataGenerator()

        generators = {}

        for split_name, split_data in splits.items():
            if split_name == "train":
                datagen = train_datagen
                shuffle = True
            else:
                datagen = val_test_datagen
                shuffle = False

            generator = datagen.flow(
                split_data["X"],
                split_data["y_categorical"],
                batch_size=batch_size,
                shuffle=shuffle,
            )

            generators[split_name] = generator

        return generators

    def create_cnn_model(self, model_config: Dict) -> Model:
        """Create a custom CNN model"""
        model = Sequential(
            [
                Conv2D(
                    32, (3, 3), activation="relu", input_shape=(*self.target_size, 3)
                ),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(256, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                GlobalAveragePooling2D(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(256, activation="relu"),
                Dropout(0.3),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        return model

    def create_transfer_learning_model(
        self, base_model_name: str, model_config: Dict
    ) -> Model:
        """Create a transfer learning model"""
        # Base model mapping
        base_models = {
            "efficientnet": EfficientNetB0,
            "resnet50": ResNet50,
            "mobilenet": MobileNetV2,
            "vgg16": VGG16,
        }

        if base_model_name not in base_models:
            raise ValueError(f"Unsupported base model: {base_model_name}")

        # Create base model
        base_model = base_models[base_model_name](
            weights="imagenet", include_top=False, input_shape=(*self.target_size, 3)
        )

        # Freeze base model layers
        base_model.trainable = model_config.get("trainable_base", False)

        # Add custom top layers
        model = Sequential(
            [
                base_model,
                GlobalAveragePooling2D(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(256, activation="relu"),
                Dropout(0.3),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        return model

    def compile_model(self, model: Model, model_config: Dict) -> Model:
        """Compile the model"""
        optimizer_name = model_config.get("optimizer", "adam")
        learning_rate = model_config.get("learning_rate", 0.001)

        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", "top_3_accuracy"],
        )

        return model

    def create_callbacks(self, model_config: Dict) -> List:
        """Create training callbacks"""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=model_config.get("early_stopping_patience", 10),
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)

        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7, verbose=1
        )
        callbacks.append(reduce_lr)

        # Model checkpoint
        checkpoint_path = self.models_path / "best_model_checkpoint.h5"
        model_checkpoint = ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        callbacks.append(model_checkpoint)

        return callbacks

    def train_model(self, model: Model, generators: Dict, model_config: Dict) -> Dict:
        """Train the model"""
        logger.info("Starting model training...")

        epochs = model_config.get("epochs", 50)

        # Calculate steps per epoch
        train_steps = len(generators["train"])
        val_steps = len(generators["val"])

        # Train the model
        history = model.fit(
            generators["train"],
            epochs=epochs,
            validation_data=generators["val"],
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=self.create_callbacks(model_config),
            verbose=1,
        )

        logger.info("Model training completed!")
        return history.history

    def evaluate_model(self, model: Model, splits: Dict) -> Dict:
        """Evaluate the model on test set"""
        logger.info("Evaluating model on test set...")

        # Predict on test set
        test_predictions = model.predict(splits["test"]["X"])
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = splits["test"]["y"]

        # Calculate metrics
        accuracy = accuracy_score(test_true_classes, test_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true_classes, test_pred_classes, average="weighted"
        )

        # Classification report
        class_names = [
            self.label_encoder.inverse_transform([i])[0]
            for i in range(self.num_classes)
        ]
        classification_rep = classification_report(
            test_true_classes,
            test_pred_classes,
            target_names=class_names,
            output_dict=True,
        )

        # Confusion matrix
        cm = confusion_matrix(test_true_classes, test_pred_classes)

        evaluation_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": classification_rep,
            "confusion_matrix": cm.tolist(),
            "test_predictions": test_predictions.tolist(),
            "test_true_classes": test_true_classes.tolist(),
            "test_pred_classes": test_pred_classes.tolist(),
        }

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")

        return evaluation_results

    def create_visualizations(self, history: Dict, evaluation_results: Dict) -> None:
        """Create training and evaluation visualizations"""
        logger.info("Creating visualizations...")

        # Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy plot
        axes[0, 0].plot(history["accuracy"], label="Training Accuracy")
        axes[0, 0].plot(history["val_accuracy"], label="Validation Accuracy")
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss plot
        axes[0, 1].plot(history["loss"], label="Training Loss")
        axes[0, 1].plot(history["val_loss"], label="Validation Loss")
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Confusion Matrix
        cm = np.array(evaluation_results["confusion_matrix"])
        class_names = [
            self.label_encoder.inverse_transform([i])[0]
            for i in range(self.num_classes)
        ]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Confusion Matrix")
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Actual")

        # Class-wise performance
        class_report = evaluation_results["classification_report"]
        classes = [
            cls
            for cls in class_report.keys()
            if cls not in ["accuracy", "macro avg", "weighted avg"]
        ]
        f1_scores = [class_report[cls]["f1-score"] for cls in classes]

        axes[1, 1].bar(classes, f1_scores)
        axes[1, 1].set_title("F1-Score by Class")
        axes[1, 1].set_xlabel("Class")
        axes[1, 1].set_ylabel("F1-Score")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plots
        plots_path = self.artifacts_path / "training_plots.png"
        plt.savefig(plots_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualizations saved to {plots_path}")

    def save_model_artifacts(
        self, model: Model, history: Dict, evaluation_results: Dict, model_config: Dict
    ) -> str:
        """Save model and training artifacts"""
        logger.info("Saving model artifacts...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (
            f"weather_classifier_{model_config.get('model_type', 'cnn')}_{timestamp}"
        )

        # Save model
        model_path = self.models_path / f"{model_name}.h5"
        model.save(str(model_path))

        # Save training history
        history_path = self.artifacts_path / f"training_history_{timestamp}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save evaluation results
        eval_path = self.artifacts_path / f"evaluation_results_{timestamp}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        # Save model configuration
        config_path = self.artifacts_path / f"model_config_{timestamp}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        logger.info(f"Model artifacts saved with timestamp: {timestamp}")
        return model_name

    def register_model_mlflow(
        self,
        model: Model,
        model_name: str,
        evaluation_results: Dict,
        model_config: Dict,
    ) -> None:
        """Register model in MLflow"""
        logger.info("Registering model in MLflow...")

        with mlflow.start_run(run_name=f"training_{model_name}"):
            # Log parameters
            mlflow.log_params(model_config)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("num_classes", self.num_classes)
            mlflow.log_param("target_size", self.target_size)

            # Log metrics
            mlflow.log_metric("test_accuracy", evaluation_results["accuracy"])
            mlflow.log_metric("test_precision", evaluation_results["precision"])
            mlflow.log_metric("test_recall", evaluation_results["recall"])
            mlflow.log_metric("test_f1_score", evaluation_results["f1_score"])

            # Log class-wise metrics
            class_report = evaluation_results["classification_report"]
            for class_name in self.classes:
                if class_name in class_report:
                    mlflow.log_metric(
                        f"{class_name}_f1", class_report[class_name]["f1-score"]
                    )
                    mlflow.log_metric(
                        f"{class_name}_precision", class_report[class_name]["precision"]
                    )
                    mlflow.log_metric(
                        f"{class_name}_recall", class_report[class_name]["recall"]
                    )

            # Log model
            mlflow.tensorflow.log_model(
                model,
                "model",
                registered_model_name=f"weather_classifier_{model_config.get('model_type', 'cnn')}",
            )

            # Log artifacts
            mlflow.log_artifact(str(self.artifacts_path / "training_plots.png"))

            logger.info("Model registered in MLflow successfully!")

    def run_training_pipeline(self, model_configs: List[Dict]) -> Dict:
        """Run complete training pipeline for multiple model configurations"""
        logger.info("Starting training pipeline...")

        # Load processed data
        splits = self.load_processed_data()

        results = {}
        best_model = None
        best_accuracy = 0

        for i, model_config in enumerate(model_configs):
            logger.info(f"Training model {i+1}/{len(model_configs)}: {model_config}")

            try:
                # Create data generators
                generators = self.create_data_generators(
                    splits, model_config.get("batch_size", 32)
                )

                # Create model
                model_type = model_config.get("model_type", "cnn")
                if model_type == "cnn":
                    model = self.create_cnn_model(model_config)
                else:
                    model = self.create_transfer_learning_model(
                        model_type, model_config
                    )

                # Compile model
                model = self.compile_model(model, model_config)

                # Train model
                history = self.train_model(model, generators, model_config)

                # Evaluate model
                evaluation_results = self.evaluate_model(model, splits)

                # Create visualizations
                self.create_visualizations(history, evaluation_results)

                # Save artifacts
                model_name = self.save_model_artifacts(
                    model, history, evaluation_results, model_config
                )

                # Register in MLflow
                self.register_model_mlflow(
                    model, model_name, evaluation_results, model_config
                )

                # Track best model
                if evaluation_results["accuracy"] > best_accuracy:
                    best_accuracy = evaluation_results["accuracy"]
                    best_model = {
                        "model": model,
                        "model_name": model_name,
                        "config": model_config,
                        "results": evaluation_results,
                    }

                results[model_name] = {
                    "config": model_config,
                    "history": history,
                    "evaluation": evaluation_results,
                }

                logger.info(
                    f"Model {model_name} completed with accuracy: {evaluation_results['accuracy']:.4f}"
                )

            except Exception as e:
                logger.error(f"Error training model {i+1}: {str(e)}")
                continue

        # Save best model summary
        if best_model:
            best_model_summary = {
                "best_model_name": best_model["model_name"],
                "best_accuracy": best_accuracy,
                "best_config": best_model["config"],
                "all_results": {
                    name: {
                        "accuracy": res["evaluation"]["accuracy"],
                        "f1_score": res["evaluation"]["f1_score"],
                    }
                    for name, res in results.items()
                },
            }

            summary_path = self.artifacts_path / "training_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(best_model_summary, f, indent=2)

            logger.success(
                f"Training pipeline completed! Best model: {best_model['model_name']} (Accuracy: {best_accuracy:.4f})"
            )

        return results


def main():
    """Main function to run model training"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Model Training for Weather Classification"
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        default="../artifacts/processed_data",
        help="Path to processed data",
    )
    parser.add_argument(
        "--models_path",
        type=str,
        default="../models",
        help="Path to save trained models",
    )
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="../artifacts",
        help="Path to save training artifacts",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="weather_classification",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Define model configurations to train
    model_configs = [
        {
            "model_type": "cnn",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "early_stopping_patience": 10,
        },
        {
            "model_type": "efficientnet",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 30,
            "early_stopping_patience": 8,
            "trainable_base": False,
        },
        {
            "model_type": "mobilenet",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "batch_size": 32,
            "epochs": 30,
            "early_stopping_patience": 8,
            "trainable_base": False,
        },
    ]

    # Initialize trainer
    trainer = WeatherClassificationTrainer(
        processed_data_path=args.processed_data_path,
        models_path=args.models_path,
        artifacts_path=args.artifacts_path,
        experiment_name=args.experiment_name,
    )

    # Run training pipeline
    results = trainer.run_training_pipeline(model_configs)

    logger.success("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
