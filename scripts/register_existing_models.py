"""
Register Existing Models with MLflow.
"""

import json
from pathlib import Path

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from loguru import logger


def register_existing_models():
    """Register existing trained models in MLflow Model Registry"""

    # Set MLflow experiment
    mlflow.set_experiment("Weather Classification - Model Registration")

    # Paths
    models_path = Path("../models")
    processed_data_path = Path("../artifacts/processed_data")

    # Load metadata
    with open(processed_data_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Find all model files
    model_files = list(models_path.glob("weather_classifier_*.h5"))

    logger.info(f"Found {len(model_files)} model files to register")

    for model_file in model_files:
        try:
            # Extract model type from filename
            model_name = model_file.stem
            model_type = model_name.split("_")[
                2
            ]  # e.g., 'cnn', 'efficientnet', 'mobilenet'

            logger.info(f"Registering model: {model_name}")

            with mlflow.start_run(run_name=f"register_{model_type}"):
                # Load the model
                model = tf.keras.models.load_model(str(model_file))

                # Log basic parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("model_file", str(model_file))
                mlflow.log_param("num_classes", len(metadata["classes"]))
                mlflow.log_param("target_size", metadata["target_size"])

                # Log model architecture info
                mlflow.log_param("total_params", model.count_params())
                mlflow.log_param(
                    "trainable_params",
                    sum(
                        [
                            tf.keras.backend.count_params(w)
                            for w in model.trainable_weights
                        ]
                    ),
                )

                # Log the model and register it
                model_name_registry = f"weather-classifier-{model_type}"

                mlflow.tensorflow.log_model(
                    model, "model", registered_model_name=model_name_registry
                )

                # Log metadata as artifacts
                mlflow.log_dict(metadata, "metadata.json")

                logger.info(f"Successfully registered {model_name_registry}")

        except Exception as e:
            logger.error(f"Error registering model {model_file}: {str(e)}")
            continue

    logger.success("Model registration completed!")


if __name__ == "__main__":
    register_existing_models()
