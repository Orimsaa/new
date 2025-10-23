"""
Model Serving API for Weather Classification MLOps Pipeline
This script provides a REST API for loading trained models and making predictions.
"""

import os
import io
import json
import pickle
import base64
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

import mlflow
import mlflow.tensorflow
from loguru import logger

# Configure logging
logger.add("logs/model_serving.log", rotation="10 MB")

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    image: str  # base64 encoded image

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    image_paths: List[str]

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[Dict]
    model_info: Dict
    processing_time: float

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    classes: List[str]
    target_size: List[int]
    loaded_at: str

class WeatherClassificationAPI:
    def __init__(self, 
                 models_path: str = "../models",
                 artifacts_path: str = "../artifacts",
                 default_model_name: Optional[str] = None):
        """
        Initialize Weather Classification API
        
        Args:
            models_path: Path to saved models
            artifacts_path: Path to model artifacts
            default_model_name: Default model to load
        """
        self.models_path = Path(models_path)
        self.artifacts_path = Path(artifacts_path)
        
        # Model storage
        self.loaded_models = {}
        self.current_model = None
        self.model_metadata = {}
        self.label_encoder = None
        
        # Load default model if specified
        if default_model_name:
            self.load_model(default_model_name)
        else:
            self._load_latest_model()
    
    def _load_latest_model(self) -> None:
        """Load the latest trained model"""
        try:
            # Find the latest model file
            model_files = list(self.models_path.glob("*.h5"))
            if not model_files:
                logger.warning("No model files found in models directory")
                return
            
            # Sort by modification time and get the latest
            latest_model = max(model_files, key=os.path.getmtime)
            model_name = latest_model.stem
            
            logger.info(f"Loading latest model: {model_name}")
            self.load_model(model_name)
            
        except Exception as e:
            logger.error(f"Error loading latest model: {str(e)}")
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            model_path = self.models_path / f"{model_name}.h5"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the model
            from tensorflow.keras.models import load_model as tf_load_model
            model = tf_load_model(str(model_path))
            
            # Load metadata
            metadata_path = self.artifacts_path / "processed_data" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                # Default metadata if not found
                metadata = {
                    'classes': ['cloudy', 'foggy', 'rainy', 'snowy', 'sunny'],
                    'target_size': [224, 224]
                }
            
            # Load label encoder
            label_encoder_path = self.artifacts_path / "processed_data" / "label_encoder.pkl"
            if label_encoder_path.exists():
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
            else:
                # Create a simple label encoder if not found
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                label_encoder.fit(metadata['classes'])
            
            # Store loaded model and metadata
            self.loaded_models[model_name] = model
            self.current_model = model
            self.model_metadata = metadata
            self.label_encoder = label_encoder
            
            logger.success(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image: Input image in various formats
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            target_size = tuple(self.model_metadata['target_size'])
            
            # Handle different input types
            if isinstance(image, bytes):
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image))
            
            if isinstance(image, Image.Image):
                # Convert PIL Image to numpy array
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.array(image)
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")
    
    def predict_single(self, image: Union[np.ndarray, Image.Image, bytes]) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image: Input image
            
        Returns:
            Dict: Prediction results
        """
        if self.current_model is None:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        try:
            start_time = datetime.now()
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.current_model.predict(processed_image)
            
            # Get prediction probabilities and class
            probabilities = predictions[0].tolist()
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            confidence = float(probabilities[predicted_class_idx])
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    self.label_encoder.inverse_transform([i])[0]: prob 
                    for i, prob in enumerate(probabilities)
                },
                'processing_time_seconds': processing_time
            }
            
            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image, bytes]]) -> List[Dict]:
        """
        Make predictions on multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List[Dict]: List of prediction results
        """
        if self.current_model is None:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        try:
            start_time = datetime.now()
            
            # Preprocess all images
            processed_images = []
            for image in images:
                processed_image = self.preprocess_image(image)
                processed_images.append(processed_image[0])  # Remove batch dimension
            
            # Stack images for batch prediction
            batch_images = np.array(processed_images)
            
            # Make batch prediction
            predictions = self.current_model.predict(batch_images)
            
            # Process results
            results = []
            for i, pred in enumerate(predictions):
                probabilities = pred.tolist()
                predicted_class_idx = np.argmax(pred)
                predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                confidence = float(probabilities[predicted_class_idx])
                
                result = {
                    'image_index': i,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': {
                        self.label_encoder.inverse_transform([j])[0]: prob 
                        for j, prob in enumerate(probabilities)
                    }
                }
                results.append(result)
            
            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Batch prediction completed: {len(images)} images in {processing_time:.4f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error making batch prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making batch prediction: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the currently loaded model"""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        return {
            'model_loaded': True,
            'classes': self.model_metadata.get('classes', []),
            'target_size': self.model_metadata.get('target_size', []),
            'num_classes': len(self.model_metadata.get('classes', [])),
            'model_type': type(self.current_model).__name__
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        model_files = list(self.models_path.glob("*.h5"))
        return [model_file.stem for model_file in model_files]

# Initialize API
api_instance = WeatherClassificationAPI()

# Create FastAPI app
app = FastAPI(
    title="Weather Classification API",
    description="MLOps API for weather image classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Weather Classification API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": api_instance.current_model is not None
    }

@app.get("/model/info", response_model=Dict)
async def get_model_info():
    """Get information about the currently loaded model"""
    return api_instance.get_model_info()

@app.get("/models/available")
async def get_available_models():
    """Get list of available models"""
    return {
        "available_models": api_instance.get_available_models(),
        "current_model": "loaded" if api_instance.current_model else "none"
    }

@app.post("/model/load/{model_name}")
async def load_model(model_name: str):
    """Load a specific model"""
    success = api_instance.load_model(model_name)
    if success:
        return {
            "message": f"Model {model_name} loaded successfully",
            "model_info": api_instance.get_model_info()
        }
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found or failed to load")

@app.post("/predict")
async def predict_image(request: PredictionRequest):
    """Make prediction on a base64 encoded image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        
        # Make prediction
        result = api_instance.predict_single(image_data)
        
        return {
            "prediction": result,
            "model_info": {
                "classes": api_instance.model_metadata.get('classes', []),
                "target_size": api_instance.model_metadata.get('target_size', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/single")
async def predict_single_image(file: UploadFile = File(...)):
    """Make prediction on a single uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Make prediction
        result = api_instance.predict_single(image_bytes)
        
        return {
            "filename": file.filename,
            "prediction": result,
            "model_info": {
                "classes": api_instance.model_metadata.get('classes', []),
                "target_size": api_instance.model_metadata.get('target_size', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in single prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """Make predictions on multiple uploaded images"""
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    try:
        # Read all image files
        images = []
        filenames = []
        
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
            
            image_bytes = await file.read()
            images.append(image_bytes)
            filenames.append(file.filename)
        
        # Make batch prediction
        results = api_instance.predict_batch(images)
        
        # Combine results with filenames
        combined_results = []
        for i, result in enumerate(results):
            combined_results.append({
                "filename": filenames[i],
                "prediction": result
            })
        
        return {
            "batch_size": len(files),
            "results": combined_results,
            "model_info": {
                "classes": api_instance.model_metadata.get('classes', []),
                "target_size": api_instance.model_metadata.get('target_size', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/url")
async def predict_from_url(image_url: str):
    """Make prediction on an image from URL"""
    try:
        import requests
        from PIL import Image
        
        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(response.content))
        
        # Make prediction
        result = api_instance.predict_single(image)
        
        return {
            "image_url": image_url,
            "prediction": result,
            "model_info": {
                "classes": api_instance.model_metadata.get('classes', []),
                "target_size": api_instance.model_metadata.get('target_size', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in URL prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get API metrics (placeholder for monitoring)"""
    return {
        "total_predictions": "N/A",  # Would be tracked in production
        "average_response_time": "N/A",
        "model_accuracy": "N/A",
        "uptime": "N/A"
    }

def create_app():
    """Factory function to create the FastAPI app"""
    return app

def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Weather Classification API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models_path", type=str, default="../models", help="Path to models directory")
    parser.add_argument("--artifacts_path", type=str, default="../artifacts", help="Path to artifacts directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Update API instance paths
    global api_instance
    api_instance = WeatherClassificationAPI(
        models_path=args.models_path,
        artifacts_path=args.artifacts_path
    )
    
    logger.info(f"Starting Weather Classification API on {args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "04_load_and_predict:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()