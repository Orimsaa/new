"""
Test Model Loading and Prediction
Tests loading models and making predictions with actual images
"""

import base64
import os
from pathlib import Path

import requests

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"


def load_model(model_name):
    """Load a specific model"""
    print(f"🔄 Loading model: {model_name}")
    try:
        response = requests.post(f"{API_BASE_URL}/model/load/{model_name}")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Model loaded successfully!")
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ Failed to load model: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def encode_image_to_base64(image_path):
    """Helper function to encode image to base64"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def test_prediction_with_image(image_path, expected_category=None):
    """Test prediction with a specific image"""
    print(f"🖼️  Testing prediction with: {image_path}")

    try:
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)

        # Prepare request data
        request_data = {
            "image_data": image_base64,
            "filename": os.path.basename(image_path),
        }

        # Make prediction request
        response = requests.post(
            f"{API_BASE_URL}/predict/single",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "N/A")
            confidence = result.get("confidence", "N/A")
            processing_time = result.get("processing_time", "N/A")

            print(f"✅ Prediction: {prediction}")
            print(f"   Confidence: {confidence}")
            print(f"   Processing Time: {processing_time}s")

            if expected_category:
                if prediction.lower() == expected_category.lower():
                    print(f"✅ Correct prediction! Expected: {expected_category}")
                else:
                    print("⚠️  Different prediction.")
                    print(f"Expected: {expected_category}, Got: {prediction}")

            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Testing Model Loading and Prediction")
    print("=" * 60)

    # Test 1: Load CNN model
    if not load_model("weather-classifier-cnn"):
        print("❌ Failed to load CNN model. Trying other models...")

        # Try other models
        models_to_try = [
            "weather-classifier-efficientnet",
            "weather-classifier-mobilenet",
        ]
        model_loaded = False

        for model in models_to_try:
            if load_model(model):
                model_loaded = True
                break

        if not model_loaded:
            print("❌ No models could be loaded. Exiting...")
            return

    print("\n" + "=" * 60)

    # Test 2: Find and test images
    data_dir = Path("../../data")  # Adjusted path
    if not data_dir.exists():
        data_dir = Path("../data")

    if not data_dir.exists():
        print("❌ Data directory not found. Creating test with sample data...")
        # You could add code here to download sample images or use placeholder
        return

    # Test images from different categories
    categories = ["sunny", "cloudy", "rainy", "foggy"]
    test_count = 0
    success_count = 0

    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            # Get first few images from each category
            images = list(category_path.glob("*.jpg"))[:2]  # Test 2 images per category

            for image_path in images:
                print(f"\n📸 Testing {category} image:")
                if test_prediction_with_image(image_path, category):
                    success_count += 1
                test_count += 1
                print("-" * 40)

    # Summary
    print("\n" + "=" * 60)
    print("📊 PREDICTION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Images Tested: {test_count}")
    print(f"Successful Predictions: {success_count}")
    print(
        f"Success Rate: {(success_count/test_count)*100:.1f}%"
        if test_count > 0
        else "No tests run"
    )

    if success_count == test_count and test_count > 0:
        print("\n🎉 ALL PREDICTION TESTS PASSED!")
    elif success_count > 0:
        print(f"\n✅ {success_count} out of {test_count} predictions successful!")
    else:
        print("\n❌ No successful predictions. Please check the model and API.")


if __name__ == "__main__":
    main()
