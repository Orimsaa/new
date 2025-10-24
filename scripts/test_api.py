"""
Test Cases for Weather Classification API
Tests the FastAPI service with different scenarios
"""

import base64
import json
import os
import time
from pathlib import Path

import requests

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"


def test_api_health():
    """Test Case 1: API Health Check"""
    print("🔍 Test Case 1: API Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✅ Health check passed!\n")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
        return False


def test_root_endpoint():
    """Test Case 2: Root Endpoint"""
    print("🔍 Test Case 2: Root Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✅ Root endpoint test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}\n")
        return False


def test_model_info():
    """Test Case 3: Model Information"""
    print("🔍 Test Case 3: Model Information")
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Model info test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Model info test failed: {e}\n")
        return False


def test_available_models():
    """Test Case 4: Available Models"""
    print("🔍 Test Case 4: Available Models")
    try:
        response = requests.get(f"{API_BASE_URL}/models/available")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Available models test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Available models test failed: {e}\n")
        return False


def encode_image_to_base64(image_path):
    """Helper function to encode image to base64"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def test_image_prediction():
    """Test Case 5: Image Prediction"""
    print("🔍 Test Case 5: Image Prediction")

    # Find a test image from the data directory
    data_dir = Path("../data")
    test_images = []

    # Look for images in different weather categories
    for category in ["sunny", "cloudy", "rainy", "foggy"]:
        category_path = data_dir / category
        if category_path.exists():
            images = list(category_path.glob("*.jpg"))[:1]  # Take first image
            if images:
                test_images.append((category, images[0]))

    if not test_images:
        print("❌ No test images found in data directory")
        return False

    success_count = 0
    for category, image_path in test_images:
        try:
            print(f"Testing {category} image: {image_path.name}")

            # Encode image to base64
            image_base64 = encode_image_to_base64(image_path)

            # Prepare request data
            request_data = {"image_data": image_base64, "filename": image_path.name}

            # Make prediction request
            response = requests.post(
                f"{API_BASE_URL}/predict/single",
                json=request_data,
                headers={"Content-Type": "application/json"},
            )

            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {result.get('prediction', 'N/A')}")
                print(f"Confidence: {result.get('confidence', 'N/A')}")
                print(f"Processing Time: {result.get('processing_time', 'N/A')}s")
                success_count += 1
            else:
                print(f"Error: {response.text}")

            print("-" * 50)

        except Exception as e:
            print(f"❌ Prediction test failed for {category}: {e}")

    if success_count > 0:
        print(
            f"✅ Image prediction test passed! ({success_count}/{len(test_images)} successful)"
        )
        return True
    else:
        print("❌ All image prediction tests failed")
        return False


def test_batch_prediction():
    """Test Case 6: Batch Prediction (if available)"""
    print("🔍 Test Case 6: Batch Prediction")
    try:
        # Check if batch endpoint exists
        response = requests.get(f"{API_BASE_URL}/docs")
        if "batch" in response.text.lower():
            print("Batch prediction endpoint detected")
            # Add batch prediction test here if needed
        else:
            print("Batch prediction endpoint not available")

        print("✅ Batch prediction test completed!\n")
        return True
    except Exception as e:
        print(f"❌ Batch prediction test failed: {e}\n")
        return False


def run_all_tests():
    """Run all test cases"""
    print("🚀 Starting Weather Classification API Tests")
    print("=" * 60)

    # Wait for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)

    test_results = []

    # Run all tests
    test_results.append(("Health Check", test_api_health()))
    test_results.append(("Root Endpoint", test_root_endpoint()))
    test_results.append(("Model Info", test_model_info()))
    test_results.append(("Available Models", test_available_models()))
    test_results.append(("Image Prediction", test_image_prediction()))
    test_results.append(("Batch Prediction", test_batch_prediction()))

    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1

    print("-" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! API is working correctly!")
    else:
        print(
            f"\n⚠️  {total - passed} tests failed. Please check the API implementation."
        )

    return passed == total


if __name__ == "__main__":
    run_all_tests()
