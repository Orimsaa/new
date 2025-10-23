"""
Complete API Test Script for Weather Classification MLOps
Tests all endpoints including image prediction
"""

import requests
import json
import base64
import os
from pathlib import Path
import time

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Test 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        result = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Response: {result}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    print("\n🔍 Test 2: Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        result = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Response: {result}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_available_models():
    """Test available models endpoint"""
    print("\n🔍 Test 3: Available Models")
    try:
        response = requests.get(f"{BASE_URL}/models/available")
        result = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Available Models: {result.get('available_models', [])}")
        print(f"   Current Model: {result.get('current_model', 'none')}")
        return response.status_code == 200, result.get('available_models', [])
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, []

def test_model_loading(model_name):
    """Test model loading endpoint"""
    print(f"\n🔍 Test 4: Model Loading ({model_name})")
    try:
        response = requests.post(f"{BASE_URL}/model/load/{model_name}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Message: {result.get('message', 'N/A')}")
            model_info = result.get('model_info', {})
            print(f"   Classes: {model_info.get('classes', [])}")
            print(f"   Target Size: {model_info.get('target_size', [])}")
            return True
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Test 5: Model Info")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        result = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Model Loaded: {result.get('model_loaded', False)}")
        print(f"   Classes: {result.get('classes', [])}")
        print(f"   Target Size: {result.get('target_size', [])}")
        print(f"   Number of Classes: {result.get('num_classes', 0)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_image_prediction():
    """Test image prediction with sample images"""
    print("\n🔍 Test 6: Image Prediction")
    
    # Look for test images in data directory
    data_dir = Path("../data")
    
    # Find images from different weather categories
    test_images = []
    categories = ['cloudy', 'foggy', 'rainy', 'snowy', 'sunny']
    
    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            image_files = list(category_path.glob("*.jpg"))
            if image_files:
                test_images.append((category, image_files[0]))
                break
    
    if not test_images:
        # Fallback: find any image
        for ext in ['.jpg', '.jpeg', '.png']:
            images = list(data_dir.rglob(f"*{ext}"))
            if images:
                test_images.append(("unknown", images[0]))
                break
    
    if not test_images:
        print("   ❌ No test images found")
        return False
    
    success_count = 0
    for category, image_path in test_images:
        print(f"\n   Testing with {category} image: {image_path.name}")
        try:
            # Encode image to base64
            image_base64 = encode_image_to_base64(image_path)
            
            # Prepare request
            payload = {"image": image_base64}
            
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                if predictions:
                    pred = predictions[0]
                    print(f"   ✅ Predicted: {pred.get('class', 'N/A')} (confidence: {pred.get('confidence', 0):.3f})")
                    success_count += 1
                else:
                    print(f"   ❌ No predictions returned")
            else:
                print(f"   ❌ Prediction failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return success_count > 0

def run_comprehensive_tests():
    """Run all API tests"""
    print("🚀 Weather Classification API - Comprehensive Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Health Check
    results['health_check'] = test_health_check()
    
    # Test 2: Root Endpoint
    results['root_endpoint'] = test_root_endpoint()
    
    # Test 3: Available Models
    models_success, available_models = test_available_models()
    results['available_models'] = models_success
    
    # Test 4: Model Loading
    if available_models:
        # Try to load the first available model
        first_model = available_models[0]
        results['model_loading'] = test_model_loading(first_model)
        
        # Test 5: Model Info (after loading)
        if results['model_loading']:
            results['model_info'] = test_model_info()
            
            # Test 6: Image Prediction (after loading)
            results['image_prediction'] = test_image_prediction()
        else:
            results['model_info'] = False
            results['image_prediction'] = False
    else:
        results['model_loading'] = False
        results['model_info'] = False
        results['image_prediction'] = False
    
    # Print Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! API is fully functional.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed! API is working well.")
    else:
        print("⚠️  Several tests failed. Please check the API implementation.")
    
    return results

if __name__ == "__main__":
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    run_comprehensive_tests()