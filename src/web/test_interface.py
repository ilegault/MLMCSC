#!/usr/bin/env python3
"""
Test script for MLMCSC Human-in-the-Loop Web Interface

This script tests the basic functionality of the web interface
including API endpoints and database operations.
"""

import sys
import asyncio
import base64
import json
import logging
from pathlib import Path
import requests
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.web.database import DatabaseManager, LabelRecord, PredictionRecord
from src.web.config import get_config

logger = logging.getLogger(__name__)


def create_test_image():
    """Create a simple test image as base64."""
    import cv2
    import numpy as np
    
    # Create a more realistic test image that might be detected
    img = np.ones((640, 640, 3), dtype=np.uint8) * 50  # Dark background
    
    # Add a more specimen-like object
    # Create a metallic-looking rectangular specimen
    cv2.rectangle(img, (200, 250), (440, 390), (120, 120, 120), -1)
    
    # Add some texture and highlights to make it look more metallic
    cv2.rectangle(img, (210, 260), (430, 380), (140, 140, 140), 2)
    cv2.rectangle(img, (220, 270), (420, 370), (160, 160, 160), 1)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


def test_database():
    """Test database operations."""
    print("Testing database operations...")
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        
        # Test storing a prediction
        prediction = PredictionRecord(
            id=None,
            timestamp="2024-01-15T10:30:00",
            specimen_id=1,
            shear_percentage=75.5,
            confidence=0.85,
            detection_bbox=[100, 150, 200, 180],
            detection_confidence=0.92,
            processing_time=0.45,
            image_data=create_test_image()
        )
        
        pred_id = db_manager.store_prediction(prediction)
        print(f"✓ Stored prediction with ID: {pred_id}")
        
        # Test storing a label
        label = LabelRecord(
            id=None,
            timestamp="2024-01-15T10:35:00",
            specimen_id=1,
            technician_label=78.0,
            model_prediction=75.5,
            model_confidence=0.85,
            technician_id="test_tech",
            notes="Test label",
            image_data=create_test_image()
        )
        
        label_id = db_manager.store_label(label)
        print(f"✓ Stored label with ID: {label_id}")
        
        # Test retrieving data
        predictions = db_manager.get_predictions(limit=5)
        labels = db_manager.get_labels(limit=5)
        
        print(f"✓ Retrieved {len(predictions)} predictions")
        print(f"✓ Retrieved {len(labels)} labels")
        
        # Test metrics calculation
        metrics = db_manager.calculate_accuracy_metrics()
        print(f"✓ Calculated metrics: {metrics}")
        
        print("Database tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False


def test_api_endpoints(base_url="http://localhost:8000"):
    """Test API endpoints."""
    print(f"Testing API endpoints at {base_url}...")
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
            return False
        
        # Test main page
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✓ Main page accessible")
        else:
            print(f"✗ Main page failed: {response.status_code}")
            return False
        
        # Test prediction endpoint
        test_image = create_test_image()
        prediction_data = {
            "image_data": test_image,
            "image_format": "jpg"
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction endpoint working: {result.get('shear_percentage', 'N/A')}%")
            
            # Test label submission
            label_data = {
                "specimen_id": result.get('specimen_id', 1),
                "image_data": test_image,
                "technician_label": 80.0,
                "model_prediction": result.get('shear_percentage', 50.0),
                "model_confidence": result.get('confidence', 0.5),
                "technician_id": "test_user",
                "notes": "API test"
            }
            
            response = requests.post(
                f"{base_url}/submit_label",
                json=label_data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✓ Label submission working")
            else:
                print(f"✗ Label submission failed: {response.status_code}")
                return False
                
        elif response.status_code == 400 and "No specimens detected" in response.text:
            print("⚠ Prediction endpoint working but no specimens detected in test image (expected)")
            print("  This is normal - the model is trained on real Charpy specimens")
            
            # Still test label submission with dummy data
            label_data = {
                "specimen_id": 1,
                "image_data": test_image,
                "technician_label": 80.0,
                "model_prediction": 50.0,
                "model_confidence": 0.5,
                "technician_id": "test_user",
                "notes": "API test with synthetic image"
            }
            
            response = requests.post(
                f"{base_url}/submit_label",
                json=label_data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✓ Label submission working")
            else:
                print(f"✗ Label submission failed: {response.status_code}")
                return False
        else:
            print(f"✗ Prediction endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Test metrics endpoint
        response = requests.get(f"{base_url}/get_metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            print(f"✓ Metrics endpoint working: {metrics.get('total_predictions', 0)} predictions")
        else:
            print(f"✗ Metrics endpoint failed: {response.status_code}")
            return False
        
        # Test history endpoint
        response = requests.get(f"{base_url}/get_history", timeout=10)
        if response.status_code == 200:
            history = response.json()
            print(f"✓ History endpoint working: {len(history.get('items', []))} items")
        else:
            print(f"✗ History endpoint failed: {response.status_code}")
            return False
        
        print("API tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server. Make sure it's running.")
        return False
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        config = get_config()
        print(f"✓ Configuration loaded: {config.host}:{config.port}")
        print(f"✓ Database path: {config.database_path}")
        print(f"✓ Online learning: {config.online_learning_enabled}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("MLMCSC Human-in-the-Loop Interface Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test configuration
    if test_configuration():
        tests_passed += 1
    
    print()
    
    # Test database
    if test_database():
        tests_passed += 1
    
    print()
    
    # Test API (only if server is running)
    print("Note: API tests require the server to be running.")
    print("Start the server with: python run_server.py")
    
    user_input = input("Is the server running? (y/n): ").lower().strip()
    if user_input == 'y':
        if test_api_endpoints():
            tests_passed += 1
    else:
        print("Skipping API tests.")
        total_tests -= 1
    
    print()
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The interface is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    sys.exit(0 if success else 1)