#!/usr/bin/env python3
"""
Test script for Live Microscope Feature

This script tests the live microscope endpoints to ensure they work correctly.
"""

import requests
import time
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_camera_endpoints():
    """Test all camera-related endpoints."""
    
    print("Testing Live Microscope Feature...")
    print("=" * 50)
    
    # Test 1: Check camera status
    print("1. Testing camera status...")
    try:
        response = requests.get(f"{BASE_URL}/camera/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✓ Camera status: {status}")
        else:
            print(f"   ✗ Failed to get camera status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error getting camera status: {e}")
    
    # Test 2: Start camera
    print("\n2. Testing camera start...")
    try:
        response = requests.post(f"{BASE_URL}/camera/start", json={"camera_id": 1})
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Camera start result: {result}")
            
            # Wait a moment for camera to initialize
            time.sleep(2)
            
        else:
            print(f"   ✗ Failed to start camera: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ✗ Error starting camera: {e}")
    
    # Test 3: Check camera status again
    print("\n3. Testing camera status after start...")
    try:
        response = requests.get(f"{BASE_URL}/camera/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✓ Camera status after start: {status}")
        else:
            print(f"   ✗ Failed to get camera status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error getting camera status: {e}")
    
    # Test 4: Test video feed endpoint (just check if it responds)
    print("\n4. Testing video feed endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/video_feed", stream=True, timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Video feed endpoint responding")
            print(f"   Content-Type: {response.headers.get('content-type')}")
        else:
            print(f"   ✗ Video feed endpoint failed: {response.status_code}")
    except requests.exceptions.Timeout:
        print("   ⚠ Video feed endpoint timeout (expected for streaming)")
    except Exception as e:
        print(f"   ✗ Error accessing video feed: {e}")
    
    # Test 5: Capture frame
    print("\n5. Testing frame capture...")
    try:
        response = requests.post(f"{BASE_URL}/camera/capture")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Frame captured successfully")
            print(f"   Frame size: {result.get('frame_size', 'Unknown')}")
            print(f"   Timestamp: {result.get('timestamp', 'Unknown')}")
        else:
            print(f"   ✗ Failed to capture frame: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ✗ Error capturing frame: {e}")
    
    # Test 6: Live prediction
    print("\n6. Testing live prediction...")
    try:
        response = requests.post(f"{BASE_URL}/camera/predict_live")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Live prediction successful")
            if 'prediction' in result:
                pred = result['prediction']
                print(f"   Shear percentage: {pred.get('shear_percentage', 'Unknown')}")
                print(f"   Confidence: {pred.get('confidence', 'Unknown')}")
                print(f"   Specimen ID: {pred.get('specimen_id', 'Unknown')}")
        else:
            print(f"   ✗ Failed to run live prediction: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ✗ Error in live prediction: {e}")
    
    # Test 7: Stop camera
    print("\n7. Testing camera stop...")
    try:
        response = requests.post(f"{BASE_URL}/camera/stop")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Camera stop result: {result}")
        else:
            print(f"   ✗ Failed to stop camera: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ✗ Error stopping camera: {e}")
    
    # Test 8: Final status check
    print("\n8. Testing final camera status...")
    try:
        response = requests.get(f"{BASE_URL}/camera/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✓ Final camera status: {status}")
        else:
            print(f"   ✗ Failed to get final camera status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error getting final camera status: {e}")
    
    print("\n" + "=" * 50)
    print("Live Microscope Feature Test Complete!")

def test_health_endpoint():
    """Test the health endpoint to see camera status."""
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Health check: {health}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Error in health check: {e}")

if __name__ == "__main__":
    print("Live Microscope Feature Test")
    print("Make sure the API server is running on http://localhost:8000")
    print()
    
    # Test health first
    test_health_endpoint()
    
    # Test camera endpoints
    test_camera_endpoints()
    
    print("\nNote: Some tests may fail if no camera is connected.")
    print("This is expected behavior for development environments.")