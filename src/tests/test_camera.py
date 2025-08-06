#!/usr/bin/env python3
"""
Camera Test Script for MLMCSC Integrated Server

This script tests the camera functionality step by step.
"""

import requests
import json
import time
import base64
import cv2
import numpy as np
from pathlib import Path

def test_camera_functionality():
    """Test camera functionality step by step."""
    base_url = "http://127.0.0.1:8000"
    
    print("🎥 Testing Camera Functionality")
    print("=" * 50)
    
    # Step 1: Check server health
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Server is healthy")
            camera_status = health.get('camera_status', {})
            print(f"   Camera active: {camera_status.get('active', False)}")
            print(f"   Camera initialized: {camera_status.get('initialized', False)}")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure to start the server with: python app.py --server-only")
        return False
    
    # Step 2: Detect available cameras
    print("\n2. Detecting available cameras...")
    try:
        response = requests.get(f"{base_url}/camera/detect", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total_found']} cameras")
            if result['available_cameras']:
                for cam in result['available_cameras']:
                    print(f"   Camera {cam['camera_id']}: {cam['width']}x{cam['height']} @ {cam['fps']} FPS")
                camera_id = result['available_cameras'][0]['camera_id']
            else:
                print("⚠️  No cameras detected")
                return False
        else:
            print(f"❌ Camera detection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Camera detection error: {e}")
        return False
    
    # Step 3: Start the camera
    print(f"\n3. Starting camera {camera_id}...")
    try:
        response = requests.post(f"{base_url}/camera/start", 
                               json={"camera_id": camera_id}, 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Camera started: {result['message']}")
        else:
            print(f"❌ Failed to start camera: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Camera start error: {e}")
        return False
    
    # Step 4: Check camera status
    print("\n4. Checking camera status...")
    try:
        response = requests.get(f"{base_url}/camera/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Camera status:")
            print(f"   Active: {status['camera_active']}")
            print(f"   Initialized: {status['camera_initialized']}")
            
            if not status['camera_active']:
                print("❌ Camera is not active")
                return False
        else:
            print(f"❌ Camera status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Camera status error: {e}")
        return False
    
    # Step 5: Capture a frame
    print("\n5. Capturing a test frame...")
    try:
        response = requests.post(f"{base_url}/camera/capture", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Frame captured successfully")
            print(f"   Frame size: {result['frame_size']['width']}x{result['frame_size']['height']}")
            
            # Save the captured frame
            image_data = result['image_data']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode and save image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                output_path = Path("test_capture.jpg")
                cv2.imwrite(str(output_path), image)
                print(f"   Saved test image to: {output_path}")
            else:
                print("⚠️  Could not decode captured image")
                
        else:
            print(f"❌ Frame capture failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Frame capture error: {e}")
        return False
    
    # Step 6: Test live prediction
    print("\n6. Testing live prediction...")
    try:
        response = requests.post(f"{base_url}/camera/predict_live", timeout=15)
        if response.status_code == 200:
            result = response.json()
            print("✅ Live prediction successful")
            prediction = result['prediction']
            print(f"   Shear percentage: {prediction['shear_percentage']:.2f}%")
            print(f"   Confidence: {prediction['confidence']:.2f}")
        else:
            print(f"⚠️  Live prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"⚠️  Live prediction error: {e}")
    
    # Step 7: Test video feed access
    print("\n7. Testing video feed access...")
    try:
        response = requests.get(f"{base_url}/video_feed", timeout=5, stream=True)
        if response.status_code == 200:
            print("✅ Video feed is accessible")
            print(f"   Content type: {response.headers.get('content-type')}")
            print(f"   Access video feed at: {base_url}/video_feed")
        else:
            print(f"❌ Video feed failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Video feed error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Camera test completed!")
    print(f"🌐 Web interface: {base_url}")
    print(f"📹 Live video feed: {base_url}/video_feed")
    print(f"📚 API docs: {base_url}/docs")
    
    return True

if __name__ == "__main__":
    test_camera_functionality()