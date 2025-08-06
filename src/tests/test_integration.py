#!/usr/bin/env python3
"""
Test script to verify the integrated live microscope functionality
"""

import requests
import time
import json

def test_integration():
    """Test the integrated server functionality."""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing MLMCSC Integrated Server with Live Microscope")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check passed")
            print(f"   Models loaded: {health['models_loaded']}")
            print(f"   Camera status: {health.get('camera_status', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure to start the server with: python app.py --server-only")
        return False
    
    # Test 2: Camera detection
    print("\n2. Testing camera detection...")
    try:
        response = requests.get(f"{base_url}/camera/detect", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Camera detection endpoint working")
            print(f"   Found {result['total_found']} cameras")
            if result['available_cameras']:
                for cam in result['available_cameras']:
                    print(f"   Camera {cam['camera_id']}: {cam['width']}x{cam['height']}")
        else:
            print(f"⚠️  Camera detection returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Camera detection failed: {e}")
    
    # Test 3: Camera status
    print("\n3. Testing camera status...")
    try:
        response = requests.get(f"{base_url}/camera/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("✅ Camera status endpoint working")
            print(f"   Camera active: {status['camera_active']}")
            print(f"   Camera initialized: {status['camera_initialized']}")
        else:
            print(f"❌ Camera status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Camera status error: {e}")
    
    # Test 4: Web interface
    print("\n4. Testing web interface...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Web interface accessible")
        else:
            print(f"❌ Web interface failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Web interface error: {e}")
    
    # Test 5: API documentation
    print("\n5. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation accessible")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed!")
    print(f"🌐 Access your integrated server at: {base_url}")
    print(f"📚 API documentation at: {base_url}/docs")
    print(f"📹 Live video feed at: {base_url}/video_feed")
    
    return True

if __name__ == "__main__":
    test_integration()