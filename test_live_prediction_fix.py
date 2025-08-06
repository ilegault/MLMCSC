#!/usr/bin/env python3
"""
Test script to verify the live prediction fix works correctly.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_live_prediction_fix():
    """Test the live prediction endpoint with the new error handling."""
    
    print("🧪 Testing Live Prediction Fix")
    print("=" * 50)
    
    # First, check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not running. Please start the server first.")
            return
        print("✅ Server is running")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Please start the server first.")
        return
    
    # Check detector info
    try:
        response = requests.get(f"{BASE_URL}/camera/detector_info", timeout=5)
        if response.status_code == 200:
            detector_info = response.json()
            print(f"🔍 Detector Info: {json.dumps(detector_info, indent=2)}")
        else:
            print(f"⚠️  Could not get detector info: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Error getting detector info: {e}")
    
    # Check camera status
    try:
        response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
        camera_status = response.json()
        print(f"📹 Camera Status: {camera_status}")
        
        if not camera_status.get("active", False):
            print("📹 Starting camera...")
            start_response = requests.post(f"{BASE_URL}/camera/start", timeout=10)
            if start_response.status_code == 200:
                print("✅ Camera started successfully")
                time.sleep(2)  # Give camera time to initialize
            else:
                print(f"❌ Failed to start camera: {start_response.status_code}")
                return
    except Exception as e:
        print(f"❌ Error with camera: {e}")
        return
    
    # Test live prediction multiple times
    print("\n🔄 Testing live predictions...")
    success_count = 0
    no_specimens_count = 0
    error_count = 0
    
    for i in range(5):
        try:
            print(f"  Test {i+1}/5: ", end="")
            response = requests.post(f"{BASE_URL}/camera/predict_live", timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                
                if status == "success":
                    print("✅ Success - Specimen detected and predicted")
                    success_count += 1
                elif status == "no_specimens":
                    print("ℹ️  No specimens detected (this is now handled gracefully)")
                    no_specimens_count += 1
                else:
                    print(f"⚠️  Unexpected status: {status}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                error_count += 1
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            error_count += 1
        
        time.sleep(1)  # Wait between tests
    
    # Summary
    print("\n📊 Test Results Summary:")
    print(f"  ✅ Successful predictions: {success_count}")
    print(f"  ℹ️  No specimens detected: {no_specimens_count}")
    print(f"  ❌ Errors: {error_count}")
    
    if error_count == 0:
        print("\n🎉 SUCCESS: Live prediction error handling is working correctly!")
        print("   The 'No specimens detected' error is now handled gracefully.")
    else:
        print(f"\n⚠️  There were {error_count} errors. Please check the server logs.")

if __name__ == "__main__":
    test_live_prediction_fix()