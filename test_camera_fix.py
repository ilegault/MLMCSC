#!/usr/bin/env python3
"""
Quick test to verify the camera fix is working
"""

import requests
import time

def test_camera_fix():
    """Test that the camera no longer freezes."""
    
    base_url = "http://127.0.0.1:8000"
    
    print("🔧 Testing Camera Fix")
    print("=" * 30)
    
    # Test 1: Check server
    print("1. Checking server...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server not responding")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test 2: Start camera
    print("2. Starting camera...")
    try:
        response = requests.post(f"{base_url}/camera/start", 
                               json={"camera_id": 0}, 
                               timeout=10)
        if response.status_code == 200:
            print("✅ Camera started successfully")
        else:
            print(f"❌ Camera start failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Camera start error: {e}")
        return
    
    # Test 3: Monitor for 15 seconds
    print("3. Monitoring camera for 15 seconds...")
    
    for i in range(3):
        try:
            response = requests.get(f"{base_url}/camera/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                active = status.get("camera_active", False)
                failures = status.get("failure_count", 0)
                health = status.get("health_status", "unknown")
                
                print(f"   {(i+1)*5}s: Active={active}, Failures={failures}, Health={health}")
                
                if not active:
                    print("❌ Camera became inactive!")
                    break
            else:
                print(f"   {(i+1)*5}s: Status check failed")
        except Exception as e:
            print(f"   {(i+1)*5}s: Error - {e}")
        
        if i < 2:  # Don't sleep after last check
            time.sleep(5)
    
    # Test 4: Test video feed
    print("4. Testing video feed...")
    try:
        response = requests.get(f"{base_url}/video_feed", timeout=3, stream=True)
        if response.status_code == 200:
            print("✅ Video feed is accessible")
        else:
            print(f"❌ Video feed failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Video feed error: {e}")
    
    print("\n" + "=" * 30)
    print("🎉 Camera fix test completed!")
    print("\n📝 If you see no errors above, the camera should now work without freezing.")
    print("🌐 Open your browser to: http://127.0.0.1:8000/video_feed")

if __name__ == "__main__":
    test_camera_fix()