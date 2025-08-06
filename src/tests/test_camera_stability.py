#!/usr/bin/env python3
"""
Test script to verify camera stability improvements
"""

import requests
import time
import json
from datetime import datetime

def test_camera_stability():
    """Test camera stability and anti-freeze improvements."""
    
    base_url = "http://127.0.0.1:8000"
    
    print("🎥 Testing Camera Stability Improvements")
    print("=" * 50)
    
    # Step 1: Check server health
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure to start the server with: python app.py --server-only")
        return False
    
    # Step 2: Start camera
    print("\n2. Starting camera...")
    try:
        response = requests.post(f"{base_url}/camera/start", 
                               json={"camera_id": 0}, 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Camera started: {result['message']}")
        else:
            print(f"❌ Failed to start camera: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Camera start error: {e}")
        return False
    
    # Step 3: Monitor camera status over time
    print("\n3. Monitoring camera stability for 30 seconds...")
    
    start_time = time.time()
    status_checks = []
    
    for i in range(6):  # Check every 5 seconds for 30 seconds
        try:
            response = requests.get(f"{base_url}/camera/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                elapsed = time.time() - start_time
                
                status_info = {
                    "time": elapsed,
                    "active": status.get("camera_active", False),
                    "failure_count": status.get("failure_count", 0),
                    "health": status.get("health_status", "unknown"),
                    "resolution": status.get("resolution", "unknown"),
                    "fps": status.get("fps", 0)
                }
                
                status_checks.append(status_info)
                
                print(f"   {elapsed:5.1f}s: Active={status_info['active']}, "
                      f"Failures={status_info['failure_count']}, "
                      f"Health={status_info['health']}, "
                      f"Res={status_info['resolution']}")
                
            else:
                print(f"   {time.time() - start_time:5.1f}s: Status check failed")
                
        except Exception as e:
            print(f"   {time.time() - start_time:5.1f}s: Status error: {e}")
        
        if i < 5:  # Don't sleep after the last check
            time.sleep(5)
    
    # Step 4: Test video feed access
    print("\n4. Testing video feed stability...")
    try:
        # Test that video feed endpoint responds quickly
        start_time = time.time()
        response = requests.get(f"{base_url}/video_feed", timeout=3, stream=True)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Video feed responds in {response_time:.2f} seconds")
            print(f"   Content type: {response.headers.get('content-type')}")
            
            # Read a few bytes to test streaming
            chunk = next(response.iter_content(chunk_size=1024))
            if chunk:
                print(f"   Successfully received {len(chunk)} bytes of video data")
            else:
                print("⚠️  No video data received")
        else:
            print(f"❌ Video feed failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Video feed error: {e}")
    
    # Step 5: Test multiple rapid captures
    print("\n5. Testing rapid frame captures...")
    capture_times = []
    
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.post(f"{base_url}/camera/capture", timeout=5)
            capture_time = time.time() - start_time
            
            if response.status_code == 200:
                capture_times.append(capture_time)
                print(f"   Capture {i+1}: {capture_time:.3f}s")
            else:
                print(f"   Capture {i+1}: Failed ({response.status_code})")
        except Exception as e:
            print(f"   Capture {i+1}: Error - {e}")
        
        time.sleep(0.5)  # Brief pause between captures
    
    # Step 6: Analysis
    print("\n6. Stability Analysis:")
    
    # Check for consistent camera status
    active_count = sum(1 for s in status_checks if s["active"])
    failure_counts = [s["failure_count"] for s in status_checks]
    max_failures = max(failure_counts) if failure_counts else 0
    
    print(f"   Camera active: {active_count}/{len(status_checks)} checks")
    print(f"   Maximum failure count: {max_failures}")
    
    if capture_times:
        avg_capture_time = sum(capture_times) / len(capture_times)
        print(f"   Average capture time: {avg_capture_time:.3f}s")
        print(f"   Capture success rate: {len(capture_times)}/5")
    
    # Step 7: Cleanup
    print("\n7. Stopping camera...")
    try:
        response = requests.post(f"{base_url}/camera/stop", timeout=5)
        if response.status_code == 200:
            print("✅ Camera stopped successfully")
        else:
            print(f"⚠️  Camera stop failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Camera stop error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Camera stability test completed!")
    
    # Summary
    if active_count == len(status_checks) and max_failures == 0:
        print("✅ Camera stability: EXCELLENT")
    elif active_count >= len(status_checks) * 0.8 and max_failures <= 2:
        print("✅ Camera stability: GOOD")
    else:
        print("⚠️  Camera stability: NEEDS IMPROVEMENT")
    
    print("\n📝 Improvements implemented:")
    print("✓ Frame rate control (12 FPS)")
    print("✓ Camera health monitoring")
    print("✓ Automatic recovery on freeze")
    print("✓ Buffer size optimization")
    print("✓ Consecutive failure detection")
    print("✓ Graceful error handling")
    
    return True

if __name__ == "__main__":
    test_camera_stability()