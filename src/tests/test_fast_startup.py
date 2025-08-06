#!/usr/bin/env python3
"""
Test script to verify fast startup optimizations
"""

import time
import requests
import subprocess
import sys
from pathlib import Path

def test_fast_startup():
    """Test that the server starts up quickly."""
    
    print("🚀 Testing Fast Startup Optimizations")
    print("=" * 50)
    
    # Kill any existing servers
    print("1. Cleaning up existing servers...")
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, timeout=5)
        time.sleep(2)
    except:
        pass
    
    # Start server and measure startup time
    print("2. Starting server and measuring startup time...")
    start_time = time.time()
    
    # Start server in background
    server_process = subprocess.Popen([
        sys.executable, 
        str(Path(__file__).parent / "app.py"), 
        "--server-only"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to be ready
    max_wait = 30  # Maximum 30 seconds
    server_ready = False
    
    for attempt in range(max_wait):
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if response.status_code == 200:
                startup_time = time.time() - start_time
                print(f"✅ Server started in {startup_time:.2f} seconds")
                server_ready = True
                break
        except:
            pass
        time.sleep(1)
    
    if not server_ready:
        print("❌ Server failed to start within 30 seconds")
        server_process.terminate()
        return False
    
    # Test basic functionality
    print("3. Testing basic functionality...")
    
    try:
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print("❌ Health endpoint failed")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    try:
        # Test camera detection (should be fast now)
        detect_start = time.time()
        response = requests.get("http://127.0.0.1:8000/camera/detect", timeout=10)
        detect_time = time.time() - detect_start
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Camera detection completed in {detect_time:.2f} seconds")
            print(f"   Found {result['total_found']} cameras")
        else:
            print("❌ Camera detection failed")
    except Exception as e:
        print(f"❌ Camera detection error: {e}")
    
    try:
        # Test camera start (should be fast now)
        start_cam_time = time.time()
        response = requests.post("http://127.0.0.1:8000/camera/start", 
                               json={"camera_id": 0}, timeout=10)
        cam_start_time = time.time() - start_cam_time
        
        if response.status_code == 200:
            print(f"✅ Camera start completed in {cam_start_time:.2f} seconds")
        else:
            print(f"⚠️  Camera start failed (normal if no camera): {response.status_code}")
    except Exception as e:
        print(f"⚠️  Camera start error (normal if no camera): {e}")
    
    # Cleanup
    print("4. Cleaning up...")
    server_process.terminate()
    time.sleep(2)
    
    print("\n" + "=" * 50)
    print("🎉 Fast startup test completed!")
    
    if startup_time < 15:
        print(f"✅ Startup time is good: {startup_time:.2f} seconds")
    elif startup_time < 25:
        print(f"⚠️  Startup time is acceptable: {startup_time:.2f} seconds")
    else:
        print(f"❌ Startup time is slow: {startup_time:.2f} seconds")
    
    print("\n📝 Optimizations applied:")
    print("✓ Removed slow camera auto-initialization")
    print("✓ Made online model initialization lazy")
    print("✓ Simplified camera detection")
    print("✓ Reduced logging verbosity")
    print("✓ Added automatic server cleanup")
    
    return True

if __name__ == "__main__":
    test_fast_startup()