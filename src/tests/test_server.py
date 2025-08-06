#!/usr/bin/env python3
"""
Simple script to test if the MLMCSC server is running properly
"""

import requests
import time
import sys

def test_server():
    """Test if the server is running and responding"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing MLMCSC Server...")
    print("=" * 40)
    
    # Test health endpoint
    try:
        print("📡 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Models loaded:")
            models = health_data.get('models_loaded', {})
            for model, loaded in models.items():
                status = "✅" if loaded else "❌"
                print(f"     {model}: {status}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running with: python app.py")
        return False
    
    # Test main page
    try:
        print("\n🌐 Testing main web interface...")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Main web interface accessible")
        else:
            print(f"⚠️  Main page returned: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Main page test failed: {e}")
    
    # Test API docs
    try:
        print("\n📚 Testing API documentation...")
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation accessible")
        else:
            print(f"⚠️  API docs returned: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs test failed: {e}")
    
    print("\n" + "=" * 40)
    print("✅ Server tests completed!")
    print(f"🌐 Access your application at: {base_url}")
    return True

if __name__ == "__main__":
    if test_server():
        print("\n🎉 Everything looks good!")
    else:
        print("\n❌ Server tests failed!")
        sys.exit(1)