#!/usr/bin/env python3
"""
Camera Test Script
Test different camera device IDs to find your microscope
"""

import cv2
import time

def test_camera_devices():
    """Test camera devices from 0 to 5."""
    print("🔍 Testing camera devices...")
    print("=" * 40)
    
    working_devices = []
    
    for device_id in range(6):  # Test devices 0-5
        print(f"\n📷 Testing device {device_id}...")
        
        # Try different backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(device_id, backend)
                
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        
                        if width > 0 and height > 0:
                            print(f"  ✅ Device {device_id} (backend {backend}): {width}x{height}")
                            working_devices.append({
                                'device_id': device_id,
                                'backend': backend,
                                'resolution': f"{width}x{height}"
                            })
                            
                            # Show a preview for 2 seconds
                            cv2.imshow(f'Device {device_id} Preview', frame)
                            cv2.waitKey(2000)
                            cv2.destroyAllWindows()
                            
                            cap.release()
                            break  # Found working backend for this device
                        else:
                            print(f"  ❌ Device {device_id} (backend {backend}): Invalid frame size")
                    else:
                        print(f"  ❌ Device {device_id} (backend {backend}): Cannot read frame")
                else:
                    print(f"  ❌ Device {device_id} (backend {backend}): Cannot open")
                
                cap.release()
                
            except Exception as e:
                print(f"  ❌ Device {device_id} (backend {backend}): Error - {e}")
    
    print("\n" + "=" * 40)
    print("📋 SUMMARY:")
    
    if working_devices:
        print("✅ Working camera devices found:")
        for device in working_devices:
            print(f"  - Device {device['device_id']}: {device['resolution']} (backend {device['backend']})")
        
        print(f"\n💡 Recommended: Use device_id={working_devices[0]['device_id']} in your microscope script")
    else:
        print("❌ No working camera devices found")
        print("\n🔧 Troubleshooting tips:")
        print("  1. Make sure your microscope is connected and powered on")
        print("  2. Check if microscope software is running (close it if needed)")
        print("  3. Try different USB ports")
        print("  4. Check Windows Device Manager for camera devices")

if __name__ == "__main__":
    test_camera_devices()