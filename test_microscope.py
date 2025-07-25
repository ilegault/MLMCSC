#!/usr/bin/env python3
"""
Test script for the MicroscopeCapture interface.
This script demonstrates the usage of the microscope interface module.
"""

import sys
import time
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.camera.microscope_interface import MicroscopeCapture


def test_microscope_interface():
    """Test the microscope interface functionality."""
    print("Testing Microscope Interface")
    print("=" * 50)
    
    # Test with different device IDs to find available cameras
    for device_id in [0, 1, 2]:
        print(f"\nTesting device ID: {device_id}")
        
        # Initialize microscope capture
        microscope = MicroscopeCapture(device_id=device_id, target_fps=30, resolution=(1920, 1080))
        
        try:
            # Attempt to connect
            if microscope.connect():
                print(f"✓ Successfully connected to camera {device_id}")
                
                # Get camera info
                dimensions = microscope.get_frame_dimensions()
                fps = microscope.get_current_fps()
                print(f"  Resolution: {dimensions[0]}x{dimensions[1]}")
                print(f"  FPS: {fps}")
                
                # Test single frame capture
                frame = microscope.get_frame()
                if frame is not None:
                    print(f"✓ Frame captured successfully: {frame.shape}")
                    
                    # Save test image
                    test_image_path = f"test_frame_device_{device_id}.jpg"
                    cv2.imwrite(test_image_path, frame)
                    print(f"✓ Test image saved: {test_image_path}")
                else:
                    print("✗ Failed to capture frame")
                
                # Test camera settings adjustment
                print("\nTesting camera settings...")
                settings_result = microscope.adjust_settings(
                    brightness=0.6,
                    contrast=0.7,
                    saturation=0.5
                )
                print(f"Settings adjustment results: {settings_result}")
                
                # Test auto-exposure and auto-focus
                if microscope.enable_auto_exposure(True):
                    print("✓ Auto-exposure enabled")
                else:
                    print("✗ Failed to enable auto-exposure")
                
                if microscope.enable_auto_focus(True):
                    print("✓ Auto-focus enabled")
                else:
                    print("✗ Failed to enable auto-focus")
                
                # Test calibration save/load
                if microscope.save_calibration():
                    print("✓ Calibration saved")
                    
                    if microscope.load_calibration():
                        print("✓ Calibration loaded")
                    else:
                        print("✗ Failed to load calibration")
                else:
                    print("✗ Failed to save calibration")
                
                # Test streaming for a short duration
                print("\nTesting video streaming...")
                frame_count = 0
                
                def stream_callback(frame):
                    nonlocal frame_count
                    frame_count += 1
                    if frame_count <= 5:  # Print first 5 frames
                        print(f"  Stream frame {frame_count}: {frame.shape}")
                
                if microscope.start_stream(callback=stream_callback):
                    print("✓ Streaming started")
                    time.sleep(2)  # Stream for 2 seconds
                    microscope.stop_stream()
                    print(f"✓ Streaming stopped. Total frames: {frame_count}")
                else:
                    print("✗ Failed to start streaming")
                
                # Test context manager usage
                print("\nTesting context manager...")
                with MicroscopeCapture(device_id=device_id) as mic:
                    test_frame = mic.get_frame()
                    if test_frame is not None:
                        print("✓ Context manager works correctly")
                    else:
                        print("✗ Context manager failed")
                
                microscope.disconnect()
                print(f"✓ Disconnected from camera {device_id}")
                
                # If we successfully tested one camera, we can stop
                break
                
            else:
                print(f"✗ Failed to connect to camera {device_id}")
                
        except Exception as e:
            print(f"✗ Error testing camera {device_id}: {e}")
            
        finally:
            microscope.disconnect()
    
    print("\n" + "=" * 50)
    print("Microscope interface test completed")


def test_example_usage():
    """Test the exact example usage from the requirements."""
    print("\nTesting Example Usage (as specified in requirements)")
    print("=" * 60)
    
    try:
        # Example usage as requested
        microscope = MicroscopeCapture(device_id=1)
        
        if microscope.connect():
            frame = microscope.get_frame()
            
            if frame is not None:
                print("✓ Example usage works correctly")
                print(f"  Frame shape: {frame.shape}")
            else:
                print("✗ Example usage failed - no frame captured")
                
            microscope.disconnect()
        else:
            print("✗ Example usage failed - could not connect")
            # Try with device_id=0 as fallback
            microscope = MicroscopeCapture(device_id=0)
            if microscope.connect():
                frame = microscope.get_frame()
                if frame is not None:
                    print("✓ Example usage works with device_id=0")
                    print(f"  Frame shape: {frame.shape}")
                microscope.disconnect()
            
    except Exception as e:
        print(f"✗ Example usage error: {e}")


if __name__ == "__main__":
    # Run tests
    test_microscope_interface()
    test_example_usage()
    
    print("\nTo use the microscope interface in your code:")
    print("from src.camera.microscope_interface import MicroscopeCapture")
    print("microscope = MicroscopeCapture(device_id=1)")
    print("microscope.connect()")
    print("frame = microscope.get_frame()")