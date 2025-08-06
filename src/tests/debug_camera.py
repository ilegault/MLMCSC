#!/usr/bin/env python3
"""
Camera Debug Script

This script helps debug camera issues by testing OpenCV camera access directly.
"""

import cv2
import sys
import time

def test_camera_direct(camera_id):
    """Test camera access directly with OpenCV."""
    print(f"\n=== Testing Camera {camera_id} ===")
    
    # Test different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Any Available")
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTrying {backend_name} backend...")
        
        try:
            cap = cv2.VideoCapture(camera_id, backend_id)
            
            if not cap.isOpened():
                print(f"  ✗ Failed to open camera with {backend_name}")
                cap.release()
                continue
            
            print(f"  ✓ Camera opened with {backend_name}")
            
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  Properties: {int(width)}x{int(height)} @ {fps:.1f}fps")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ Successfully read frame: {frame.shape}")
                
                # Try to set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                new_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                new_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"  After setting: {int(new_width)}x{int(new_height)}")
                
                cap.release()
                return True
            else:
                print(f"  ✗ Could not read frame")
                cap.release()
                
        except Exception as e:
            print(f"  ✗ Error with {backend_name}: {e}")
            try:
                cap.release()
            except:
                pass
    
    return False

def detect_all_cameras():
    """Detect all available cameras."""
    print("=== Camera Detection ===")
    print("Scanning for available cameras...")
    
    working_cameras = []
    
    for camera_id in range(10):  # Test cameras 0-9
        print(f"\nTesting camera {camera_id}...")
        
        if test_camera_direct(camera_id):
            working_cameras.append(camera_id)
            print(f"  ✓ Camera {camera_id} is working!")
        else:
            print(f"  ✗ Camera {camera_id} not available")
    
    print(f"\n=== Summary ===")
    if working_cameras:
        print(f"Found {len(working_cameras)} working camera(s): {working_cameras}")
        print("\nRecommendations:")
        for cam_id in working_cameras:
            print(f"  • Try using camera {cam_id} in the web interface")
    else:
        print("No working cameras found!")
        print("\nTroubleshooting:")
        print("  • Make sure a camera is connected")
        print("  • Check if camera drivers are installed")
        print("  • Close other applications that might be using the camera")
        print("  • Try running this script as administrator")
        print("  • Check Windows Camera privacy settings")
    
    return working_cameras

def test_opencv_installation():
    """Test if OpenCV is properly installed."""
    print("=== OpenCV Installation Test ===")
    
    try:
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functionality
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', test_image)
        
        if ret:
            print("✓ OpenCV image encoding works")
        else:
            print("✗ OpenCV image encoding failed")
            
        print("✓ OpenCV installation appears to be working")
        return True
        
    except Exception as e:
        print(f"✗ OpenCV installation issue: {e}")
        return False

def main():
    """Main function."""
    print("Camera Debug Script")
    print("=" * 50)
    
    # Test OpenCV installation
    if not test_opencv_installation():
        print("\nPlease fix OpenCV installation before proceeding.")
        return
    
    # Detect cameras
    working_cameras = detect_all_cameras()
    
    # If cameras found, offer to test them
    if working_cameras:
        print(f"\nWould you like to test camera display? (y/n)")
        choice = input().lower().strip()
        
        if choice == 'y':
            camera_id = working_cameras[0]
            print(f"\nTesting camera {camera_id} display...")
            print("Press 'q' to quit the camera test")
            
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow(f'Camera {camera_id} Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Failed to read frame")
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                print("Camera test completed")
            else:
                print("Failed to open camera for display test")
    
    print("\n" + "=" * 50)
    print("Debug completed!")

if __name__ == "__main__":
    main()