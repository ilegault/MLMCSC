#!/usr/bin/env python3
"""
Simple script to find and identify your Koolertron microscope.
Run this to detect which device ID your microscope is using.
"""

import sys
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.camera.camera_detector import CameraDetector
from src.camera.microscope_interface import MicroscopeCapture


def main():
    """Find the Koolertron microscope."""
    print("🔍 Koolertron Microscope Finder")
    print("=" * 50)
    print("This script will help you find your Koolertron microscope.")
    print("Make sure your microscope is connected via USB.\n")
    
    # Step 1: Comprehensive camera scan
    print("Step 1: Scanning all camera devices...")
    CameraDetector.print_camera_report()
    
    # Step 2: Test each camera with a frame capture
    print("\nStep 2: Testing each camera with frame capture...")
    cameras = CameraDetector.scan_cameras(max_cameras=15)  # Check more devices
    
    microscope_candidates = []
    
    for camera in cameras:
        device_id = camera['device_id']
        print(f"\nTesting device {device_id}: {camera['name']}")
        
        try:
            # Create microscope instance
            microscope = MicroscopeCapture(device_id=device_id)
            
            if microscope.connect():
                # Capture test frame
                frame = microscope.get_frame()
                
                if frame is not None:
                    # Save test image
                    filename = f"camera_test_device_{device_id}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    print(f"  ✅ Frame captured and saved as {filename}")
                    print(f"  📐 Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    
                    # Check if this could be a microscope
                    is_microscope_candidate = (
                        device_id > 0 and  # Not built-in camera
                        (frame.shape[1] >= 1280 or frame.shape[0] >= 720)  # Decent resolution
                    )
                    
                    if is_microscope_candidate:
                        microscope_candidates.append({
                            'device_id': device_id,
                            'name': camera['name'],
                            'resolution': (frame.shape[1], frame.shape[0]),
                            'filename': filename
                        })
                        print(f"  🔬 POTENTIAL MICROSCOPE DETECTED!")
                
                microscope.disconnect()
            else:
                print(f"  ❌ Could not connect to device {device_id}")
                
        except Exception as e:
            print(f"  ❌ Error testing device {device_id}: {e}")
    
    # Step 3: Report findings
    print("\n" + "=" * 50)
    print("🔬 MICROSCOPE DETECTION RESULTS")
    print("=" * 50)
    
    if microscope_candidates:
        print(f"Found {len(microscope_candidates)} potential microscope(s):")
        
        for i, candidate in enumerate(microscope_candidates, 1):
            print(f"\n🔬 Candidate {i}:")
            print(f"   Device ID: {candidate['device_id']}")
            print(f"   Name: {candidate['name']}")
            print(f"   Resolution: {candidate['resolution'][0]}x{candidate['resolution'][1]}")
            print(f"   Test Image: {candidate['filename']}")
        
        # Recommend the best candidate
        best_candidate = max(microscope_candidates, 
                           key=lambda x: x['resolution'][0] * x['resolution'][1])
        
        print(f"\n🎯 RECOMMENDED MICROSCOPE:")
        print(f"   Device ID: {best_candidate['device_id']}")
        print(f"   Use this in your code:")
        print(f"   microscope = MicroscopeCapture(device_id={best_candidate['device_id']})")
        
        # Test the recommended microscope
        print(f"\n🧪 Testing recommended microscope (device {best_candidate['device_id']})...")
        test_recommended_microscope(best_candidate['device_id'])
        
    else:
        print("❌ No microscope candidates found.")
        print("\nPossible reasons:")
        print("1. Microscope is not connected")
        print("2. Microscope drivers are not installed")
        print("3. Microscope is using a device ID > 15")
        print("4. Microscope requires specific software to be recognized")
        
        print("\n🔧 Troubleshooting steps:")
        print("1. Check USB connection")
        print("2. Try different USB ports")
        print("3. Install microscope drivers if available")
        print("4. Check Windows Device Manager for unrecognized devices")
        print("5. Try the microscope software that came with it first")


def test_recommended_microscope(device_id: int):
    """Test the recommended microscope with full functionality."""
    try:
        microscope = MicroscopeCapture(device_id=device_id)
        
        if microscope.connect():
            print("✅ Connected successfully")
            
            # Test settings
            settings_result = microscope.adjust_settings(
                brightness=0.6,
                contrast=0.7,
                saturation=0.5
            )
            print(f"🔧 Settings test: {settings_result}")
            
            # Test auto features
            auto_exposure = microscope.enable_auto_exposure(True)
            auto_focus = microscope.enable_auto_focus(True)
            print(f"🤖 Auto-exposure: {'✅' if auto_exposure else '❌'}")
            print(f"🤖 Auto-focus: {'✅' if auto_focus else '❌'}")
            
            # Test calibration
            if microscope.save_calibration():
                print("💾 Calibration save: ✅")
            else:
                print("💾 Calibration save: ❌")
            
            # Capture final test image
            frame = microscope.get_frame()
            if frame is not None:
                cv2.imwrite(f"microscope_final_test_device_{device_id}.jpg", frame)
                print(f"📸 Final test image saved: microscope_final_test_device_{device_id}.jpg")
            
            microscope.disconnect()
            print("✅ All tests completed successfully!")
            
        else:
            print("❌ Could not connect for detailed testing")
            
    except Exception as e:
        print(f"❌ Error during detailed testing: {e}")


if __name__ == "__main__":
    try:
        main()
        
        print("\n" + "=" * 50)
        print("🎉 Microscope detection completed!")
        print("Check the generated test images to verify camera functionality.")
        print("Use the recommended device ID in your microscope applications.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Detection cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your camera connections and try again.")