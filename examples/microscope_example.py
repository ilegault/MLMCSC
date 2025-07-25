#!/usr/bin/env python3
"""
Example usage of the MicroscopeCapture interface for Koolertron microscope.
This demonstrates all the key features of the microscope interface.
"""

import sys
import time
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.camera.microscope_interface import MicroscopeCapture


def main():
    """Main example demonstrating microscope interface usage."""
    print("Koolertron Microscope Interface Example")
    print("=" * 50)
    
    # Initialize microscope capture (device_id=1 for USB microscope)
    microscope = MicroscopeCapture(
        device_id=1,           # USB microscope (0 for built-in camera)
        target_fps=30,         # 30 FPS target
        resolution=(1920, 1080) # Full HD resolution
    )
    
    try:
        # 1. Connect to microscope
        print("1. Connecting to microscope...")
        if not microscope.connect():
            print("   Failed to connect to USB microscope, trying built-in camera...")
            microscope = MicroscopeCapture(device_id=0)
            if not microscope.connect():
                print("   No cameras available!")
                return
        
        print(f"   ✓ Connected! Resolution: {microscope.get_frame_dimensions()}")
        
        # 2. Capture single frame
        print("\n2. Capturing single frame...")
        frame = microscope.get_frame()
        if frame is not None:
            print(f"   ✓ Frame captured: {frame.shape}")
            cv2.imwrite("microscope_sample.jpg", frame)
            print("   ✓ Sample image saved as 'microscope_sample.jpg'")
        
        # 3. Adjust camera settings
        print("\n3. Adjusting camera settings...")
        settings_result = microscope.adjust_settings(
            brightness=0.6,    # Increase brightness
            contrast=0.8,      # Increase contrast
            saturation=0.5,    # Normal saturation
            exposure=-1,       # Auto exposure
            focus=-1          # Auto focus
        )
        print(f"   Settings applied: {settings_result}")
        
        # 4. Enable auto features
        print("\n4. Enabling auto features...")
        if microscope.enable_auto_exposure(True):
            print("   ✓ Auto-exposure enabled")
        
        if microscope.enable_auto_focus(True):
            print("   ✓ Auto-focus enabled")
        
        # 5. Save calibration
        print("\n5. Saving calibration...")
        if microscope.save_calibration():
            print("   ✓ Calibration saved to data/camera_calibration.json")
        
        # 6. Start video streaming
        print("\n6. Starting video stream for 5 seconds...")
        frame_count = 0
        
        def process_frame(frame):
            nonlocal frame_count
            frame_count += 1
            
            # Add frame counter to image
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame (comment out if running headless)
            # cv2.imshow('Microscope Stream', frame)
            # cv2.waitKey(1)
            
            if frame_count % 30 == 0:  # Print every 30 frames (1 second at 30 FPS)
                print(f"   Streaming... Frame {frame_count}")
        
        if microscope.start_stream(callback=process_frame):
            print("   ✓ Streaming started")
            time.sleep(5)  # Stream for 5 seconds
            microscope.stop_stream()
            print(f"   ✓ Streaming stopped. Total frames: {frame_count}")
        
        # 7. Test resolution change
        print("\n7. Testing resolution change...")
        if microscope.set_resolution(1280, 720):
            print(f"   ✓ Resolution changed to: {microscope.get_frame_dimensions()}")
        
        # 8. Test FPS change
        print("\n8. Testing FPS change...")
        if microscope.set_fps(15):
            print(f"   ✓ FPS changed to: {microscope.get_current_fps()}")
        
        # 9. Capture final frame with new settings
        print("\n9. Capturing final frame with new settings...")
        final_frame = microscope.get_frame()
        if final_frame is not None:
            cv2.imwrite("microscope_final.jpg", final_frame)
            print("   ✓ Final image saved as 'microscope_final.jpg'")
        
        print("\n✓ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        
    finally:
        # Always disconnect
        microscope.disconnect()
        print("\n✓ Disconnected from microscope")
        cv2.destroyAllWindows()


def context_manager_example():
    """Example using context manager for automatic cleanup."""
    print("\nContext Manager Example:")
    print("-" * 30)
    
    try:
        # Using context manager ensures automatic cleanup
        with MicroscopeCapture(device_id=1) as microscope:
            frame = microscope.get_frame()
            if frame is not None:
                print("✓ Context manager example works!")
                print(f"  Frame shape: {frame.shape}")
            else:
                print("✗ No frame captured")
                
    except Exception as e:
        print(f"✗ Context manager error: {e}")


def streaming_example():
    """Example of continuous streaming with frame processing."""
    print("\nStreaming Example:")
    print("-" * 20)
    
    microscope = MicroscopeCapture(device_id=1)
    
    try:
        if microscope.connect():
            print("Starting 10-second stream with frame processing...")
            
            processed_frames = 0
            
            def frame_processor(frame):
                nonlocal processed_frames
                processed_frames += 1
                
                # Example processing: convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Example processing: apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                
                # Save every 60th frame (2 seconds at 30 FPS)
                if processed_frames % 60 == 0:
                    filename = f"processed_frame_{processed_frames}.jpg"
                    cv2.imwrite(filename, blurred)
                    print(f"  Saved processed frame: {filename}")
            
            microscope.start_stream(callback=frame_processor)
            time.sleep(10)
            microscope.stop_stream()
            
            print(f"✓ Processed {processed_frames} frames")
            
    except Exception as e:
        print(f"✗ Streaming error: {e}")
        
    finally:
        microscope.disconnect()


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run additional examples
    context_manager_example()
    streaming_example()
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nTo integrate into your application:")
    print("from src.camera.microscope_interface import MicroscopeCapture")
    print("microscope = MicroscopeCapture(device_id=1)")
    print("microscope.connect()")
    print("frame = microscope.get_frame()")