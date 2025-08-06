#!/usr/bin/env python3
"""
Open the live camera viewer in your default browser.
"""

import webbrowser
import os
from pathlib import Path

def open_camera_viewer():
    """Open the live camera viewer HTML file."""
    
    # Get the path to the HTML file
    html_file = Path(__file__).parent / "live_camera_viewer.html"
    
    if not html_file.exists():
        print("❌ Camera viewer HTML file not found!")
        return False
    
    # Convert to file URL
    file_url = f"file:///{html_file.resolve().as_posix()}"
    
    print("🚀 Opening Live Camera Viewer...")
    print(f"📁 File: {html_file}")
    print("🌐 This will open in your default browser")
    print()
    print("Features available:")
    print("✓ Live video feed from microscope")
    print("✓ Start/Stop camera controls")
    print("✓ Capture frame button")
    print("✓ Live prediction button")
    print()
    
    try:
        webbrowser.open(file_url)
        print("✅ Camera viewer opened successfully!")
        print()
        print("📝 Instructions:")
        print("1. Click 'Start Camera' to begin video feed")
        print("2. Use 'Capture Frame' to take a snapshot")
        print("3. Use 'Live Prediction' to analyze current frame")
        print("4. Click 'Stop Camera' when done")
        print()
        print("⚠️  Make sure your MLMCSC server is running:")
        print("   python app.py --server-only")
        
        return True
        
    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print(f"📂 You can manually open: {html_file}")
        return False

if __name__ == "__main__":
    open_camera_viewer()