#!/usr/bin/env python3
"""
Rapid Charpy Specimen Data Collection System

This script captures images continuously while you move specimens around,
automatically saving frames for training data collection. Perfect for
quickly building a large dataset of Charpy specimens in various positions.

Features:
- Continuous video capture with automatic frame saving
- Smart frame filtering to avoid duplicate/similar images
- Real-time preview with capture indicators
- Keyboard controls for different capture modes
- Automatic file naming and organization
- Quality filtering to ensure good training images
"""

import cv2
import numpy as np
import time
import os
import threading
from pathlib import Path
from typing import Optional, Callable
import json
import hashlib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Import your existing microscope interface
import sys

sys.path.append('src')
from src.camera.microscope_interface import MicroscopeCapture


class RapidCharpyCapture:
    """Rapid data collection system for Charpy specimens."""

    def __init__(self, device_id: int = 1, output_dir: str = "charpy_dataset_raw"):
        """
        Initialize rapid capture system.

        Args:
            device_id: Microscope camera device ID
            output_dir: Directory to save captured images
        """
        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Capture settings
        self.auto_capture_interval = 0.5  # seconds between auto captures
        self.similarity_threshold = 0.85  # Avoid capturing similar images
        self.min_frame_quality = 50  # Minimum image quality score

        # State tracking
        self.microscope: Optional[MicroscopeCapture] = None
        self.is_capturing = False
        self.is_running = False  # Overall system running state
        self.capture_mode = "manual"  # "manual", "auto", "motion"
        self.captured_count = 0
        self.last_capture_time = 0
        self.previous_frame_hash = None
        self.capture_thread = None

        # Statistics
        self.total_frames = 0
        self.saved_frames = 0
        self.skipped_similar = 0
        self.skipped_quality = 0

        print(f"🔬 Rapid Charpy Capture System Initialized")
        print(f"📁 Output directory: {self.output_dir}")

    def initialize_camera(self) -> bool:
        """Initialize microscope camera with optimal settings."""
        try:
            print("🔌 Connecting to microscope...")
            self.microscope = MicroscopeCapture(
                device_id=self.device_id,
                target_fps=30,
                resolution=(1280, 720)  # Good balance of quality and speed
            )

            if not self.microscope.connect():
                print("❌ Failed to connect to microscope")
                return False

            # Apply optimal settings for specimen capture
            settings = {
                'brightness': 0.6,
                'contrast': 0.8,
                'saturation': 0.6,
                'exposure': -1,  # Auto exposure
                'focus': -1  # Auto focus
            }

            self.microscope.adjust_settings(**settings)
            print("✅ Microscope connected and configured")
            return True

        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False

    def calculate_frame_hash(self, frame: np.ndarray) -> str:
        """Calculate a hash for frame similarity comparison."""
        # Resize to small size for fast comparison
        small_frame = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray.tobytes()).hexdigest()

    def calculate_image_quality(self, frame: np.ndarray) -> float:
        """Calculate image quality score (0-100)."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate variance of Laplacian (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Calculate brightness (avoid too dark/bright images)
        brightness = np.mean(gray)
        brightness_score = 100 - abs(brightness - 128) / 128 * 100

        # Calculate contrast
        contrast = gray.std()
        contrast_score = min(contrast / 50 * 100, 100)

        # Combined quality score
        quality = (laplacian_var / 10 + brightness_score + contrast_score) / 3
        return min(quality, 100)

    def is_frame_similar(self, frame: np.ndarray) -> bool:
        """Check if frame is too similar to previously captured frame."""
        if self.previous_frame_hash is None:
            return False

        current_hash = self.calculate_frame_hash(frame)

        # Simple hash comparison for now
        # Could be enhanced with more sophisticated similarity detection
        similarity = 1.0 if current_hash == self.previous_frame_hash else 0.0

        return similarity > self.similarity_threshold

    def should_capture_frame(self, frame: np.ndarray) -> tuple[bool, str]:
        """Determine if frame should be captured."""
        # Check image quality
        quality = self.calculate_image_quality(frame)
        if quality < self.min_frame_quality:
            return False, f"low_quality_{quality:.1f}"

        # Check similarity
        if self.is_frame_similar(frame):
            return False, "too_similar"

        # Check timing for auto mode
        if self.capture_mode == "auto":
            current_time = time.time()
            if current_time - self.last_capture_time < self.auto_capture_interval:
                return False, "too_soon"

        return True, "good"

    def save_frame(self, frame: np.ndarray, reason: str = "manual") -> str:
        """Save frame to disk with metadata."""
        timestamp = int(time.time() * 1000)  # millisecond precision
        filename = f"charpy_{timestamp:013d}_{reason}_{self.captured_count:04d}.jpg"
        filepath = self.output_dir / filename

        # Save image
        cv2.imwrite(str(filepath), frame)

        # Save metadata
        metadata = {
            'filename': filename,
            'timestamp': timestamp,
            'capture_reason': reason,
            'capture_mode': self.capture_mode,
            'image_quality': self.calculate_image_quality(frame),
            'frame_shape': frame.shape,
            'capture_number': self.captured_count
        }

        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update tracking
        self.captured_count += 1
        self.saved_frames += 1
        self.last_capture_time = time.time()
        self.previous_frame_hash = self.calculate_frame_hash(frame)

        return filename

    def draw_capture_interface(self, frame: np.ndarray) -> np.ndarray:
        """Draw capture interface overlay on frame."""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Status panel background
        cv2.rectangle(display_frame, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (350, 200), (255, 255, 255), 2)

        # Status information
        status_lines = [
            f"📸 Rapid Charpy Capture",
            f"Mode: {self.capture_mode.upper()}",
            f"Captured: {self.captured_count}",
            f"Total frames: {self.total_frames}",
            f"Saved: {self.saved_frames}",
            f"Skipped (similar): {self.skipped_similar}",
            f"Skipped (quality): {self.skipped_quality}",
            f"Success rate: {self.saved_frames / max(self.total_frames, 1) * 100:.1f}%"
        ]

        for i, line in enumerate(status_lines):
            y = 35 + i * 20
            cv2.putText(display_frame, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls panel
        cv2.rectangle(display_frame, (10, h - 120), (400, h - 10), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, h - 120), (400, h - 10), (255, 255, 255), 2)

        controls = [
            "SPACE - Manual Capture",
            "A - Auto Mode ON/OFF",
            "M - Motion Detection ON/OFF",
            "Q/ESC - Quit",
            "R - Reset counters"
        ]

        for i, control in enumerate(controls):
            y = h - 95 + i * 16
            cv2.putText(display_frame, control, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Capture indicator
        if self.capture_mode == "auto":
            # Auto mode indicator
            cv2.circle(display_frame, (w - 50, 50), 20, (0, 255, 0), -1)
            cv2.putText(display_frame, "AUTO", (w - 70, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Quality indicator
        current_quality = self.calculate_image_quality(frame)
        quality_color = (0, 255, 0) if current_quality > 70 else (0, 255, 255) if current_quality > 50 else (0, 0, 255)
        cv2.rectangle(display_frame, (w - 100, 80), (w - 20, 100), quality_color, -1)
        cv2.putText(display_frame, f"Q:{current_quality:.0f}", (w - 95, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return display_frame

    def manual_capture(self):
        """Trigger a manual capture from GUI."""
        if not self.is_running or not self.microscope:
            return False
            
        frame = self.microscope.get_frame()
        if frame is None:
            return False
            
        should_capture, reason = self.should_capture_frame(frame)
        if should_capture:
            filename = self.save_frame(frame, "manual_gui")
            print(f"📸 Manual GUI capture: {filename}")
            return True
        else:
            print(f"⚠️ Frame not captured: {reason}")
            return False

    def start_capture(self):
        """Start the capture system."""
        if self.is_running:
            print("⚠️ Capture system is already running")
            return False
            
        if not self.initialize_camera():
            return False

        print("\n🎬 Starting capture session...")
        self.is_running = True
        self.is_capturing = True
        
        # Start capture in separate thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_capture(self):
        """Stop the capture system."""
        print("🛑 Stopping capture session...")
        self.is_capturing = False
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        self.cleanup()

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        try:
            while self.is_capturing and self.is_running:
                # Get frame
                frame = self.microscope.get_frame()
                if frame is None:
                    print("⚠️ Failed to get frame")
                    time.sleep(0.1)
                    continue

                self.total_frames += 1

                # Check if we should auto-capture
                should_capture, reason = self.should_capture_frame(frame)

                if self.capture_mode == "auto" and should_capture:
                    filename = self.save_frame(frame, "auto")
                    print(f"📸 Auto-captured: {filename}")
                elif not should_capture and reason == "too_similar":
                    self.skipped_similar += 1
                elif not should_capture and reason.startswith("low_quality"):
                    self.skipped_quality += 1

                # Draw interface
                display_frame = self.draw_capture_interface(frame)

                # Show frame
                cv2.imshow('Rapid Charpy Capture', display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # Manual capture
                    if should_capture:
                        filename = self.save_frame(frame, "manual")
                        print(f"📸 Manual capture: {filename}")
                    else:
                        print(f"⚠️ Frame not captured: {reason}")

                elif key == ord('a') or key == ord('A'):  # Toggle auto mode
                    self.capture_mode = "auto" if self.capture_mode != "auto" else "manual"
                    print(f"🔄 Switched to {self.capture_mode} mode")

                elif key == ord('m') or key == ord('M'):  # Toggle motion detection
                    self.capture_mode = "motion" if self.capture_mode != "motion" else "manual"
                    print(f"🔄 Switched to {self.capture_mode} mode")

                elif key == ord('r') or key == ord('R'):  # Reset counters
                    self.reset_counters()
                    print("🔄 Counters reset")

                elif key == 27 or key == ord('q') or key == ord('Q'):  # Quit
                    self.stop_capture()
                    break

                # Update display every few frames
                if self.total_frames % 30 == 0:  # Every ~1 second at 30fps
                    self.print_status()

        except KeyboardInterrupt:
            print("\n⚠️ Capture interrupted by user")
        except Exception as e:
            print(f"❌ Error in capture loop: {e}")
        finally:
            self.is_capturing = False

    def reset_counters(self):
        """Reset all counters."""
        self.total_frames = 0
        self.saved_frames = 0
        self.skipped_similar = 0
        self.skipped_quality = 0
        self.captured_count = 0

    def print_status(self):
        """Print current capture status."""
        success_rate = self.saved_frames / max(self.total_frames, 1) * 100
        print(f"📊 Status: {self.saved_frames} saved | "
              f"{self.skipped_similar} similar | "
              f"{self.skipped_quality} low quality | "
              f"{success_rate:.1f}% success rate")

    def cleanup(self):
        """Clean up resources and print final summary."""
        self.is_capturing = False
        self.is_running = False

        if self.microscope:
            self.microscope.disconnect()
            self.microscope = None

        cv2.destroyAllWindows()

        # Final summary
        print("\n" + "=" * 50)
        print("📈 CAPTURE SESSION SUMMARY")
        print("=" * 50)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📸 Total frames processed: {self.total_frames}")
        print(f"✅ Images saved: {self.saved_frames}")
        print(f"⚠️ Skipped (similar): {self.skipped_similar}")
        print(f"⚠️ Skipped (quality): {self.skipped_quality}")

        if self.total_frames > 0:
            success_rate = self.saved_frames / self.total_frames * 100
            print(f"📊 Success rate: {success_rate:.1f}%")

        if self.last_capture_time > 0:
            session_duration = time.time() - self.last_capture_time
            print(f"⏱️ Session duration: {session_duration:.1f} seconds")
        print("=" * 50)

        if self.saved_frames > 0:
            print("🎉 Ready for annotation! Next steps:")
            print("   1. Review captured images")
            print("   2. Use annotation tool to label specimens")
            print("   3. Train custom Charpy detection model")


class CharpyCaptureGUI:
    """GUI interface for the Rapid Charpy Capture System."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rapid Charpy Data Collection System")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Capture system
        self.capture_system = None
        self.status_update_job = None
        
        # Create GUI elements
        self.create_widgets()
        self.update_status()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔬 Rapid Charpy Data Collection", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)
        
        # Device ID
        ttk.Label(settings_frame, text="Camera Device ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.device_id_var = tk.StringVar(value="1")
        device_id_entry = ttk.Entry(settings_frame, textvariable=self.device_id_var, width=10)
        device_id_entry.grid(row=0, column=1, sticky=tk.W)
        
        # Output directory
        ttk.Label(settings_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.output_dir_var = tk.StringVar(value="data/charpy_training_images")
        output_dir_entry = ttk.Entry(settings_frame, textvariable=self.output_dir_var, width=40)
        output_dir_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0), padx=(0, 10))
        
        browse_btn = ttk.Button(settings_frame, text="Browse", command=self.browse_output_dir)
        browse_btn.grid(row=1, column=2, pady=(10, 0))
        
        # Capture mode
        ttk.Label(settings_frame, text="Capture Mode:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.capture_mode_var = tk.StringVar(value="manual")
        mode_combo = ttk.Combobox(settings_frame, textvariable=self.capture_mode_var, 
                                 values=["manual", "auto", "motion"], state="readonly", width=15)
        mode_combo.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(control_frame, text="🎬 Start Capture", 
                                   command=self.start_capture, style="Accent.TButton")
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="🛑 Stop Capture", 
                                  command=self.stop_capture, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Manual capture button
        self.capture_btn = ttk.Button(control_frame, text="📸 Manual Capture", 
                                     command=self.manual_capture, state="disabled")
        self.capture_btn.grid(row=0, column=2, padx=(0, 10))
        
        # Reset button
        self.reset_btn = ttk.Button(control_frame, text="🔄 Reset Counters", 
                                   command=self.reset_counters)
        self.reset_btn.grid(row=0, column=3)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Status text
        self.status_text = tk.Text(status_frame, height=15, width=70, wrap=tk.WORD)
        status_scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        status_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Instructions
        instructions = """
📋 INSTRUCTIONS:
1. Set your camera device ID (usually 0 or 1)
2. Choose output directory for saved images
3. Select capture mode:
   • Manual: Use 'Manual Capture' button or SPACE in camera window
   • Auto: Automatically captures at intervals
   • Motion: Captures when motion is detected
4. Click 'Start Capture' to begin
5. Position Charpy specimens and move them around
6. Use 'Manual Capture' button or camera window controls
7. Click 'Stop Capture' when done

🎮 GUI CONTROLS:
• Start Capture - Initialize camera and begin session
• Stop Capture - End session and close camera
• Manual Capture - Take a photo right now
• Reset Counters - Clear statistics

🎮 CAMERA WINDOW CONTROLS:
• SPACE - Manual capture
• A - Toggle auto mode
• M - Toggle motion mode  
• R - Reset counters
• Q/ESC - Stop capture
        """
        
        self.status_text.insert(tk.END, instructions)
        self.status_text.config(state=tk.DISABLED)
        
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
            
    def start_capture(self):
        """Start the capture system."""
        try:
            device_id = int(self.device_id_var.get())
        except ValueError:
            messagebox.showerror("Error", "Device ID must be a number")
            return
            
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory")
            return
            
        # Create capture system
        self.capture_system = RapidCharpyCapture(
            device_id=device_id,
            output_dir=output_dir
        )
        
        # Set capture mode
        self.capture_system.capture_mode = self.capture_mode_var.get()
        
        # Start capture
        if self.capture_system.start_capture():
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.capture_btn.config(state="normal")
            self.log_status("✅ Capture started successfully!")
            self.log_status("🎥 Camera window opened - use keyboard controls there")
            
            # Start status updates
            self.update_capture_status()
        else:
            self.log_status("❌ Failed to start capture system")
            messagebox.showerror("Error", "Failed to start capture system. Check camera connection.")
            
    def stop_capture(self):
        """Stop the capture system."""
        if self.capture_system:
            self.capture_system.stop_capture()
            self.capture_system = None
            
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.capture_btn.config(state="disabled")
        self.log_status("🛑 Capture stopped")
        
        # Cancel status updates
        if self.status_update_job:
            self.root.after_cancel(self.status_update_job)
            self.status_update_job = None
            
    def manual_capture(self):
        """Trigger manual capture from GUI."""
        if self.capture_system and self.capture_system.is_running:
            if self.capture_system.manual_capture():
                self.log_status("📸 Manual capture successful!")
            else:
                self.log_status("⚠️ Manual capture failed - check image quality or similarity")
        else:
            self.log_status("⚠️ No active capture system")
            
    def reset_counters(self):
        """Reset capture counters."""
        if self.capture_system:
            self.capture_system.reset_counters()
            self.log_status("🔄 Counters reset")
        else:
            self.log_status("⚠️ No active capture system to reset")
            
    def log_status(self, message):
        """Add a status message to the log."""
        self.status_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"\n[{timestamp}] {message}")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        
    def update_capture_status(self):
        """Update capture statistics in the GUI."""
        if self.capture_system and self.capture_system.is_running:
            # Log current statistics
            stats = (f"📊 Stats: {self.capture_system.saved_frames} saved | "
                    f"{self.capture_system.skipped_similar} similar | "
                    f"{self.capture_system.skipped_quality} low quality | "
                    f"Total: {self.capture_system.total_frames}")
            
            # Only log if there's activity
            if self.capture_system.total_frames > 0:
                self.log_status(stats)
                
            # Schedule next update
            self.status_update_job = self.root.after(5000, self.update_capture_status)  # Every 5 seconds
            
    def update_status(self):
        """Initial status update."""
        self.log_status("🚀 Rapid Charpy Data Collection System Ready")
        self.log_status("📝 Configure settings above and click 'Start Capture' to begin")
        
    def on_closing(self):
        """Handle window closing."""
        if self.capture_system and self.capture_system.is_running:
            self.stop_capture()
        self.root.destroy()
        
    def run(self):
        """Run the GUI application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main function to run rapid capture with GUI."""
    print("🚀 Starting Rapid Charpy Specimen Data Collection GUI")
    
    # Create and run GUI
    app = CharpyCaptureGUI()
    app.run()


if __name__ == "__main__":
    main()