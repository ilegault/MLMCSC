#!/usr/bin/env python3
"""
MLMCSC - Main Entry Point

This is the main entry point for the MLMCSC system. It provides a command-line
interface to access all major functionality.

Usage:
    python main.py live                    # Start live microscope viewer
    python main.py train detection         # Train detection model
    python main.py train classification    # Train classification model
    python main.py analyze dataset         # Analyze dataset
    python main.py config                  # Show configuration
    python main.py migrate                 # Show migration guide
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MLMCSC - Machine Learning Microscope Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py live                    # Start live viewer
  python main.py train detection         # Train detection model
  python main.py config                  # Show configuration
  python main.py migrate                 # Show migration guide
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Live viewer command
    live_parser = subparsers.add_parser('live', help='Start live microscope viewer')
    live_parser.add_argument('--yolo-model', default='models/detection/best.pt',
                           help='Path to YOLO model')
    live_parser.add_argument('--regression-model', default='models/classification/charpy_shear_regressor.pkl',
                           help='Path to regression model')
    live_parser.add_argument('--config', help='Configuration file path')
    
    # Training commands
    train_parser = subparsers.add_parser('train', help='Training commands')
    train_subparsers = train_parser.add_subparsers(dest='train_type', help='Training type')
    
    detection_parser = train_subparsers.add_parser('detection', help='Train detection model')
    detection_parser.add_argument('--data', required=True, help='Dataset path')
    detection_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    classification_parser = train_subparsers.add_parser('classification', help='Train classification model')
    classification_parser.add_argument('--data', required=True, help='Dataset path')
    classification_parser.add_argument('--model-type', choices=['shiny', 'traditional'], default='shiny',
                                     help='Model type to train')
    
    # Analysis commands
    analyze_parser = subparsers.add_parser('analyze', help='Analysis commands')
    analyze_parser.add_argument('target', choices=['dataset', 'results'], help='What to analyze')
    analyze_parser.add_argument('--path', required=True, help='Path to analyze')
    
    # Configuration command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--create-default', action='store_true',
                             help='Create default configuration file')
    
    # Migration guide command
    migrate_parser = subparsers.add_parser('migrate', help='Show migration guide')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'live':
            run_live_viewer(args)
        elif args.command == 'train':
            run_training(args)
        elif args.command == 'analyze':
            run_analysis(args)
        elif args.command == 'config':
            show_config(args)
        elif args.command == 'migrate':
            show_migration_guide()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_live_viewer(args):
    """Run the live microscope viewer."""
    print("🔬 Starting MLMCSC Live Microscope Viewer...")
    
    try:
        from apps.live_viewer import WorkingLiveMicroscopeViewer
        
        viewer = WorkingLiveMicroscopeViewer(
            yolo_model_path=args.yolo_model,
            regression_model_path=args.regression_model
        )
        
        print("Loading calibration...")
        if not viewer.load_calibration():
            print("⚠️ Failed to load calibration, using defaults")
        
        print("Loading models...")
        if not viewer.load_models():
            print("❌ Failed to load models")
            return
        
        print("Connecting to camera...")
        if not viewer.connect_camera():
            print("❌ Failed to connect to camera")
            return
        
        print("✅ Starting live view (Press 'q' to quit)")
        viewer.run()
        
    except ImportError as e:
        print(f"❌ Failed to import live viewer: {e}")
        print("Make sure all dependencies are installed.")

def run_training(args):
    """Run training commands."""
    if not args.train_type:
        print("Please specify training type: detection or classification")
        return
    
    print(f"🏋️ Starting {args.train_type} training...")
    
    if args.train_type == 'detection':
        print(f"Training detection model with data: {args.data}")
        print(f"Epochs: {args.epochs}")
        # Import and run detection training
        try:
            from apps.trainer.train_specimen_detector import main as train_detection
            train_detection()
        except ImportError:
            print("Detection training module not available")
    
    elif args.train_type == 'classification':
        print(f"Training classification model with data: {args.data}")
        print(f"Model type: {args.model_type}")
        # Import and run classification training
        try:
            from apps.trainer.main_shiny_trainer import main as train_classification
            train_classification()
        except ImportError:
            print("Classification training module not available")

def run_analysis(args):
    """Run analysis commands."""
    print(f"📊 Analyzing {args.target}: {args.path}")
    
    if args.target == 'dataset':
        try:
            from src.apps.analyzer.analyze_dataset import main as analyze_dataset
            analyze_dataset()
        except ImportError:
            print("Dataset analysis tool not available")
    
    elif args.target == 'results':
        print("Results analysis not yet implemented")

def show_config(args):
    """Show configuration information."""
    try:
        from mlmcsc.config.settings import load_config, save_default_config
        from mlmcsc.config.paths import paths
        
        if args.create_default:
            config_path = save_default_config()
            print(f"✅ Default configuration created: {config_path}")
            return
        
        print("📋 MLMCSC Configuration")
        print("=" * 50)
        print(f"Project Root: {paths.project_root}")
        print(f"Source: {paths.src}")
        print(f"Data: {paths.data}")
        print(f"Models: {paths.models}")
        print(f"Results: {paths.results}")
        print()
        
        config = load_config()
        print("Current Configuration:")
        print(f"  Camera Device: {config.camera.device_id}")
        print(f"  Resolution: {config.camera.resolution}")
        print(f"  YOLO Model: {config.model.yolo_model_path}")
        print(f"  Classification Model: {config.model.classification_model_path}")
        print(f"  Debug Mode: {config.debug}")
        
    except ImportError as e:
        print(f"Configuration module not available: {e}")

def show_migration_guide():
    """Show migration guide."""
    try:
        import compatibility
        compatibility.show_migration_guide()
    except ImportError:
        print("Migration guide not available")

if __name__ == "__main__":
    main()