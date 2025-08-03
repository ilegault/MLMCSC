#!/usr/bin/env python3
"""
Startup script for MLMCSC Human-in-the-Loop Web Interface

This script initializes and runs the web server with proper configuration
and error handling.
"""

import sys
import logging
import argparse
from pathlib import Path
import uvicorn
from logging.config import dictConfig

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.web.config import get_config, LOGGING_CONFIG
from src.web.database import DatabaseManager
from src.web.online_learning import OnlineLearningManager

# Configure logging
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment for the web interface."""
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Loaded configuration: {config}")
        
        # Initialize database
        db_manager = DatabaseManager(config.database_path)
        logger.info("Database initialized successfully")
        
        # Initialize online learning if enabled
        if config.online_learning_enabled:
            online_manager = OnlineLearningManager(
                db_manager=db_manager,
                model_save_path=config.online_model_path,
                update_threshold=config.update_threshold,
                update_interval=config.update_interval
            )
            online_manager.start_online_learning()
            logger.info("Online learning started")
        
        return config
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MLMCSC Human-in-the-Loop Web Interface")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--config", type=Path, help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        # Set up environment
        config = setup_environment()
        
        # Override config with command line arguments
        if args.host:
            config.host = args.host
        if args.port:
            config.port = args.port
        if args.debug:
            config.debug = True
        if args.no_reload:
            config.reload = False
        
        logger.info(f"Starting web server on {config.host}:{config.port}")
        logger.info(f"Debug mode: {config.debug}")
        logger.info(f"Auto-reload: {config.reload}")
        
        # Start the server
        uvicorn.run(
            "src.web.api:app",
            host=config.host,
            port=config.port,
            reload=config.reload,
            log_level="info" if not config.debug else "debug",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()