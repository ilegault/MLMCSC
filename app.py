import sys
import os
from pathlib import Path
import yaml
import logging
from threading import Thread
import webbrowser
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all components
from web.api import app as fastapi_app
from database import DatabaseManager, DataPipeline
from mlmcsc.regression.online_learning import OnlineLearningSystem
from postprocessing import CharpyLateralExpansionMeasurer
import uvicorn

class MLMCSCIntegratedApp:
    def __init__(self, config_path="config/app_config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.initialize_components()
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        # Configure logging based on config
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_dir = log_config.get('log_dir', 'logs')
        
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(log_dir) / 'app.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_components(self):
        # 1. Initialize database
        db_path = self.config['database']['path']
        # Convert path to SQLAlchemy URL format
        if self.config['database']['type'] == 'sqlite':
            db_url = f"sqlite:///{db_path}"
        else:
            db_url = db_path  # For other database types, assume full URL is provided
        self.db_manager = DatabaseManager(db_url)
        
        # 2. Initialize online learning system
        self.online_learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy=self.config['online_learning']['update_strategy']
        )
        
        # 3. Initialize Charpy measurer
        self.charpy_measurer = CharpyLateralExpansionMeasurer(
            calibration_factor=self.config['charpy']['calibration_factor']
        )
        
        # 4. Inject components into FastAPI app
        fastapi_app.state.db_manager = self.db_manager
        fastapi_app.state.online_learner = self.online_learner
        fastapi_app.state.charpy_measurer = self.charpy_measurer
        fastapi_app.state.config = self.config
    
    def start_server(self):
        # Start FastAPI server
        # Note: For multiple workers, we need to use reload=False and pass app directly
        workers = self.config['app']['workers']
        if workers > 1:
            # For multiple workers, use single worker mode to avoid import string requirement
            workers = 1
            
        uvicorn.run(
            fastapi_app,
            host=self.config['app']['host'],
            port=self.config['app']['port'],
            workers=workers,
            reload=False
        )
    
    def open_browser(self):
        # Wait for server to start then open browser
        time.sleep(2)
        webbrowser.open(f"http://{self.config['app']['host']}:{self.config['app']['port']}")
    
    def run(self):
        # Start browser in separate thread
        Thread(target=self.open_browser, daemon=True).start()
        # Start server (blocking)
        self.start_server()

if __name__ == "__main__":
    app = MLMCSCIntegratedApp()
    app.run()