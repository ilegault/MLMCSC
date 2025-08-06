import sys
import os
from pathlib import Path
import yaml
import logging
from threading import Thread
import webbrowser
import time
import subprocess
import platform
import socket

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import requests
except ImportError:
    print("❌ Missing required dependency: requests")
    print("📦 Install with: pip install requests")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all components with error handling
try:
    from web.api import app as fastapi_app
    from database import DatabaseManager, DataPipeline
    from mlmcsc.regression.online_learning import OnlineLearningSystem
    from postprocessing import CharpyLateralExpansionMeasurer
    import uvicorn
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("📁 Make sure you're running from the project root directory")
    print("📦 Install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

class MLMCSCIntegratedApp:
    def __init__(self, config_path="config/app_config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.initialize_components()
        self.print_startup_info()
        
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
    
    def kill_existing_servers(self):
        """Kill any existing servers running on the target port."""
        target_port = self.config['app']['port']
        target_host = self.config['app']['host']
        
        print(f"🔍 Checking for existing servers on {target_host}:{target_port}...")
        
        try:
            # Method 1: Check if port is in use
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((target_host, target_port))
            sock.close()
            
            if result == 0:
                print(f"⚠️  Port {target_port} is in use. Attempting to free it...")
                
                # Method 2: Find and kill processes using the port
                killed_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'connections']):
                    try:
                        connections = proc.info['connections']
                        if connections:
                            for conn in connections:
                                if (hasattr(conn, 'laddr') and conn.laddr and
                                    conn.laddr.port == target_port):
                                    print(f"🔪 Killing process {proc.info['pid']} ({proc.info['name']})")
                                    proc.kill()
                                    killed_processes.append(proc.info['pid'])
                                    break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue

                if killed_processes:
                    print(f"✅ Killed {len(killed_processes)} processes using port {target_port}")
                    time.sleep(2)  # Give processes time to clean up
                else:
                    # Method 3: Try Windows netstat approach
                    try:
                        result = subprocess.run(
                            ['netstat', '-ano'],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )

                        for line in result.stdout.split('\n'):
                            if f':{target_port}' in line and 'LISTENING' in line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    pid = parts[-1]
                                    try:
                                        subprocess.run(['taskkill', '/F', '/PID', pid],
                                                     capture_output=True, timeout=5)
                                        print(f"✅ Killed process {pid} using port {target_port}")
                                        time.sleep(1)
                                    except:
                                        pass
                    except:
                        pass
            else:
                print(f"✅ Port {target_port} is available")

        except Exception as e:
            print(f"⚠️  Error checking port: {e}")
            # Try to kill any Python processes that might be servers
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if (proc.info['name'] == 'python.exe' and
                            proc.info['cmdline'] and
                            any('app.py' in arg or 'uvicorn' in arg or 'fastapi' in arg
                                for arg in proc.info['cmdline'])):
                            print(f"🔪 Killing Python server process {proc.info['pid']}")
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as cleanup_error:
                print(f"⚠️  Error during cleanup: {cleanup_error}")

    def initialize_components(self):
        # Kill any existing servers first
        self.kill_existing_servers()
        
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
    
    def print_startup_info(self):
        """Print comprehensive startup information."""
        print("=" * 60)
        print("MLMCSC Human-in-the-Loop Interface with Live Microscope")
        print("=" * 60)
        print()
        print("Features included:")
        print("✓ Image upload and analysis")
        print("✓ Technician labeling interface")
        print("✓ Model performance metrics")
        print("✓ Labeling history and export")
        print("✓ Online learning system")
        print("✓ Live microscope video streaming")
        print("✓ Real-time predictions")
        print("✓ Frame capture and analysis")
        print("✓ Auto prediction with configurable intervals")
        print()
        print(f"Server will be available at: http://{self.config['app']['host']}:{self.config['app']['port']}")
        print(f"API documentation at: http://{self.config['app']['host']}:{self.config['app']['port']}/docs")
        print()
        print("Live Microscope Endpoints:")
        print("• GET  /video_feed - Live video stream")
        print("• POST /camera/start - Start camera")
        print("• POST /camera/stop - Stop camera")
        print("• GET  /camera/status - Camera status")
        print("• POST /camera/capture - Capture frame")
        print("• POST /camera/predict_live - Live prediction")
        print("• GET  /camera/detect - Detect available cameras")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        print()
    
    def start_server(self):
        # Start FastAPI server with live microscope functionality
        # Note: For multiple workers, we need to use reload=False and pass app directly
        workers = self.config['app']['workers']
        if workers > 1:
            # For multiple workers, use single worker mode to avoid import string requirement
            workers = 1
            
        logging.info(f"Starting MLMCSC server with live microscope on {self.config['app']['host']}:{self.config['app']['port']}")
        uvicorn.run(
            fastapi_app,
            host=self.config['app']['host'],
            port=self.config['app']['port'],
            workers=workers,
            reload=False,
            log_level="info"
        )
    
    def check_server_running(self):
        """Check if server is already running"""
        try:
            url = f"http://{self.config['app']['host']}:{self.config['app']['port']}/health"
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def wait_for_server_ready(self, max_attempts=30, delay=1):
        """Wait for server to be ready before opening browser"""
        url = f"http://{self.config['app']['host']}:{self.config['app']['port']}/health"
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    logging.info("Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(delay)
            logging.info(f"Waiting for server... (attempt {attempt + 1}/{max_attempts})")
        
        logging.warning("Server readiness check timed out")
        return False
    
    def open_browser(self):
        # Wait for server to be ready then open browser
        if self.wait_for_server_ready():
            webbrowser.open(f"http://{self.config['app']['host']}:{self.config['app']['port']}")
            logging.info("Browser opened successfully")
        else:
            logging.error("Failed to open browser - server not ready")
    
    def run(self):
        # Start browser in separate thread only if not running in server-only mode
        if "--server-only" not in sys.argv:
            Thread(target=self.open_browser, daemon=True).start()
        # Start server (blocking)
        self.start_server()
    
    def run_in_powershell(self):
        """Launch the application in a new PowerShell window with better management."""
        
        script_path = Path(__file__).parent / "start_server_simple.ps1"
        
        try:
            # Launch PowerShell with the script
            if platform.system() == "Windows":
                print("🚀 Launching MLMCSC server in PowerShell...")
                print(f"📁 Script location: {script_path}")
                print("💡 The server will start in a new PowerShell window")
                print("🌐 Browser will open automatically once server is ready")
                print("⚠️  Keep the PowerShell window open to keep the server running")
                
                # Use Popen to start PowerShell without blocking, with NoExit to keep window open
                subprocess.Popen([
                    "powershell.exe", 
                    "-ExecutionPolicy", "Bypass",
                    "-NoExit",
                    "-File", str(script_path)
                ], creationflags=subprocess.CREATE_NEW_CONSOLE)
                
                print("✅ PowerShell window launched successfully!")
                print("📝 Check the PowerShell window for server status and controls")
                
            else:
                print("PowerShell management is only available on Windows.")
                print("Starting server normally...")
                self.run()
        except Exception as e:
            print(f"❌ Error launching PowerShell script: {e}")
            print("🔄 Starting server normally...")
            self.run()

if __name__ == "__main__":
    # Check if we should run in server-only mode (called from PowerShell)
    if "--server-only" in sys.argv:
        app = MLMCSCIntegratedApp()
        app.run()
    else:
        # Launch in PowerShell management mode
        app = MLMCSCIntegratedApp()
        
        # Check if server is already running
        if app.check_server_running():
            print("⚠️  Server is already running!")
            print(f"🌐 Access it at: http://{app.config['app']['host']}:{app.config['app']['port']}")
            
            # Ask user what to do
            choice = input("Do you want to:\n1. Open browser to existing server\n2. Stop and restart server\n3. Exit\nChoice (1-3): ").strip()
            
            if choice == "1":
                webbrowser.open(f"http://{app.config['app']['host']}:{app.config['app']['port']}")
                print("✅ Browser opened to existing server")
            elif choice == "2":
                print("🔄 Restarting server in PowerShell...")
                app.run_in_powershell()
            else:
                print("👋 Exiting...")
        else:
            # Give user options for how to run
            print("🚀 MLMCSC Server Launcher with Live Microscope")
            print("=" * 50)
            print("How would you like to run the server?")
            print("1. PowerShell window (simple)")
            print("2. PowerShell window (with management)")
            print("3. Current terminal")
            print()
            
            choice = input("Choose option (1-3) [1]: ").strip() or "1"
            
            if choice == "1":
                # Simple PowerShell launch
                script_path = Path(__file__).parent / "start_server_simple.ps1"
                subprocess.Popen([
                    "powershell.exe", 
                    "-ExecutionPolicy", "Bypass",
                    "-NoExit",
                    "-File", str(script_path)
                ], creationflags=subprocess.CREATE_NEW_CONSOLE)
                print("✅ Server launched in PowerShell window!")
                
            elif choice == "2":
                # Full management PowerShell
                app.run_in_powershell()
                
            else:
                # Run in current terminal
                print("🚀 Starting server in current terminal...")
                app.run()