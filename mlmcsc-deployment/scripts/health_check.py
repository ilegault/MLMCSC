#!/usr/bin/env python3
"""
MLMCSC System Health Check Script
This script verifies that all system components are working correctly.
"""

import os
import sys
import time
import json
import requests
import psycopg2
import redis
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthChecker:
    """System health checker for MLMCSC"""
    
    def __init__(self):
        self.results = {}
        self.load_environment()
        
    def load_environment(self):
        """Load environment variables from .env file"""
        env_file = Path(__file__).parent.parent.parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Set defaults if not present
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://mlmcsc:mlmcsc123@localhost:5432/mlmcsc')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.api_url = os.getenv('API_URL', 'http://localhost:8000')
        
    def check_database(self) -> Tuple[bool, str]:
        """Check PostgreSQL database connection"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # Check if tables exist (basic structure check)
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return True, f"Connected successfully. Version: {version[:50]}... Tables: {len(tables)}"
            
        except Exception as e:
            return False, f"Database connection failed: {str(e)}"
    
    def check_redis(self) -> Tuple[bool, str]:
        """Check Redis connection"""
        try:
            r = redis.from_url(self.redis_url)
            
            # Test basic operations
            r.ping()
            r.set('health_check', 'test', ex=10)
            value = r.get('health_check')
            r.delete('health_check')
            
            info = r.info()
            version = info.get('redis_version', 'unknown')
            memory = info.get('used_memory_human', 'unknown')
            
            return True, f"Connected successfully. Version: {version}, Memory: {memory}"
            
        except Exception as e:
            return False, f"Redis connection failed: {str(e)}"
    
    def check_api_server(self) -> Tuple[bool, str]:
        """Check API server health"""
        try:
            # Check if server is responding
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                return True, f"API server healthy. Status: {health_data.get('status', 'unknown')}"
            else:
                return False, f"API server returned status code: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "API server is not responding (connection refused)"
        except requests.exceptions.Timeout:
            return False, "API server timeout"
        except Exception as e:
            return False, f"API server check failed: {str(e)}"
    
    def check_api_endpoints(self) -> Tuple[bool, str]:
        """Check critical API endpoints"""
        endpoints = [
            "/",
            "/docs",
            "/openapi.json"
        ]
        
        results = []
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.api_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    results.append(f"{endpoint}: OK")
                else:
                    results.append(f"{endpoint}: {response.status_code}")
            except Exception as e:
                results.append(f"{endpoint}: ERROR - {str(e)[:50]}")
        
        success = all("OK" in result for result in results)
        return success, "; ".join(results)
    
    def check_file_system(self) -> Tuple[bool, str]:
        """Check file system and required directories"""
        base_path = Path(__file__).parent.parent.parent
        required_dirs = [
            'data',
            'data/images',
            'data/models',
            'data/temp',
            'logs',
            'models',
            'models/cache'
        ]
        
        issues = []
        for dir_path in required_dirs:
            full_path = base_path / dir_path
            if not full_path.exists():
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    issues.append(f"Created missing directory: {dir_path}")
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_path}: {str(e)}")
            elif not full_path.is_dir():
                issues.append(f"Path exists but is not a directory: {dir_path}")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(base_path)
            free_gb = free // (1024**3)
            if free_gb < 1:
                issues.append(f"Low disk space: {free_gb}GB free")
        except Exception as e:
            issues.append(f"Cannot check disk space: {str(e)}")
        
        success = len(issues) == 0 or all("Created missing" in issue for issue in issues)
        message = "; ".join(issues) if issues else "All directories exist and accessible"
        
        return success, message
    
    def check_python_environment(self) -> Tuple[bool, str]:
        """Check Python environment and dependencies"""
        issues = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(f"Python version too old: {python_version.major}.{python_version.minor}")
        
        # Check critical packages
        critical_packages = [
            'fastapi',
            'uvicorn',
            'celery',
            'redis',
            'psycopg2',
            'sqlalchemy',
            'pydantic',
            'numpy',
            'pillow'
        ]
        
        missing_packages = []
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            issues.append(f"Missing packages: {', '.join(missing_packages)}")
        
        success = len(issues) == 0
        message = "; ".join(issues) if issues else f"Python {python_version.major}.{python_version.minor}.{python_version.micro}, all packages available"
        
        return success, message
    
    def check_celery_worker(self) -> Tuple[bool, str]:
        """Check if Celery worker is running"""
        try:
            # Try to connect to Celery broker
            from celery import Celery
            
            broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
            app = Celery('health_check', broker=broker_url)
            
            # Get active workers
            inspect = app.control.inspect()
            active_workers = inspect.active()
            
            if active_workers:
                worker_count = len(active_workers)
                return True, f"Celery workers active: {worker_count}"
            else:
                return False, "No active Celery workers found"
                
        except Exception as e:
            return False, f"Celery worker check failed: {str(e)}"
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        checks = [
            ("Python Environment", self.check_python_environment),
            ("File System", self.check_file_system),
            ("Database", self.check_database),
            ("Redis", self.check_redis),
            ("API Server", self.check_api_server),
            ("API Endpoints", self.check_api_endpoints),
            ("Celery Worker", self.check_celery_worker),
        ]
        
        results = {}
        overall_success = True
        
        print("MLMCSC System Health Check")
        print("=" * 50)
        
        for check_name, check_func in checks:
            print(f"\nChecking {check_name}...")
            try:
                success, message = check_func()
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"{status}: {message}")
                
                results[check_name] = {
                    "success": success,
                    "message": message,
                    "timestamp": time.time()
                }
                
                if not success:
                    overall_success = False
                    
            except Exception as e:
                error_msg = f"Check failed with exception: {str(e)}"
                print(f"✗ ERROR: {error_msg}")
                results[check_name] = {
                    "success": False,
                    "message": error_msg,
                    "timestamp": time.time()
                }
                overall_success = False
        
        # Summary
        print("\n" + "=" * 50)
        if overall_success:
            print("✓ ALL CHECKS PASSED - System is healthy!")
        else:
            print("✗ SOME CHECKS FAILED - System needs attention!")
            failed_checks = [name for name, result in results.items() if not result["success"]]
            print(f"Failed checks: {', '.join(failed_checks)}")
        
        results["overall_success"] = overall_success
        results["timestamp"] = time.time()
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save health check results to file"""
        try:
            log_dir = Path(__file__).parent.parent.parent / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            results_file = log_dir / 'health_check_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nHealth check results saved to: {results_file}")
            
        except Exception as e:
            print(f"Warning: Could not save results to file: {str(e)}")

def main():
    """Main function"""
    checker = HealthChecker()
    results = checker.run_all_checks()
    checker.save_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    main()