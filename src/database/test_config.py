#!/usr/bin/env python3
"""
Test script to verify database configuration loading.
This script tests the configuration system without requiring SQLAlchemy.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Test configuration loading functionality."""
    try:
        from src.database.config import load_config, DataManagementConfig
        
        print("🧪 Testing Configuration System")
        print("=" * 50)
        
        # Test loading default configuration
        config = load_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Database type: {config.database.db_type.value}")
        print(f"   SQLite path: {config.database.sqlite_path}")
        print(f"   Backup enabled: {config.backup.enabled}")
        print(f"   GDPR compliance: {config.retention.gdpr_compliance}")
        
        # Test configuration validation
        issues = config.validate()
        if issues:
            print(f"⚠️  Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"✅ Configuration validation passed")
        
        # Test database URL generation
        db_url = config.database.get_connection_url(project_root)
        print(f"✅ Database URL: {db_url}")
        
        # Test configuration file path detection
        config_file_path = Path(__file__).parent / "database_config.json"
        if config_file_path.exists():
            print(f"✅ Configuration file found: {config_file_path}")
        else:
            print(f"❌ Configuration file not found: {config_file_path}")
        
        print("\n📋 Configuration Summary:")
        print(f"   Database: {config.database.db_type.value}")
        print(f"   Backup Schedule: {config.backup.schedule.value}")
        print(f"   Quality Monitoring: {config.quality.enabled}")
        print(f"   Privacy Compliance: {config.retention.gdpr_compliance}")
        print(f"   Security: {config.security.require_authentication}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test that required directories can be created."""
    print("\n🗂️  Testing Directory Structure")
    print("=" * 50)
    
    try:
        # Test database data directory
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        print(f"✅ Data directory: {data_dir}")
        
        # Test backup directory
        backup_dir = Path(__file__).parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        print(f"✅ Backup directory: {backup_dir}")
        
        # Test archive directory
        archive_dir = Path(__file__).parent / "archive"
        archive_dir.mkdir(exist_ok=True)
        print(f"✅ Archive directory: {archive_dir}")
        
        # Test logs directory
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        print(f"✅ Logs directory: {logs_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

def main():
    """Run configuration tests."""
    print("🚀 MLMCSC Database Configuration Test")
    print("=" * 60)
    
    success = True
    
    # Test configuration loading
    if not test_config_loading():
        success = False
    
    # Test directory structure
    if not test_directory_structure():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL CONFIGURATION TESTS PASSED!")
        print("\n🎯 Next steps:")
        print("   1. Install required dependencies: pip install sqlalchemy pandas numpy")
        print("   2. Run the full demo: python examples/data_management_demo.py")
        print("   3. Use the CLI: python -m src.database.cli --help")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Please check the error messages above and fix any issues.")
    
    print("=" * 60)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())