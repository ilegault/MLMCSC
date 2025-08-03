#!/usr/bin/env python3
"""
Database Management CLI for MLMCSC

Command-line interface for managing the MLMCSC database system including
backups, data versioning, quality checks, and system maintenance.
"""

import click
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.manager import DatabaseManager
from src.database.pipeline import DataPipeline, PipelineConfig
from src.database.config import DataManagementConfig, load_config
from src.database.models import Image, Label, Prediction, Technician

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """MLMCSC Database Management CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if config:
        ctx.obj = {'config': DataManagementConfig.from_file(config)}
    else:
        ctx.obj = {'config': load_config()}
    
    # Initialize database manager
    db_config = ctx.obj['config'].database
    db_url = db_config.get_connection_url(project_root)
    ctx.obj['db_manager'] = DatabaseManager(db_url)


@cli.group()
def database():
    """Database operations"""
    pass


@database.command()
@click.pass_context
def init(ctx):
    """Initialize database tables"""
    try:
        db_manager = ctx.obj['db_manager']
        db_manager.init_database()
        click.echo("✅ Database initialized successfully")
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}")
        sys.exit(1)


@database.command()
@click.pass_context
def status(ctx):
    """Show database status"""
    try:
        db_manager = ctx.obj['db_manager']
        health = db_manager.get_system_health()
        
        click.echo("📊 Database Status")
        click.echo("=" * 50)
        
        for category, metrics in health.items():
            if isinstance(metrics, dict):
                click.echo(f"\n{category.replace('_', ' ').title()}:")
                for key, value in metrics.items():
                    click.echo(f"  {key}: {value}")
            else:
                click.echo(f"{category}: {metrics}")
                
    except Exception as e:
        click.echo(f"❌ Failed to get database status: {e}")
        sys.exit(1)


@database.command()
@click.option('--table', help='Specific table to check')
@click.pass_context
def validate(ctx, table):
    """Validate database integrity"""
    try:
        db_manager = ctx.obj['db_manager']
        
        with db_manager.get_session() as session:
            issues = []
            
            # Check for orphaned records
            if not table or table == 'predictions':
                orphaned_predictions = session.query(Prediction).outerjoin(Image).filter(
                    Image.id.is_(None)
                ).count()
                if orphaned_predictions > 0:
                    issues.append(f"Found {orphaned_predictions} orphaned predictions")
            
            if not table or table == 'labels':
                orphaned_labels = session.query(Label).outerjoin(Image).filter(
                    Image.id.is_(None)
                ).count()
                if orphaned_labels > 0:
                    issues.append(f"Found {orphaned_labels} orphaned labels")
            
            # Check for missing files
            if not table or table == 'images':
                missing_files = 0
                for image in session.query(Image).all():
                    if not Path(image.path).exists():
                        missing_files += 1
                if missing_files > 0:
                    issues.append(f"Found {missing_files} images with missing files")
            
            if issues:
                click.echo("⚠️  Database validation issues found:")
                for issue in issues:
                    click.echo(f"  - {issue}")
            else:
                click.echo("✅ Database validation passed")
                
    except Exception as e:
        click.echo(f"❌ Database validation failed: {e}")
        sys.exit(1)


@cli.group()
def backup():
    """Backup operations"""
    pass


@backup.command()
@click.option('--name', help='Backup name (default: timestamp)')
@click.pass_context
def create(ctx, name):
    """Create database backup"""
    try:
        db_manager = ctx.obj['db_manager']
        backup_path = db_manager.backup_database(name)
        
        file_size = backup_path.stat().st_size
        click.echo(f"✅ Backup created: {backup_path}")
        click.echo(f"   Size: {file_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        click.echo(f"❌ Backup creation failed: {e}")
        sys.exit(1)


@backup.command()
@click.pass_context
def list(ctx):
    """List available backups"""
    try:
        db_manager = ctx.obj['db_manager']
        backup_dir = db_manager.backup_dir
        
        if not backup_dir.exists():
            click.echo("No backup directory found")
            return
        
        backups = list(backup_dir.glob("*.db"))
        
        if not backups:
            click.echo("No backups found")
            return
        
        click.echo("📁 Available Backups")
        click.echo("=" * 50)
        
        for backup in sorted(backups, key=lambda f: f.stat().st_mtime, reverse=True):
            size = backup.stat().st_size / 1024 / 1024
            modified = datetime.fromtimestamp(backup.stat().st_mtime)
            click.echo(f"{backup.name:<30} {size:>8.2f} MB  {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            
    except Exception as e:
        click.echo(f"❌ Failed to list backups: {e}")
        sys.exit(1)


@backup.command()
@click.argument('backup_name')
@click.option('--force', is_flag=True, help='Force restore without confirmation')
@click.pass_context
def restore(ctx, backup_name, force):
    """Restore from backup"""
    try:
        db_manager = ctx.obj['db_manager']
        backup_path = db_manager.backup_dir / backup_name
        
        if not backup_path.exists():
            click.echo(f"❌ Backup not found: {backup_name}")
            sys.exit(1)
        
        if not force:
            click.confirm(f"Are you sure you want to restore from {backup_name}? This will overwrite the current database.", abort=True)
        
        # For SQLite, copy backup over current database
        if "sqlite" in db_manager.db_url:
            import shutil
            db_path = db_manager.db_url.replace("sqlite:///", "")
            shutil.copy2(backup_path, db_path)
            click.echo(f"✅ Database restored from {backup_name}")
        else:
            click.echo("❌ Restore not implemented for non-SQLite databases")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Restore failed: {e}")
        sys.exit(1)


@cli.group()
def version():
    """Data versioning operations"""
    pass


@version.command()
@click.argument('version_name')
@click.option('--description', help='Version description')
@click.pass_context
def create(ctx, version_name, description):
    """Create new data version"""
    try:
        db_manager = ctx.obj['db_manager']
        version_id = db_manager.create_data_version(
            version=version_name,
            name=version_name,
            description=description or f"Version created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            user_id="cli"
        )
        
        click.echo(f"✅ Data version created: {version_name} (ID: {version_id})")
        
    except Exception as e:
        click.echo(f"❌ Version creation failed: {e}")
        sys.exit(1)


@version.command()
@click.pass_context
def list(ctx):
    """List data versions"""
    try:
        db_manager = ctx.obj['db_manager']
        
        with db_manager.get_session() as session:
            from src.database.models import DataVersion
            versions = session.query(DataVersion).order_by(DataVersion.created_at.desc()).all()
            
            if not versions:
                click.echo("No data versions found")
                return
            
            click.echo("📋 Data Versions")
            click.echo("=" * 80)
            
            for version in versions:
                click.echo(f"Version: {version.version}")
                click.echo(f"  Name: {version.name}")
                click.echo(f"  Created: {version.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                click.echo(f"  Images: {version.total_images}, Labels: {version.total_labels}")
                if version.description:
                    click.echo(f"  Description: {version.description}")
                click.echo()
                
    except Exception as e:
        click.echo(f"❌ Failed to list versions: {e}")
        sys.exit(1)


@cli.group()
def quality():
    """Data quality operations"""
    pass


@quality.command()
@click.pass_context
def check(ctx):
    """Run data quality check"""
    try:
        db_manager = ctx.obj['db_manager']
        pipeline = DataPipeline(db_manager)
        
        report = pipeline.check_data_quality()
        
        click.echo("🔍 Data Quality Report")
        click.echo("=" * 50)
        
        # Image quality
        img_quality = report['image_quality']
        click.echo(f"\nImage Quality:")
        click.echo(f"  Total images: {img_quality['total_images']}")
        click.echo(f"  Low quality: {img_quality['low_quality_count']} ({img_quality['low_quality_percent']:.1f}%)")
        
        # Label quality
        label_quality = report['label_quality']
        click.echo(f"\nLabel Quality:")
        click.echo(f"  Total labels: {label_quality['total_labels']}")
        click.echo(f"  Verified: {label_quality['verified_labels']} ({label_quality['verification_rate']:.1f}%)")
        
        # Prediction accuracy
        pred_accuracy = report['prediction_accuracy']
        if pred_accuracy['sample_count'] > 0:
            click.echo(f"\nPrediction Accuracy (last 7 days):")
            click.echo(f"  Samples: {pred_accuracy['sample_count']}")
            click.echo(f"  MAE: {pred_accuracy['mae']:.3f}")
            click.echo(f"  RMSE: {pred_accuracy['rmse']:.3f}")
            click.echo(f"  R²: {pred_accuracy['r2']:.3f}")
        
        # Data completeness
        completeness = report['data_completeness']
        click.echo(f"\nData Completeness:")
        click.echo(f"  Images without labels: {completeness['images_without_labels']}")
        click.echo(f"  Completeness rate: {completeness['completeness_rate']:.1f}%")
        
    except Exception as e:
        click.echo(f"❌ Quality check failed: {e}")
        sys.exit(1)


@quality.command()
@click.option('--threshold', default=0.7, help='Quality threshold (0-1)')
@click.pass_context
def update_scores(ctx, threshold):
    """Update image quality scores"""
    try:
        db_manager = ctx.obj['db_manager']
        
        with db_manager.get_session() as session:
            # This is a placeholder - in practice, you'd implement actual quality assessment
            images_updated = session.query(Image).filter(
                Image.quality_score.is_(None)
            ).update({"quality_score": 0.8})  # Default score
            
            session.commit()
            
        click.echo(f"✅ Updated quality scores for {images_updated} images")
        
    except Exception as e:
        click.echo(f"❌ Quality score update failed: {e}")
        sys.exit(1)


@cli.group()
def pipeline():
    """Data pipeline operations"""
    pass


@pipeline.command()
@click.option('--config-file', help='Pipeline configuration file')
@click.pass_context
def start(ctx, config_file):
    """Start data pipeline"""
    try:
        db_manager = ctx.obj['db_manager']
        
        if config_file:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            config = PipelineConfig(**config_data)
        else:
            config = PipelineConfig()
        
        pipeline = DataPipeline(db_manager, config)
        pipeline.start()
        
        click.echo("✅ Data pipeline started")
        click.echo("Press Ctrl+C to stop...")
        
        try:
            while pipeline.is_running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            pipeline.stop()
            click.echo("\n🛑 Data pipeline stopped")
            
    except Exception as e:
        click.echo(f"❌ Pipeline start failed: {e}")
        sys.exit(1)


@pipeline.command()
@click.pass_context
def status(ctx):
    """Show pipeline status"""
    try:
        db_manager = ctx.obj['db_manager']
        pipeline = DataPipeline(db_manager)
        
        status = pipeline.get_pipeline_status()
        
        click.echo("⚙️  Pipeline Status")
        click.echo("=" * 50)
        click.echo(f"Running: {status['is_running']}")
        
        if status['last_backup']:
            backup_info = status['last_backup']
            click.echo(f"Last backup: {backup_info['created_at']}")
            click.echo(f"Backup size: {backup_info['size_bytes'] / 1024 / 1024:.2f} MB")
        
        if status['next_scheduled_tasks']:
            click.echo("\nScheduled tasks:")
            for task in status['next_scheduled_tasks']:
                click.echo(f"  - {task['task']}: {task['next_run']}")
        
    except Exception as e:
        click.echo(f"❌ Failed to get pipeline status: {e}")
        sys.exit(1)


@cli.group()
def privacy():
    """Privacy and compliance operations"""
    pass


@privacy.command()
@click.pass_context
def report(ctx):
    """Generate privacy compliance report"""
    try:
        db_manager = ctx.obj['db_manager']
        pipeline = DataPipeline(db_manager)
        
        report = pipeline.generate_privacy_report()
        
        click.echo("🔒 Privacy Compliance Report")
        click.echo("=" * 50)
        
        retention = report['data_retention']
        click.echo(f"\nData Retention (>{retention['retention_days']} days):")
        click.echo(f"  Old images: {retention['old_images']}")
        click.echo(f"  Old labels: {retention['old_labels']}")
        click.echo(f"  Old predictions: {retention['old_predictions']}")
        
        anonymization = report['anonymization']
        click.echo(f"\nAnonymization:")
        click.echo(f"  Anonymized audit logs: {anonymization['anonymized_audit_logs']}")
        click.echo(f"  Anonymization enabled: {anonymization['anonymization_enabled']}")
        
        compliance = report['compliance_status']
        click.echo(f"\nCompliance Status:")
        for key, value in compliance.items():
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Privacy report failed: {e}")
        sys.exit(1)


@privacy.command()
@click.argument('user_id')
@click.option('--output', help='Output file for exported data')
@click.pass_context
def export_user_data(ctx, user_id, output):
    """Export user data (GDPR compliance)"""
    try:
        db_manager = ctx.obj['db_manager']
        user_data = db_manager.export_user_data(user_id)
        
        if output:
            with open(output, 'w') as f:
                json.dump(user_data, f, indent=2)
            click.echo(f"✅ User data exported to {output}")
        else:
            click.echo(json.dumps(user_data, indent=2))
        
    except Exception as e:
        click.echo(f"❌ User data export failed: {e}")
        sys.exit(1)


@privacy.command()
@click.argument('user_id')
@click.option('--confirm', is_flag=True, help='Confirm deletion without prompt')
@click.pass_context
def delete_user_data(ctx, user_id, confirm):
    """Delete user data (GDPR right to be forgotten)"""
    try:
        if not confirm:
            click.confirm(f"Are you sure you want to delete all data for user {user_id}? This cannot be undone.", abort=True)
        
        db_manager = ctx.obj['db_manager']
        db_manager.delete_user_data(user_id, "cli_admin")
        
        click.echo(f"✅ User data deleted for {user_id}")
        
    except Exception as e:
        click.echo(f"❌ User data deletion failed: {e}")
        sys.exit(1)


@cli.group()
def technician():
    """Technician management"""
    pass


@technician.command()
@click.argument('technician_id')
@click.argument('name')
@click.option('--email', help='Email address')
@click.option('--department', help='Department')
@click.option('--experience', default='intermediate', help='Experience level')
@click.pass_context
def register(ctx, technician_id, name, email, department, experience):
    """Register new technician"""
    try:
        db_manager = ctx.obj['db_manager']
        tech_id = db_manager.register_technician(
            technician_id=technician_id,
            name=name,
            email=email,
            department=department,
            experience_level=experience,
            user_id="cli"
        )
        
        click.echo(f"✅ Technician registered: {name} (ID: {tech_id})")
        
    except Exception as e:
        click.echo(f"❌ Technician registration failed: {e}")
        sys.exit(1)


@technician.command()
@click.pass_context
def list(ctx):
    """List all technicians"""
    try:
        db_manager = ctx.obj['db_manager']
        
        with db_manager.get_session() as session:
            technicians = session.query(Technician).all()
            
            if not technicians:
                click.echo("No technicians found")
                return
            
            click.echo("👥 Technicians")
            click.echo("=" * 80)
            
            for tech in technicians:
                status = "Active" if tech.is_active else "Inactive"
                click.echo(f"{tech.technician_id:<15} {tech.name:<25} {tech.experience_level:<12} {status}")
                if tech.email:
                    click.echo(f"{'':15} Email: {tech.email}")
                if tech.department:
                    click.echo(f"{'':15} Department: {tech.department}")
                click.echo(f"{'':15} Labels: {tech.total_labels or 0}")
                click.echo()
                
    except Exception as e:
        click.echo(f"❌ Failed to list technicians: {e}")
        sys.exit(1)


@cli.group()
def config():
    """Configuration management"""
    pass


@config.command()
@click.argument('output_file')
def generate(output_file):
    """Generate default configuration file"""
    try:
        config = DataManagementConfig()
        config.save_to_file(output_file)
        click.echo(f"✅ Default configuration saved to {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Configuration generation failed: {e}")
        sys.exit(1)


@config.command()
@click.argument('config_file')
def validate(config_file):
    """Validate configuration file"""
    try:
        config = DataManagementConfig.from_file(config_file)
        issues = config.validate()
        
        if issues:
            click.echo("⚠️  Configuration validation issues:")
            for issue in issues:
                click.echo(f"  - {issue}")
        else:
            click.echo("✅ Configuration is valid")
            
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()