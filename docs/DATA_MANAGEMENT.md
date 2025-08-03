# MLMCSC Data Management System

## Overview

The MLMCSC Data Management System provides comprehensive database management capabilities for handling growing datasets in machine learning microscopy applications. It includes automated backups, data versioning, privacy compliance, and efficient querying for training.

## Features

### 🗄️ Database Schema
- **Images Table**: Stores image metadata, paths, and quality scores
- **Predictions Table**: Model predictions with confidence scores and processing metadata
- **Labels Table**: Human annotations with technician information and verification status
- **Features Table**: Extracted feature vectors for efficient training
- **Model Versions**: Track different model iterations with performance metrics
- **Technicians**: User management with experience levels and performance tracking
- **Data Versions**: Dataset snapshots for reproducibility
- **Audit Logs**: Complete audit trail for compliance
- **System Metrics**: Performance and health monitoring

### 🔄 Data Pipeline
- **Automated Backups**: Scheduled database backups with retention policies
- **Data Versioning**: Automatic dataset versioning based on configurable triggers
- **Quality Monitoring**: Continuous data quality assessment and alerting
- **Privacy Compliance**: GDPR-compliant data handling and anonymization
- **System Metrics**: Real-time performance and health monitoring

### 🔒 Privacy & Compliance
- **GDPR Compliance**: Right to be forgotten, data export, and anonymization
- **Audit Logging**: Complete audit trail of all data operations
- **Data Retention**: Configurable retention policies for different data types
- **Anonymization**: Automatic anonymization of old personal data

### ⚡ Performance Optimization
- **Efficient Indexing**: Optimized database indexes for fast queries
- **Batch Processing**: Parallel processing for large datasets
- **Connection Pooling**: Optimized database connections
- **Query Caching**: Intelligent query result caching

## Installation

### Prerequisites
```bash
pip install sqlalchemy pandas numpy pillow psutil schedule click
```

### Database Setup
```python
from src.database import DatabaseManager, load_config

# Load configuration
config = load_config()
db_url = config.database.get_connection_url()

# Initialize database
db_manager = DatabaseManager(db_url)
```

## Quick Start

### 1. Basic Database Operations

```python
from src.database import DatabaseManager

# Initialize database
db_manager = DatabaseManager("sqlite:///mlmcsc.db")

# Store an image
image_id = db_manager.store_image(
    image_path="path/to/image.jpg",
    specimen_id="SPEC_001",
    magnification=400.0
)

# Register a technician
tech_id = db_manager.register_technician(
    technician_id="tech_001",
    name="John Doe",
    experience_level="expert"
)

# Store a prediction
pred_id = db_manager.store_prediction(
    image_id=image_id,
    model_version_id=1,
    prediction=0.85,
    confidence=0.92
)

# Store a label
label_id = db_manager.store_label(
    image_id=image_id,
    technician_id=tech_id,
    label=0.87,
    notes="High quality specimen"
)
```

### 2. Data Pipeline Setup

```python
from src.database import DataPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    backup_enabled=True,
    backup_schedule="daily",
    auto_versioning=True,
    quality_monitoring=True
)

# Start pipeline
pipeline = DataPipeline(db_manager, config)
pipeline.start()
```

### 3. Training Data Preparation

```python
# Get training data splits
training_data = db_manager.get_training_data(
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    verified_only=True
)

# Get feature vectors
image_ids = [item[0] for item in training_data['train']]
features = db_manager.get_features_for_training(
    image_ids=image_ids,
    feature_type="cnn"
)
```

## Configuration

The configuration file is located at `src/database/database_config.json`. The system will automatically look for it in this location.

### Database Configuration
```json
{
  "database": {
    "db_type": "sqlite",
    "sqlite_path": "src/database/data/mlmcsc.db",
    "pool_size": 5,
    "echo_sql": false
  }
}
```

### Backup Configuration
```json
{
  "backup": {
    "enabled": true,
    "schedule": "daily",
    "retention_days": 30,
    "compression": true
  }
}
```

### Privacy Configuration
```json
{
  "retention": {
    "image_retention_days": 730,
    "label_retention_days": 1095,
    "anonymization_enabled": true,
    "gdpr_compliance": true
  }
}
```

## Command Line Interface

The system includes a comprehensive CLI for database management:

### Database Operations
```bash
# Initialize database
python -m src.database.cli database init

# Check database status
python -m src.database.cli database status

# Validate database integrity
python -m src.database.cli database validate
```

### Backup Management
```bash
# Create backup
python -m src.database.cli backup create --name "manual_backup"

# List backups
python -m src.database.cli backup list

# Restore from backup
python -m src.database.cli backup restore backup_20240101.db
```

### Data Versioning
```bash
# Create data version
python -m src.database.cli version create "v1.0" --description "Initial dataset"

# List versions
python -m src.database.cli version list
```

### Quality Monitoring
```bash
# Run quality check
python -m src.database.cli quality check

# Update quality scores
python -m src.database.cli quality update-scores --threshold 0.7
```

### Privacy Compliance
```bash
# Export user data (GDPR)
python -m src.database.cli privacy export-user-data user_123 --output user_data.json

# Delete user data (Right to be forgotten)
python -m src.database.cli privacy delete-user-data user_123 --confirm
```

## Database Schema Details

### Images Table
```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    path VARCHAR(500) NOT NULL UNIQUE,
    filename VARCHAR(255) NOT NULL,
    timestamp DATETIME NOT NULL,
    width INTEGER,
    height INTEGER,
    channels INTEGER,
    file_size INTEGER,
    format VARCHAR(10),
    magnification FLOAT,
    specimen_id VARCHAR(100),
    acquisition_settings JSON,
    is_valid BOOLEAN DEFAULT TRUE,
    quality_score FLOAT
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    model_version_id INTEGER REFERENCES model_versions(id),
    prediction FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    detection_bbox JSON,
    detection_confidence FLOAT,
    processing_time FLOAT,
    timestamp DATETIME NOT NULL,
    raw_output JSON
);
```

### Labels Table
```sql
CREATE TABLE labels (
    id INTEGER PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    technician_id INTEGER REFERENCES technicians(id),
    label FLOAT NOT NULL,
    label_type VARCHAR(50) DEFAULT 'manual',
    original_prediction FLOAT,
    original_confidence FLOAT,
    timestamp DATETIME NOT NULL,
    time_spent FLOAT,
    difficulty_rating INTEGER,
    notes TEXT,
    is_verified BOOLEAN DEFAULT FALSE
);
```

## Performance Optimization

### Indexing Strategy
- Timestamp-based indexes for time-series queries
- Foreign key indexes for join operations
- Composite indexes for common query patterns
- Quality score indexes for filtering

### Query Optimization
- Pagination for large result sets
- Batch processing for bulk operations
- Connection pooling for concurrent access
- Query result caching

### Storage Optimization
- Feature vector compression
- Image metadata extraction
- Efficient JSON storage for flexible data
- Archive old data to reduce active database size

## Monitoring and Alerting

### System Metrics
- Database size and growth rate
- Query performance statistics
- Connection pool utilization
- Backup success/failure rates

### Data Quality Metrics
- Image quality score distribution
- Label verification rates
- Prediction accuracy trends
- Data completeness statistics

### Alerts
- Low disk space warnings
- Backup failure notifications
- Data quality degradation alerts
- System performance issues

## Security Considerations

### Access Control
- Role-based access control (RBAC)
- Session management and timeouts
- IP whitelisting capabilities
- Failed login attempt tracking

### Data Protection
- Optional data encryption at rest
- Secure backup storage
- Audit logging of sensitive operations
- Personal data anonymization

### Compliance
- GDPR compliance features
- Data retention policies
- Right to be forgotten implementation
- Complete audit trails

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check database URL configuration
   - Verify database server is running
   - Check connection pool settings

2. **Backup Failures**
   - Verify backup directory permissions
   - Check available disk space
   - Review backup configuration

3. **Performance Issues**
   - Monitor query execution times
   - Check index usage
   - Review connection pool utilization
   - Consider database optimization

4. **Data Quality Issues**
   - Run quality validation checks
   - Review data ingestion processes
   - Check for orphaned records
   - Validate file paths

### Diagnostic Commands
```bash
# Check system health
python -m src.database.cli database status

# Validate database integrity
python -m src.database.cli database validate

# Run quality assessment
python -m src.database.cli quality check

# View pipeline status
python -m src.database.cli pipeline status
```

## API Reference

### DatabaseManager Class

#### Core Methods
- `store_image(image_path, **metadata)` - Store image metadata
- `store_prediction(image_id, model_version_id, prediction, confidence)` - Store model prediction
- `store_label(image_id, technician_id, label, **metadata)` - Store human annotation
- `get_training_data(train_split, val_split, test_split)` - Get training data splits

#### Management Methods
- `backup_database(backup_name)` - Create database backup
- `create_data_version(version, name, description)` - Create data version
- `get_system_health()` - Get system health metrics
- `export_user_data(user_id)` - Export user data (GDPR)

### DataPipeline Class

#### Pipeline Methods
- `start()` - Start automated pipeline
- `stop()` - Stop pipeline
- `check_data_quality()` - Run quality assessment
- `collect_system_metrics()` - Collect system metrics
- `generate_privacy_report()` - Generate privacy compliance report

## Examples

See `examples/data_management_demo.py` for a comprehensive demonstration of all features.

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure privacy compliance for new features

## License

This project is part of the MLMCSC system. See the main project license for details.