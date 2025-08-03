# MLMCSC Deployment Guide

This directory contains all the necessary files and scripts for deploying the MLMCSC (Machine Learning Microscope Control System) in various environments.

## Directory Structure

```
mlmcsc-deployment/
├── config/                     # Configuration files
│   ├── app_config.yaml         # Main application configuration
│   ├── logging_config.yaml     # Logging configuration
│   └── model_paths.yaml        # ML model file locations
├── scripts/                    # Deployment and management scripts
│   ├── setup_environment.sh    # Linux environment setup
│   ├── setup_environment.bat   # Windows environment setup
│   ├── start_app.sh           # Linux application launcher
│   ├── start_app.bat          # Windows application launcher
│   └── health_check.py        # System health verification
├── docker/                    # Docker deployment files
│   ├── Dockerfile             # Single container deployment
│   ├── docker-compose.yml     # Multi-container deployment
│   └── .dockerignore         # Docker ignore file
├── desktop/                   # Desktop integration files
│   ├── MLMCSC.desktop        # Linux desktop entry
│   └── MLMCSC.exe.manifest   # Windows application manifest
└── README_DEPLOYMENT.md       # This file
```

## Quick Start

### Windows Deployment

1. **Run the setup script** (as Administrator recommended):
   ```cmd
   cd mlmcsc-deployment\scripts
   setup_environment.bat
   ```

2. **Start the application**:
   ```cmd
   start_app.bat
   ```

3. **Access the application**:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Linux Deployment

1. **Run the setup script**:
   ```bash
   cd mlmcsc-deployment/scripts
   chmod +x *.sh
   ./setup_environment.sh
   ```

2. **Start the application**:
   ```bash
   ./start_app.sh
   ```

3. **Access the application**:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Docker Deployment

1. **Single container** (for development):
   ```bash
   cd mlmcsc-deployment/docker
   docker build -t mlmcsc:latest ../..
   docker run -p 8000:8000 mlmcsc:latest
   ```

2. **Multi-container** (for production):
   ```bash
   cd mlmcsc-deployment/docker
   docker-compose up -d
   ```

## Configuration

### Application Configuration

Edit `config/app_config.yaml` to customize:
- Database connection settings
- Redis configuration
- Storage paths
- Security settings
- Performance parameters

### Environment Variables

Create a `.env` file in the project root with:
```env
DATABASE_URL=postgresql://mlmcsc:password@localhost:5432/mlmcsc
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
SECRET_KEY=your-secret-key-here
DEBUG=false
ENVIRONMENT=production
```

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 or Linux (Ubuntu 18.04+, CentOS 7+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **CPU**: 2 cores minimum, 4 cores recommended

### Dependencies
- PostgreSQL 12+
- Redis 6+
- Python packages (installed automatically)

## Services

The MLMCSC system consists of several services:

### Core Services
1. **API Server** (FastAPI)
   - Port: 8000
   - Handles web interface and REST API
   - Auto-starts with the application

2. **Celery Worker**
   - Processes background tasks
   - Handles ML model training and inference
   - Auto-starts with the application

### External Services
1. **PostgreSQL Database**
   - Stores application data and metadata
   - Must be running before starting MLMCSC

2. **Redis Cache/Broker**
   - Caches data and handles message queuing
   - Must be running before starting MLMCSC

## Health Monitoring

Run the health check script to verify system status:

```bash
# Linux
python mlmcsc-deployment/scripts/health_check.py

# Windows
python mlmcsc-deployment\scripts\health_check.py
```

The health check verifies:
- Python environment and dependencies
- Database connectivity
- Redis connectivity
- API server status
- File system permissions
- Celery worker status

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Ensure PostgreSQL is running
   - Check database credentials in `.env` file
   - Verify database exists and user has permissions

2. **Redis Connection Failed**
   - Ensure Redis server is running
   - Check Redis URL in configuration
   - Verify Redis is accessible on specified port

3. **API Server Won't Start**
   - Check if port 8000 is already in use
   - Review logs in `logs/api_server.log`
   - Ensure all dependencies are installed

4. **Celery Worker Issues**
   - Check Redis broker connectivity
   - Review logs in `logs/celery_worker.log`
   - Ensure worker has access to model files

### Log Files

Check these log files for debugging:
- `logs/mlmcsc.log` - Main application log
- `logs/api_server.log` - API server log
- `logs/celery_worker.log` - Worker process log
- `logs/mlmcsc_errors.log` - Error-only log

### Performance Tuning

1. **Database Performance**
   - Increase `pool_size` in `app_config.yaml`
   - Add database indexes for frequently queried fields
   - Consider read replicas for high-load scenarios

2. **Redis Performance**
   - Increase `max_connections` in configuration
   - Monitor memory usage and adjust `maxmemory` policy
   - Use Redis clustering for high availability

3. **Worker Performance**
   - Increase number of Celery workers
   - Adjust `prediction_batch_size` for ML tasks
   - Use GPU acceleration if available

## Security Considerations

### Production Deployment

1. **Change Default Passwords**
   - Update database passwords
   - Generate strong `SECRET_KEY`
   - Use environment-specific credentials

2. **Network Security**
   - Use HTTPS in production
   - Configure firewall rules
   - Limit database access to application servers

3. **File Permissions**
   - Ensure proper file ownership
   - Restrict access to configuration files
   - Use dedicated service accounts

### SSL/TLS Configuration

For production deployments, configure SSL/TLS:

1. **Obtain SSL Certificate**
   - Use Let's Encrypt for free certificates
   - Or purchase from a certificate authority

2. **Configure Nginx** (recommended)
   - Set up reverse proxy with SSL termination
   - Configure HTTP to HTTPS redirect
   - Use strong cipher suites

## Backup and Recovery

### Database Backup
```bash
# Create backup
pg_dump -U mlmcsc -h localhost mlmcsc > backup.sql

# Restore backup
psql -U mlmcsc -h localhost mlmcsc < backup.sql
```

### File System Backup
Important directories to backup:
- `data/` - Application data
- `models/` - ML model files
- `config/` - Configuration files
- `logs/` - Log files (optional)

## Scaling

### Horizontal Scaling

1. **Multiple API Servers**
   - Run multiple API server instances
   - Use load balancer (Nginx, HAProxy)
   - Share session state via Redis

2. **Multiple Workers**
   - Start additional Celery workers
   - Distribute across multiple machines
   - Use Redis as shared message broker

### Vertical Scaling

1. **Increase Resources**
   - Add more CPU cores
   - Increase RAM allocation
   - Use faster storage (SSD)

2. **Optimize Configuration**
   - Tune database connection pools
   - Adjust worker concurrency
   - Optimize caching strategies

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files for error messages
3. Run the health check script
4. Consult the main project documentation

## Version Information

- **Deployment Package Version**: 1.0.0
- **Compatible MLMCSC Versions**: 1.0.0+
- **Last Updated**: 2024-01-20

---

**Note**: This deployment package is designed for both development and production use. For production deployments, ensure you follow the security considerations and performance tuning guidelines.