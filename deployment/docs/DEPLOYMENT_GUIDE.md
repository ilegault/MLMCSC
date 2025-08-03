# MLMCSC Production Deployment Guide

This guide provides comprehensive instructions for deploying the MLMCSC system in a production environment using Docker and Kubernetes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Environment Setup](#environment-setup)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Blue-Green Deployment](#blue-green-deployment)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Security Configuration](#security-configuration)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Kubernetes Cluster**: v1.24+ with at least 3 nodes
- **Docker**: v20.10+ for building images
- **kubectl**: v1.24+ configured for your cluster
- **Helm**: v3.8+ for package management
- **Storage**: 100GB+ persistent storage per node
- **Memory**: 8GB+ RAM per node
- **CPU**: 4+ cores per node

### Required Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

## Architecture Overview

### Production Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Servers   │    │     Workers     │
│     (Nginx)     │────│   (FastAPI)     │────│    (Celery)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │      Cache      │              │
         └──────────────│     (Redis)     │──────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │    Database     │
                        │  (PostgreSQL)   │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │ Object Storage  │
                        │   (S3/MinIO)    │
                        └─────────────────┘
```

### Service Architecture

- **API Layer**: 3+ FastAPI instances behind load balancer
- **Worker Layer**: 2+ Celery workers for async processing
- **Data Layer**: PostgreSQL with read replicas
- **Cache Layer**: Redis cluster for session and prediction caching
- **Storage Layer**: S3-compatible object storage for images and models
- **Monitoring**: Prometheus + Grafana + AlertManager

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/MLMCSC.git
cd MLMCSC
```

### 2. Configure Environment Variables

Create environment-specific configuration files:

```bash
# Create production environment file
cp deployment/configs/production.env.example deployment/configs/production.env

# Edit configuration
vim deployment/configs/production.env
```

Required environment variables:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://mlmcsc:password@postgres:5432/mlmcsc

# Redis
REDIS_PASSWORD=your_redis_password
REDIS_URL=redis://:password@redis:6379/0

# S3 Storage
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_BUCKET=mlmcsc-production

# Security
JWT_SECRET=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
```

### 3. SSL Certificates

Generate or obtain SSL certificates:

```bash
# For Let's Encrypt (recommended for production)
certbot certonly --dns-cloudflare --dns-cloudflare-credentials ~/.secrets/certbot/cloudflare.ini -d mlmcsc.yourdomain.com

# Or generate self-signed for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/ssl/tls.key \
  -out deployment/ssl/tls.crt \
  -subj "/CN=mlmcsc.yourdomain.com"
```

## Docker Deployment

### 1. Build Images

```bash
# Build all images
./deployment/scripts/build-images.sh

# Or build individually
docker build -t mlmcsc/api:latest -f deployment/docker/api/Dockerfile .
docker build -t mlmcsc/worker:latest -f deployment/docker/worker/Dockerfile .
```

### 2. Docker Compose Deployment

For development or small-scale production:

```bash
# Start services
cd deployment/docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 3. Docker Swarm Deployment

For multi-node Docker deployment:

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c deployment/docker/docker-compose.yml mlmcsc

# Check services
docker service ls
```

## Kubernetes Deployment

### 1. Prepare Cluster

```bash
# Create namespace
kubectl create namespace mlmcsc

# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install ingress controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx
```

### 2. Configure Secrets

```bash
# Create secrets from environment file
kubectl create secret generic mlmcsc-secrets \
  --from-env-file=deployment/configs/production.env \
  -n mlmcsc

# Create TLS secret
kubectl create secret tls mlmcsc-tls \
  --cert=deployment/ssl/tls.crt \
  --key=deployment/ssl/tls.key \
  -n mlmcsc
```

### 3. Deploy Application

```bash
# Deploy using script
./deployment/scripts/deploy.sh production

# Or deploy manually
kubectl apply -k deployment/kubernetes/overlays/production/
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n mlmcsc

# Check services
kubectl get services -n mlmcsc

# Check ingress
kubectl get ingress -n mlmcsc

# Test application
curl -k https://mlmcsc.yourdomain.com/health
```

## Blue-Green Deployment

### 1. Automated Blue-Green Deployment

```bash
# Deploy new version
./deployment/scripts/blue-green-deploy.sh v1.2.0

# The script will:
# 1. Deploy to inactive environment
# 2. Run health checks
# 3. Switch traffic
# 4. Scale down old environment
```

### 2. Manual Blue-Green Process

```bash
# 1. Check current active environment
ACTIVE=$(kubectl get service mlmcsc-api -n mlmcsc -o jsonpath='{.spec.selector.color}')
echo "Active environment: $ACTIVE"

# 2. Deploy to inactive environment
INACTIVE=$([ "$ACTIVE" = "blue" ] && echo "green" || echo "blue")
kubectl set image deployment/mlmcsc-api-$INACTIVE api=mlmcsc/api:v1.2.0 -n mlmcsc

# 3. Wait for deployment
kubectl rollout status deployment/mlmcsc-api-$INACTIVE -n mlmcsc

# 4. Run health checks
kubectl run health-check --image=curlimages/curl --rm -i --restart=Never \
  -- curl -f http://mlmcsc-api-$INACTIVE:8000/health

# 5. Switch traffic
kubectl patch service mlmcsc-api -n mlmcsc -p "{\"spec\":{\"selector\":{\"color\":\"$INACTIVE\"}}}"

# 6. Scale down old environment
kubectl scale deployment mlmcsc-api-$ACTIVE --replicas=0 -n mlmcsc
```

## Monitoring and Observability

### 1. Deploy Monitoring Stack

```bash
# Deploy Prometheus
kubectl apply -f deployment/kubernetes/monitoring/prometheus.yaml

# Deploy Grafana
kubectl apply -f deployment/kubernetes/monitoring/grafana.yaml

# Deploy AlertManager
kubectl apply -f deployment/kubernetes/monitoring/alertmanager.yaml
```

### 2. Access Dashboards

```bash
# Port forward to Grafana
kubectl port-forward service/grafana 3000:3000 -n mlmcsc

# Access at http://localhost:3000
# Default credentials: admin / (check secret)
```

### 3. Configure Alerts

Edit `deployment/monitoring/alert_rules.yml` to customize alerts:

```yaml
- alert: MLMCSCAPIDown
  expr: up{job="mlmcsc-api"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "MLMCSC API server is down"
```

## Security Configuration

### 1. Network Policies

```bash
# Apply network policies
kubectl apply -f deployment/kubernetes/security/network-policies.yaml
```

### 2. RBAC Configuration

```bash
# Create service accounts and roles
kubectl apply -f deployment/kubernetes/security/rbac.yaml
```

### 3. Pod Security Standards

```bash
# Apply pod security policies
kubectl apply -f deployment/kubernetes/security/pod-security.yaml
```

### 4. Image Security Scanning

```bash
# Scan images with Trivy
trivy image mlmcsc/api:latest
trivy image mlmcsc/worker:latest
```

## Backup and Recovery

### 1. Database Backup

```bash
# Manual backup
kubectl exec -it postgres-0 -n mlmcsc -- pg_dump -U mlmcsc mlmcsc > backup.sql

# Automated backup (runs daily)
kubectl apply -f deployment/kubernetes/backup/postgres-backup-cronjob.yaml
```

### 2. Object Storage Backup

```bash
# Backup to secondary region
aws s3 sync s3://mlmcsc-production s3://mlmcsc-backup --region us-west-2
```

### 3. Disaster Recovery

```bash
# Restore from backup
kubectl exec -it postgres-0 -n mlmcsc -- psql -U mlmcsc -d mlmcsc < backup.sql

# Restore object storage
aws s3 sync s3://mlmcsc-backup s3://mlmcsc-production
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n mlmcsc

# Check logs
kubectl logs <pod-name> -n mlmcsc

# Common causes:
# - Resource limits
# - Image pull errors
# - Configuration issues
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl run db-test --image=postgres:15 --rm -it --restart=Never \
  -- psql -h postgres -U mlmcsc -d mlmcsc

# Check database logs
kubectl logs postgres-0 -n mlmcsc
```

#### 3. High Memory Usage

```bash
# Check resource usage
kubectl top pods -n mlmcsc

# Scale up if needed
kubectl scale deployment mlmcsc-api --replicas=5 -n mlmcsc
```

#### 4. SSL Certificate Issues

```bash
# Check certificate status
kubectl describe certificate mlmcsc-tls -n mlmcsc

# Renew certificate
kubectl delete certificate mlmcsc-tls -n mlmcsc
kubectl apply -f deployment/kubernetes/base/ingress.yaml
```

### Performance Tuning

#### 1. Database Optimization

```sql
-- Optimize PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

#### 2. Redis Optimization

```bash
# Optimize Redis configuration
kubectl patch configmap redis-config -n mlmcsc --patch '
data:
  redis.conf: |
    maxmemory 1gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
'
```

#### 3. Application Scaling

```bash
# Enable horizontal pod autoscaling
kubectl autoscale deployment mlmcsc-api --cpu-percent=70 --min=3 --max=10 -n mlmcsc
kubectl autoscale deployment mlmcsc-worker --cpu-percent=80 --min=2 --max=8 -n mlmcsc
```

### Monitoring Commands

```bash
# Check cluster health
kubectl get nodes
kubectl get pods --all-namespaces

# Check resource usage
kubectl top nodes
kubectl top pods -n mlmcsc

# Check events
kubectl get events -n mlmcsc --sort-by='.lastTimestamp'

# Check logs
kubectl logs -f deployment/mlmcsc-api -n mlmcsc
kubectl logs -f deployment/mlmcsc-worker -n mlmcsc
```

## Maintenance

### Regular Tasks

1. **Update Dependencies**: Monthly security updates
2. **Certificate Renewal**: Automated with cert-manager
3. **Database Maintenance**: Weekly VACUUM and ANALYZE
4. **Log Rotation**: Automated with logrotate
5. **Backup Verification**: Weekly restore tests
6. **Security Scans**: Weekly vulnerability scans
7. **Performance Review**: Monthly performance analysis

### Upgrade Process

1. **Test in Staging**: Deploy to staging environment first
2. **Backup Data**: Create full backup before upgrade
3. **Blue-Green Deploy**: Use blue-green deployment for zero downtime
4. **Monitor**: Watch metrics during and after deployment
5. **Rollback Plan**: Have rollback procedure ready

For additional support, refer to the [troubleshooting documentation](TROUBLESHOOTING.md) or contact the development team.