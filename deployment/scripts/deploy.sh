#!/bin/bash
# Deploy MLMCSC to Kubernetes

set -e

# Configuration
ENVIRONMENT=${1:-"development"}
NAMESPACE="mlmcsc-$ENVIRONMENT"
PROJECT_ROOT=$(dirname $(dirname $(dirname $(realpath $0))))
KUBE_DIR="$PROJECT_ROOT/deployment/kubernetes"

echo "Deploying MLMCSC to $ENVIRONMENT environment..."
echo "Namespace: $NAMESPACE"

# Validate environment
case $ENVIRONMENT in
    development|staging|production)
        echo "Valid environment: $ENVIRONMENT"
        ;;
    *)
        echo "Error: Invalid environment '$ENVIRONMENT'"
        echo "Valid environments: development, staging, production"
        exit 1
        ;;
esac

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster"
    exit 1
fi

# Create namespace if it doesn't exist
echo "Creating namespace $NAMESPACE..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply base configurations
echo "Applying base configurations..."
kubectl apply -f $KUBE_DIR/base/configmap.yaml -n $NAMESPACE
kubectl apply -f $KUBE_DIR/base/secrets.yaml -n $NAMESPACE

# Apply database and cache
echo "Deploying database and cache..."
kubectl apply -f $KUBE_DIR/base/postgres.yaml -n $NAMESPACE
kubectl apply -f $KUBE_DIR/base/redis.yaml -n $NAMESPACE

# Wait for database and cache to be ready
echo "Waiting for database and cache to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# Apply MinIO if not production (use external S3 in production)
if [[ $ENVIRONMENT != "production" ]]; then
    echo "Deploying MinIO..."
    kubectl apply -f $KUBE_DIR/base/minio.yaml -n $NAMESPACE
    kubectl wait --for=condition=ready pod -l app=minio -n $NAMESPACE --timeout=300s
fi

# Apply application components
echo "Deploying application components..."
kubectl apply -f $KUBE_DIR/base/api.yaml -n $NAMESPACE
kubectl apply -f $KUBE_DIR/base/workers.yaml -n $NAMESPACE

# Apply environment-specific overlays if they exist
OVERLAY_DIR="$KUBE_DIR/overlays/$ENVIRONMENT"
if [[ -d $OVERLAY_DIR ]]; then
    echo "Applying $ENVIRONMENT-specific configurations..."
    kubectl apply -k $OVERLAY_DIR -n $NAMESPACE
fi

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment/mlmcsc-api -n $NAMESPACE --timeout=600s
kubectl wait --for=condition=available deployment/mlmcsc-worker -n $NAMESPACE --timeout=600s

# Apply ingress
echo "Configuring ingress..."
kubectl apply -f $KUBE_DIR/base/ingress.yaml -n $NAMESPACE

# Apply monitoring if enabled
if [[ -f $KUBE_DIR/monitoring/prometheus.yaml ]]; then
    echo "Deploying monitoring stack..."
    kubectl apply -f $KUBE_DIR/monitoring/ -n $NAMESPACE
fi

# Run database migrations
echo "Running database migrations..."
kubectl run mlmcsc-migrate-$(date +%s) \
    --image=mlmcsc/api:latest \
    --rm -i --restart=Never \
    --env="DATABASE_URL=postgresql://mlmcsc:$(kubectl get secret mlmcsc-secrets -n $NAMESPACE -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)@postgres:5432/mlmcsc" \
    --command -- python -m alembic upgrade head

# Display deployment status
echo ""
echo "Deployment completed successfully!"
echo ""
echo "Deployment Status:"
kubectl get deployments -n $NAMESPACE
echo ""
echo "Services:"
kubectl get services -n $NAMESPACE
echo ""
echo "Ingress:"
kubectl get ingress -n $NAMESPACE

# Get application URL
if kubectl get ingress mlmcsc-ingress -n $NAMESPACE &> /dev/null; then
    APP_URL=$(kubectl get ingress mlmcsc-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}')
    echo ""
    echo "Application URL: https://$APP_URL"
fi

# Display useful commands
echo ""
echo "Useful commands:"
echo "  View logs: kubectl logs -f deployment/mlmcsc-api -n $NAMESPACE"
echo "  Scale API: kubectl scale deployment mlmcsc-api --replicas=5 -n $NAMESPACE"
echo "  Port forward: kubectl port-forward service/mlmcsc-api 8080:8000 -n $NAMESPACE"
echo "  Delete deployment: kubectl delete namespace $NAMESPACE"