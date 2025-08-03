#!/bin/bash
# Blue-Green Deployment Script for MLMCSC

set -e

# Configuration
NAMESPACE=${NAMESPACE:-"mlmcsc"}
NEW_IMAGE_TAG=${1:-"latest"}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-300}
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}

echo "Starting Blue-Green Deployment for MLMCSC"
echo "Namespace: $NAMESPACE"
echo "New Image Tag: $NEW_IMAGE_TAG"

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster"
    exit 1
fi

# Function to get current active color
get_active_color() {
    kubectl get service mlmcsc-api -n $NAMESPACE -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue"
}

# Function to get inactive color
get_inactive_color() {
    local active_color=$1
    if [[ "$active_color" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Function to check deployment health
check_deployment_health() {
    local color=$1
    local deployment="mlmcsc-api-$color"
    
    echo "Checking health of $deployment..."
    
    # Wait for deployment to be ready
    if ! kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=${HEALTH_CHECK_TIMEOUT}s; then
        echo "Error: Deployment $deployment failed to become ready"
        return 1
    fi
    
    # Get pod IP for health check
    local pod_ip=$(kubectl get pod -l app=mlmcsc-api,color=$color -n $NAMESPACE -o jsonpath='{.items[0].status.podIP}')
    
    if [[ -z "$pod_ip" ]]; then
        echo "Error: Could not get pod IP for $deployment"
        return 1
    fi
    
    # Perform health check
    echo "Performing health check on $pod_ip..."
    
    # Use a temporary pod to perform the health check
    kubectl run health-check-$(date +%s) \
        --image=curlimages/curl:latest \
        --rm -i --restart=Never \
        --timeout=30s \
        -- curl -f http://$pod_ip:8000/health
    
    if [[ $? -eq 0 ]]; then
        echo "Health check passed for $deployment"
        return 0
    else
        echo "Health check failed for $deployment"
        return 1
    fi
}

# Function to switch traffic
switch_traffic() {
    local new_color=$1
    
    echo "Switching traffic to $new_color environment..."
    
    kubectl patch service mlmcsc-api -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"color\":\"$new_color\"}}}"
    
    if [[ $? -eq 0 ]]; then
        echo "Traffic switched to $new_color successfully"
        return 0
    else
        echo "Failed to switch traffic to $new_color"
        return 1
    fi
}

# Function to scale down old deployment
scale_down_old() {
    local old_color=$1
    
    echo "Scaling down $old_color environment..."
    
    kubectl scale deployment mlmcsc-api-$old_color --replicas=0 -n $NAMESPACE
    kubectl scale deployment mlmcsc-worker-$old_color --replicas=0 -n $NAMESPACE
    
    echo "Scaled down $old_color environment"
}

# Function to rollback deployment
rollback_deployment() {
    local old_color=$1
    local new_color=$2
    
    echo "Rolling back deployment..."
    
    # Scale up old deployment
    kubectl scale deployment mlmcsc-api-$old_color --replicas=3 -n $NAMESPACE
    kubectl scale deployment mlmcsc-worker-$old_color --replicas=2 -n $NAMESPACE
    
    # Wait for old deployment to be ready
    kubectl rollout status deployment/mlmcsc-api-$old_color -n $NAMESPACE --timeout=300s
    
    # Switch traffic back
    switch_traffic $old_color
    
    # Scale down failed deployment
    kubectl scale deployment mlmcsc-api-$new_color --replicas=0 -n $NAMESPACE
    kubectl scale deployment mlmcsc-worker-$new_color --replicas=0 -n $NAMESPACE
    
    echo "Rollback completed"
}

# Main deployment logic
main() {
    # Get current active color
    ACTIVE_COLOR=$(get_active_color)
    INACTIVE_COLOR=$(get_inactive_color $ACTIVE_COLOR)
    
    echo "Current active environment: $ACTIVE_COLOR"
    echo "Deploying to inactive environment: $INACTIVE_COLOR"
    
    # Update deployment manifests with new image tag
    echo "Updating deployment manifests..."
    
    # Create temporary manifests with new image tags
    TEMP_DIR=$(mktemp -d)
    
    # API deployment
    cat > $TEMP_DIR/api-$INACTIVE_COLOR.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlmcsc-api-$INACTIVE_COLOR
  namespace: $NAMESPACE
  labels:
    app: mlmcsc-api
    color: $INACTIVE_COLOR
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlmcsc-api
      color: $INACTIVE_COLOR
  template:
    metadata:
      labels:
        app: mlmcsc-api
        color: $INACTIVE_COLOR
        component: api
    spec:
      containers:
      - name: api
        image: mlmcsc/api:$NEW_IMAGE_TAG
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(POSTGRES_USER):\$(POSTGRES_PASSWORD)@postgres:5432/\$(POSTGRES_DB)"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@redis:6379/\$(REDIS_DB)"
        - name: CELERY_BROKER_URL
          value: "redis://:\$(REDIS_PASSWORD)@redis:6379/\$(CELERY_BROKER_DB)"
        - name: CELERY_RESULT_BACKEND
          value: "redis://:\$(REDIS_PASSWORD)@redis:6379/\$(CELERY_RESULT_DB)"
        envFrom:
        - configMapRef:
            name: mlmcsc-config
        - secretRef:
            name: mlmcsc-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
EOF

    # Worker deployment
    cat > $TEMP_DIR/worker-$INACTIVE_COLOR.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlmcsc-worker-$INACTIVE_COLOR
  namespace: $NAMESPACE
  labels:
    app: mlmcsc-worker
    color: $INACTIVE_COLOR
    component: worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlmcsc-worker
      color: $INACTIVE_COLOR
  template:
    metadata:
      labels:
        app: mlmcsc-worker
        color: $INACTIVE_COLOR
        component: worker
    spec:
      containers:
      - name: worker
        image: mlmcsc/worker:$NEW_IMAGE_TAG
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(POSTGRES_USER):\$(POSTGRES_PASSWORD)@postgres:5432/\$(POSTGRES_DB)"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@redis:6379/\$(REDIS_DB)"
        - name: CELERY_BROKER_URL
          value: "redis://:\$(REDIS_PASSWORD)@redis:6379/\$(CELERY_BROKER_DB)"
        - name: CELERY_RESULT_BACKEND
          value: "redis://:\$(REDIS_PASSWORD)@redis:6379/\$(CELERY_RESULT_DB)"
        envFrom:
        - configMapRef:
            name: mlmcsc-config
        - secretRef:
            name: mlmcsc-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
EOF

    # Deploy to inactive environment
    echo "Deploying to $INACTIVE_COLOR environment..."
    
    kubectl apply -f $TEMP_DIR/api-$INACTIVE_COLOR.yaml
    kubectl apply -f $TEMP_DIR/worker-$INACTIVE_COLOR.yaml
    
    # Wait for deployment to be ready and perform health checks
    if check_deployment_health $INACTIVE_COLOR; then
        echo "Deployment to $INACTIVE_COLOR successful"
        
        # Switch traffic to new environment
        if switch_traffic $INACTIVE_COLOR; then
            echo "Traffic switch successful"
            
            # Wait a bit to ensure traffic is flowing
            echo "Waiting 30 seconds to verify traffic flow..."
            sleep 30
            
            # Perform final health check on the service
            echo "Performing final health check..."
            SERVICE_IP=$(kubectl get service mlmcsc-api -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
            
            kubectl run final-health-check-$(date +%s) \
                --image=curlimages/curl:latest \
                --rm -i --restart=Never \
                --timeout=30s \
                -- curl -f http://$SERVICE_IP:8000/health
            
            if [[ $? -eq 0 ]]; then
                echo "Final health check passed"
                
                # Scale down old environment
                scale_down_old $ACTIVE_COLOR
                
                echo "Blue-Green deployment completed successfully!"
                echo "Active environment is now: $INACTIVE_COLOR"
                
                # Clean up temporary files
                rm -rf $TEMP_DIR
                
                exit 0
            else
                echo "Final health check failed"
                if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                    rollback_deployment $ACTIVE_COLOR $INACTIVE_COLOR
                fi
                exit 1
            fi
        else
            echo "Traffic switch failed"
            if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                rollback_deployment $ACTIVE_COLOR $INACTIVE_COLOR
            fi
            exit 1
        fi
    else
        echo "Deployment to $INACTIVE_COLOR failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment $ACTIVE_COLOR $INACTIVE_COLOR
        fi
        exit 1
    fi
    
    # Clean up temporary files
    rm -rf $TEMP_DIR
}

# Run main function
main "$@"