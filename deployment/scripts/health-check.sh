#!/bin/bash
# Health Check Script for MLMCSC Production Deployment

set -e

# Configuration
NAMESPACE=${NAMESPACE:-"mlmcsc"}
TIMEOUT=${TIMEOUT:-30}
VERBOSE=${VERBOSE:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if kubectl is available and configured
check_kubectl() {
    log_info "Checking kubectl configuration..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        return 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    log_success "kubectl is configured and cluster is accessible"
    return 0
}

# Function to check namespace
check_namespace() {
    log_info "Checking namespace '$NAMESPACE'..."
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_success "Namespace '$NAMESPACE' exists"
        return 0
    else
        log_error "Namespace '$NAMESPACE' does not exist"
        return 1
    fi
}

# Function to check pod status
check_pods() {
    log_info "Checking pod status..."
    
    local failed_pods=0
    local total_pods=0
    
    # Get all pods in the namespace
    while IFS= read -r line; do
        if [[ $line == *"NAME"* ]]; then
            continue  # Skip header
        fi
        
        total_pods=$((total_pods + 1))
        
        local pod_name=$(echo $line | awk '{print $1}')
        local ready=$(echo $line | awk '{print $2}')
        local status=$(echo $line | awk '{print $3}')
        local restarts=$(echo $line | awk '{print $4}')
        
        if [[ $status != "Running" ]] || [[ $ready == *"0/"* ]]; then
            log_error "Pod $pod_name is not healthy: Status=$status, Ready=$ready, Restarts=$restarts"
            failed_pods=$((failed_pods + 1))
            
            if [[ $VERBOSE == "true" ]]; then
                log_info "Pod details for $pod_name:"
                kubectl describe pod $pod_name -n $NAMESPACE | tail -20
            fi
        else
            if [[ $VERBOSE == "true" ]]; then
                log_success "Pod $pod_name is healthy: Status=$status, Ready=$ready"
            fi
        fi
    done < <(kubectl get pods -n $NAMESPACE --no-headers)
    
    if [[ $failed_pods -eq 0 ]]; then
        log_success "All $total_pods pods are healthy"
        return 0
    else
        log_error "$failed_pods out of $total_pods pods are unhealthy"
        return 1
    fi
}

# Function to check deployments
check_deployments() {
    log_info "Checking deployment status..."
    
    local failed_deployments=0
    local total_deployments=0
    
    while IFS= read -r line; do
        if [[ $line == *"NAME"* ]]; then
            continue  # Skip header
        fi
        
        total_deployments=$((total_deployments + 1))
        
        local deployment_name=$(echo $line | awk '{print $1}')
        local ready=$(echo $line | awk '{print $2}')
        local up_to_date=$(echo $line | awk '{print $3}')
        local available=$(echo $line | awk '{print $4}')
        
        # Parse ready status (e.g., "3/3")
        local ready_count=$(echo $ready | cut -d'/' -f1)
        local desired_count=$(echo $ready | cut -d'/' -f2)
        
        if [[ $ready_count != $desired_count ]] || [[ $available == "0" ]]; then
            log_error "Deployment $deployment_name is not healthy: Ready=$ready, Available=$available"
            failed_deployments=$((failed_deployments + 1))
            
            if [[ $VERBOSE == "true" ]]; then
                log_info "Deployment details for $deployment_name:"
                kubectl describe deployment $deployment_name -n $NAMESPACE | tail -10
            fi
        else
            if [[ $VERBOSE == "true" ]]; then
                log_success "Deployment $deployment_name is healthy: Ready=$ready, Available=$available"
            fi
        fi
    done < <(kubectl get deployments -n $NAMESPACE --no-headers)
    
    if [[ $failed_deployments -eq 0 ]]; then
        log_success "All $total_deployments deployments are healthy"
        return 0
    else
        log_error "$failed_deployments out of $total_deployments deployments are unhealthy"
        return 1
    fi
}

# Function to check services
check_services() {
    log_info "Checking service status..."
    
    local services_count=0
    
    while IFS= read -r line; do
        if [[ $line == *"NAME"* ]]; then
            continue  # Skip header
        fi
        
        services_count=$((services_count + 1))
        
        local service_name=$(echo $line | awk '{print $1}')
        local type=$(echo $line | awk '{print $2}')
        local cluster_ip=$(echo $line | awk '{print $3}')
        
        if [[ $cluster_ip == "<none>" ]] && [[ $type != "ExternalName" ]]; then
            log_warning "Service $service_name has no cluster IP"
        else
            if [[ $VERBOSE == "true" ]]; then
                log_success "Service $service_name is available: Type=$type, ClusterIP=$cluster_ip"
            fi
        fi
    done < <(kubectl get services -n $NAMESPACE --no-headers)
    
    log_success "Checked $services_count services"
    return 0
}

# Function to check ingress
check_ingress() {
    log_info "Checking ingress status..."
    
    if ! kubectl get ingress -n $NAMESPACE &> /dev/null; then
        log_warning "No ingress resources found"
        return 0
    fi
    
    while IFS= read -r line; do
        if [[ $line == *"NAME"* ]]; then
            continue  # Skip header
        fi
        
        local ingress_name=$(echo $line | awk '{print $1}')
        local hosts=$(echo $line | awk '{print $3}')
        local address=$(echo $line | awk '{print $4}')
        
        if [[ -z $address ]]; then
            log_warning "Ingress $ingress_name has no external address"
        else
            if [[ $VERBOSE == "true" ]]; then
                log_success "Ingress $ingress_name is configured: Hosts=$hosts, Address=$address"
            fi
        fi
    done < <(kubectl get ingress -n $NAMESPACE --no-headers)
    
    return 0
}

# Function to check persistent volumes
check_persistent_volumes() {
    log_info "Checking persistent volume claims..."
    
    if ! kubectl get pvc -n $NAMESPACE &> /dev/null; then
        log_info "No persistent volume claims found"
        return 0
    fi
    
    local failed_pvcs=0
    local total_pvcs=0
    
    while IFS= read -r line; do
        if [[ $line == *"NAME"* ]]; then
            continue  # Skip header
        fi
        
        total_pvcs=$((total_pvcs + 1))
        
        local pvc_name=$(echo $line | awk '{print $1}')
        local status=$(echo $line | awk '{print $2}')
        local volume=$(echo $line | awk '{print $3}')
        
        if [[ $status != "Bound" ]]; then
            log_error "PVC $pvc_name is not bound: Status=$status"
            failed_pvcs=$((failed_pvcs + 1))
        else
            if [[ $VERBOSE == "true" ]]; then
                log_success "PVC $pvc_name is bound to volume $volume"
            fi
        fi
    done < <(kubectl get pvc -n $NAMESPACE --no-headers)
    
    if [[ $failed_pvcs -eq 0 ]]; then
        log_success "All $total_pvcs PVCs are bound"
        return 0
    else
        log_error "$failed_pvcs out of $total_pvcs PVCs are not bound"
        return 1
    fi
}

# Function to perform application health checks
check_application_health() {
    log_info "Performing application health checks..."
    
    # Get API service cluster IP
    local api_service_ip=$(kubectl get service mlmcsc-api -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    
    if [[ -z $api_service_ip ]]; then
        log_error "Cannot find mlmcsc-api service"
        return 1
    fi
    
    # Perform health check using a temporary pod
    log_info "Testing API health endpoint..."
    
    local health_check_result=$(kubectl run health-check-$(date +%s) \
        --image=curlimages/curl:latest \
        --rm -i --restart=Never \
        --timeout=${TIMEOUT}s \
        --quiet \
        -- curl -s -f http://$api_service_ip:8000/health 2>/dev/null || echo "FAILED")
    
    if [[ $health_check_result == "FAILED" ]]; then
        log_error "API health check failed"
        return 1
    else
        log_success "API health check passed"
        if [[ $VERBOSE == "true" ]]; then
            log_info "Health check response: $health_check_result"
        fi
        return 0
    fi
}

# Function to check resource usage
check_resource_usage() {
    log_info "Checking resource usage..."
    
    # Check if metrics server is available
    if ! kubectl top nodes &> /dev/null; then
        log_warning "Metrics server not available, skipping resource usage check"
        return 0
    fi
    
    log_info "Node resource usage:"
    kubectl top nodes
    
    log_info "Pod resource usage in namespace $NAMESPACE:"
    kubectl top pods -n $NAMESPACE
    
    return 0
}

# Function to check recent events
check_events() {
    log_info "Checking recent events..."
    
    local warning_events=$(kubectl get events -n $NAMESPACE --field-selector type=Warning --no-headers 2>/dev/null | wc -l)
    local error_events=$(kubectl get events -n $NAMESPACE --field-selector type=Error --no-headers 2>/dev/null | wc -l)
    
    if [[ $error_events -gt 0 ]]; then
        log_error "Found $error_events error events in the last hour"
        if [[ $VERBOSE == "true" ]]; then
            kubectl get events -n $NAMESPACE --field-selector type=Error --sort-by='.lastTimestamp' | tail -5
        fi
        return 1
    elif [[ $warning_events -gt 0 ]]; then
        log_warning "Found $warning_events warning events in the last hour"
        if [[ $VERBOSE == "true" ]]; then
            kubectl get events -n $NAMESPACE --field-selector type=Warning --sort-by='.lastTimestamp' | tail -5
        fi
    else
        log_success "No error or warning events found"
    fi
    
    return 0
}

# Function to generate summary report
generate_summary() {
    local total_checks=$1
    local failed_checks=$2
    
    echo ""
    echo "=================================="
    echo "MLMCSC Health Check Summary"
    echo "=================================="
    echo "Namespace: $NAMESPACE"
    echo "Timestamp: $(date)"
    echo "Total Checks: $total_checks"
    echo "Failed Checks: $failed_checks"
    echo "Success Rate: $(( (total_checks - failed_checks) * 100 / total_checks ))%"
    echo "=================================="
    
    if [[ $failed_checks -eq 0 ]]; then
        log_success "All health checks passed! System is healthy."
        return 0
    else
        log_error "Some health checks failed. Please investigate the issues above."
        return 1
    fi
}

# Main function
main() {
    echo "Starting MLMCSC Health Check..."
    echo "Namespace: $NAMESPACE"
    echo "Timeout: ${TIMEOUT}s"
    echo "Verbose: $VERBOSE"
    echo ""
    
    local total_checks=0
    local failed_checks=0
    
    # Run all health checks
    local checks=(
        "check_kubectl"
        "check_namespace"
        "check_pods"
        "check_deployments"
        "check_services"
        "check_ingress"
        "check_persistent_volumes"
        "check_application_health"
        "check_resource_usage"
        "check_events"
    )
    
    for check in "${checks[@]}"; do
        total_checks=$((total_checks + 1))
        
        if ! $check; then
            failed_checks=$((failed_checks + 1))
        fi
        
        echo ""  # Add spacing between checks
    done
    
    # Generate summary
    generate_summary $total_checks $failed_checks
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --namespace NAMESPACE    Kubernetes namespace (default: mlmcsc)"
            echo "  -t, --timeout TIMEOUT       Timeout in seconds (default: 30)"
            echo "  -v, --verbose               Enable verbose output"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Basic health check"
            echo "  $0 -n production -v         # Verbose check in production namespace"
            echo "  $0 -t 60                    # Check with 60 second timeout"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"