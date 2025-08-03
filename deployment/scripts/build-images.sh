#!/bin/bash
# Build Docker images for MLMCSC deployment

set -e

# Configuration
REGISTRY=${REGISTRY:-"localhost:5000"}
TAG=${TAG:-"latest"}
PROJECT_ROOT=$(dirname $(dirname $(dirname $(realpath $0))))

echo "Building MLMCSC Docker images..."
echo "Registry: $REGISTRY"
echo "Tag: $TAG"
echo "Project root: $PROJECT_ROOT"

# Build API image
echo "Building API image..."
docker build \
    -t $REGISTRY/mlmcsc/api:$TAG \
    -f $PROJECT_ROOT/deployment/docker/api/Dockerfile \
    $PROJECT_ROOT

# Build Worker image
echo "Building Worker image..."
docker build \
    -t $REGISTRY/mlmcsc/worker:$TAG \
    -f $PROJECT_ROOT/deployment/docker/worker/Dockerfile \
    $PROJECT_ROOT

# Build Nginx image
echo "Building Nginx image..."
docker build \
    -t $REGISTRY/mlmcsc/nginx:$TAG \
    -f $PROJECT_ROOT/deployment/docker/nginx/Dockerfile \
    $PROJECT_ROOT/deployment/docker/nginx/

# Push images if registry is not localhost
if [[ $REGISTRY != "localhost:5000" ]]; then
    echo "Pushing images to registry..."
    docker push $REGISTRY/mlmcsc/api:$TAG
    docker push $REGISTRY/mlmcsc/worker:$TAG
    docker push $REGISTRY/mlmcsc/nginx:$TAG
fi

echo "Docker images built successfully!"
echo "Images:"
echo "  - $REGISTRY/mlmcsc/api:$TAG"
echo "  - $REGISTRY/mlmcsc/worker:$TAG"
echo "  - $REGISTRY/mlmcsc/nginx:$TAG"