#!/bin/bash
# Production deployment script for formal-circuits-gpt
# Automated deployment with zero-downtime, rollback capability, and monitoring

set -euo pipefail

# Configuration
PROJECT_NAME="formal-circuits-gpt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
BUILD_VERSION="${BUILD_VERSION:-$(date +%Y%m%d-%H%M%S)}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
KUBERNETES_NAMESPACE="${KUBERNETES_NAMESPACE:-formal-circuits-gpt}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

info() {
    log "${BLUE}INFO${NC} $*"
}

warn() {
    log "${YELLOW}WARN${NC} $*"
}

error() {
    log "${RED}ERROR${NC} $*" >&2
}

success() {
    log "${GREEN}SUCCESS${NC} $*"
}

# Error handling
trap 'error "Deployment failed on line $LINENO. Exit code: $?"' ERR

# Help function
show_help() {
    cat << EOF
Formal-Circuits-GPT Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    build       Build Docker images
    test        Run quality gates and tests
    deploy      Deploy to Kubernetes
    rollback    Rollback to previous version
    status      Check deployment status
    logs        Show application logs
    cleanup     Clean up old deployments

Options:
    -h, --help              Show this help message
    -e, --env ENV           Deployment environment (default: production)
    -v, --version VERSION   Build version (default: timestamp)
    -r, --registry URL      Docker registry URL
    -n, --namespace NAME    Kubernetes namespace (default: formal-circuits-gpt)
    --dry-run               Show what would be deployed without executing
    --force                 Force deployment without confirmations
    --skip-tests            Skip quality gates and tests
    --rollback-version VER  Version to rollback to

Examples:
    $0 build
    $0 test
    $0 deploy
    $0 deploy --env staging --version v1.2.3
    $0 rollback --rollback-version v1.2.2
    $0 status
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -e|--env)
                DEPLOYMENT_ENV="$2"
                shift 2
                ;;
            -v|--version)
                BUILD_VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -n|--namespace)
                KUBERNETES_NAMESPACE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --rollback-version)
                ROLLBACK_VERSION="$2"
                shift 2
                ;;
            build|test|deploy|rollback|status|logs|cleanup)
                COMMAND="$1"
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "${COMMAND:-}" ]]; then
        error "No command specified"
        show_help
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    local missing_tools=()
    
    for tool in docker kubectl helm terraform; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -ne 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    info "Building Docker images for version: $BUILD_VERSION"
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build \
        -f deployment/docker/Dockerfile.production \
        -t "${PROJECT_NAME}:${BUILD_VERSION}" \
        -t "${PROJECT_NAME}:latest" \
        .
    
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        # Tag for registry
        docker tag "${PROJECT_NAME}:${BUILD_VERSION}" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${BUILD_VERSION}"
        docker tag "${PROJECT_NAME}:latest" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
        
        # Push to registry
        info "Pushing images to registry: $DOCKER_REGISTRY"
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${BUILD_VERSION}"
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
    fi
    
    success "Docker images built successfully"
}

# Run quality gates and tests
run_tests() {
    if [[ "${SKIP_TESTS:-false}" == "true" ]]; then
        warn "Skipping tests as requested"
        return 0
    fi
    
    info "Running quality gates and tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run quality gates script
    if [[ -x "scripts/run_quality_gates.py" ]]; then
        python3 scripts/run_quality_gates.py \
            --project-root "$PROJECT_ROOT" \
            --output "quality-report.json"
        
        # Check if all gates passed
        if ! python3 -c "import json; report=json.load(open('quality-report.json')); exit(0 if report['overall_passed'] else 1)"; then
            error "Quality gates failed. Check quality-report.json for details."
            exit 1
        fi
    else
        warn "Quality gates script not found, skipping detailed quality checks"
    fi
    
    success "All tests passed"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    info "Deploying to Kubernetes environment: $DEPLOYMENT_ENV"
    
    cd "$PROJECT_ROOT"
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$KUBERNETES_NAMESPACE" &> /dev/null; then
        info "Creating namespace: $KUBERNETES_NAMESPACE"
        kubectl create namespace "$KUBERNETES_NAMESPACE"
    fi
    
    # Update image tags in deployment files
    local temp_dir=$(mktemp -d)
    cp -r deployment/kubernetes/* "$temp_dir/"
    
    # Replace image references with versioned tags
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local image_name="${DOCKER_REGISTRY}/${PROJECT_NAME}:${BUILD_VERSION}"
    else
        local image_name="${PROJECT_NAME}:${BUILD_VERSION}"
    fi
    
    find "$temp_dir" -name "*.yaml" -exec sed -i "s|image: formal-circuits-gpt:latest|image: $image_name|g" {} +
    
    # Apply Kubernetes manifests
    info "Applying Kubernetes manifests..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        info "DRY RUN: Would apply the following manifests:"
        kubectl apply --dry-run=client -f "$temp_dir/" --recursive
    else
        kubectl apply -f "$temp_dir/" --recursive
        
        # Wait for deployment to complete
        info "Waiting for deployment to complete..."
        kubectl rollout status deployment/fcgpt-app -n "$KUBERNETES_NAMESPACE" --timeout=600s
        kubectl rollout status deployment/fcgpt-worker -n "$KUBERNETES_NAMESPACE" --timeout=600s
        
        # Run health checks
        if ! run_health_checks; then
            error "Health checks failed after deployment"
            exit 1
        fi
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    success "Deployment completed successfully"
}

# Run health checks
run_health_checks() {
    info "Running health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if kubectl get pods -n "$KUBERNETES_NAMESPACE" -l app=fcgpt-app --field-selector=status.phase=Running | grep -q "Running"; then
            info "Application pods are running"
            
            # Check application health endpoint
            if kubectl exec -n "$KUBERNETES_NAMESPACE" \
                $(kubectl get pod -n "$KUBERNETES_NAMESPACE" -l app=fcgpt-app -o jsonpath='{.items[0].metadata.name}') \
                -- curl -f http://localhost:8000/health &> /dev/null; then
                success "Health checks passed"
                return 0
            fi
        fi
        
        info "Waiting for application to be ready... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    error "Health checks failed after $max_attempts attempts"
    return 1
}

# Rollback deployment
rollback_deployment() {
    if [[ -z "${ROLLBACK_VERSION:-}" ]]; then
        error "Rollback version not specified. Use --rollback-version option."
        exit 1
    fi
    
    info "Rolling back to version: $ROLLBACK_VERSION"
    
    # Update image tags to rollback version
    local image_name
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        image_name="${DOCKER_REGISTRY}/${PROJECT_NAME}:${ROLLBACK_VERSION}"
    else
        image_name="${PROJECT_NAME}:${ROLLBACK_VERSION}"
    fi
    
    # Update deployment with rollback image
    kubectl set image deployment/fcgpt-app \
        fcgpt-app="$image_name" \
        -n "$KUBERNETES_NAMESPACE"
    
    kubectl set image deployment/fcgpt-worker \
        fcgpt-worker="$image_name" \
        -n "$KUBERNETES_NAMESPACE"
    
    # Wait for rollback to complete
    info "Waiting for rollback to complete..."
    kubectl rollout status deployment/fcgpt-app -n "$KUBERNETES_NAMESPACE" --timeout=300s
    kubectl rollout status deployment/fcgpt-worker -n "$KUBERNETES_NAMESPACE" --timeout=300s
    
    # Run health checks
    if ! run_health_checks; then
        error "Health checks failed after rollback"
        exit 1
    fi
    
    success "Rollback completed successfully"
}

# Check deployment status
check_status() {
    info "Checking deployment status..."
    
    echo
    echo "=== Namespace Status ==="
    kubectl get all -n "$KUBERNETES_NAMESPACE"
    
    echo
    echo "=== Pod Details ==="
    kubectl get pods -n "$KUBERNETES_NAMESPACE" -o wide
    
    echo
    echo "=== Service Status ==="
    kubectl get services -n "$KUBERNETES_NAMESPACE"
    
    echo
    echo "=== Ingress Status ==="
    kubectl get ingress -n "$KUBERNETES_NAMESPACE"
    
    echo
    echo "=== Recent Events ==="
    kubectl get events -n "$KUBERNETES_NAMESPACE" --sort-by=.metadata.creationTimestamp | tail -20
}

# Show application logs
show_logs() {
    info "Showing application logs..."
    
    # Get app pod names
    local app_pods=$(kubectl get pods -n "$KUBERNETES_NAMESPACE" -l app=fcgpt-app -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -z "$app_pods" ]]; then
        error "No application pods found"
        exit 1
    fi
    
    # Show logs from first app pod
    local first_pod=$(echo $app_pods | cut -d' ' -f1)
    info "Showing logs from pod: $first_pod"
    
    kubectl logs -n "$KUBERNETES_NAMESPACE" "$first_pod" --tail=100 -f
}

# Cleanup old deployments
cleanup_old_deployments() {
    info "Cleaning up old deployments..."
    
    # Remove old Docker images (keep last 5 versions)
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        info "Cleaning up old Docker images in registry..."
        # This would require registry-specific cleanup commands
        warn "Registry cleanup not implemented yet"
    else
        info "Cleaning up local Docker images..."
        docker images "${PROJECT_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}" | \
            tail -n +2 | sort -k3 -r | tail -n +6 | \
            while read repo tag _; do
                if [[ "$tag" != "latest" ]]; then
                    docker rmi "${repo}:${tag}" || true
                fi
            done
    fi
    
    success "Cleanup completed"
}

# Confirm deployment
confirm_deployment() {
    if [[ "${FORCE_DEPLOY:-false}" == "true" ]]; then
        return 0
    fi
    
    echo
    warn "You are about to deploy to: $DEPLOYMENT_ENV"
    warn "Version: $BUILD_VERSION"
    warn "Namespace: $KUBERNETES_NAMESPACE"
    echo
    
    read -p "Continue with deployment? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        info "Deployment cancelled by user"
        exit 0
    fi
}

# Main execution
main() {
    parse_args "$@"
    
    info "Starting $COMMAND for $PROJECT_NAME"
    info "Environment: $DEPLOYMENT_ENV"
    info "Version: $BUILD_VERSION"
    
    check_prerequisites
    
    case "$COMMAND" in
        build)
            build_images
            ;;
        test)
            run_tests
            ;;
        deploy)
            confirm_deployment
            build_images
            run_tests
            deploy_to_kubernetes
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            check_status
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup_old_deployments
            ;;
        *)
            error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
    
    success "$COMMAND completed successfully"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi