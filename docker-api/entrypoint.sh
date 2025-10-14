#!/bin/bash

set -e

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "=== MonkeyOCR Docker Container Starting ==="

# Check GPU availability
if command -v nvidia-smi >/dev/null 2>&1; then
    log_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    log_warn "NVIDIA GPU not detected"
fi

# Download models
log_info "Checking and downloading models..."
if /app/MonkeyOCR/download_models.sh; then
    log_info "Models ready"
else
    log_error "Model download failed, but container will continue running"
    log_error "You can manually download models after entering the container"
fi

# Decide startup method based on passed arguments
if [ $# -eq 0 ]; then
    log_info "Starting interactive Python environment"
    exec python
elif [ "$1" = "demo" ]; then
    log_info "Starting Gradio demo"
    exec python -u demo/demo_gradio.py
elif [ "$1" = "bash" ]; then
    log_info "Starting Bash shell"
    exec /bin/bash
elif [ "$1" = "fastapi" ]; then
    log_info "Starting FastAPI server"
    pip install fastapi-mcp uvicorn schedule
    mkdir -p /app/tmp
    cd /app/MonkeyOCR
    
    # 设置默认worker数量为1,避免多个进程重复加载模型
    WORKERS=${UVICORN_WORKERS:-1}
    TIMEOUT=${UVICORN_TIMEOUT_KEEP_ALIVE:-30}
    LIMIT_CONCURRENCY=${UVICORN_LIMIT_CONCURRENCY:-10}
    
    log_info "Starting uvicorn with workers=$WORKERS, timeout=$TIMEOUT, limit_concurrency=$LIMIT_CONCURRENCY"
    
    # 使用单worker模式,避免多进程重复加载模型导致内存占用过高
    exec uvicorn api.main:app \
        --host ${FASTAPI_HOST:-0.0.0.0} \
        --port ${FASTAPI_PORT:-7861} \
        --workers ${WORKERS} \
        --timeout-keep-alive ${TIMEOUT} \
        --limit-concurrency ${LIMIT_CONCURRENCY} \
        --backlog 100

else
    log_info "Executing custom command: $*"
    exec "$@"
fi
