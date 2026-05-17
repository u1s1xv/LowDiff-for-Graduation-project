#!/bin/zsh

set -o pipefail

################################################################################
# LowDiff Pipeline Recovery Launcher
# Purpose: Test pipeline-based differential checkpoint recovery
# Usage: bash scripts/pipeline_recovery.sh
################################################################################

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + RANDOM % 1000))
export NCCL_IB_DISABLE=1

# Redirect all caches to /mnt/newdisk (avoid root partition full)
export HF_HOME=/mnt/newdisk/xiekunpeng/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/newdisk/xiekunpeng/.cache/huggingface/datasets
export TORCH_HOME=/mnt/newdisk/xiekunpeng/.cache/torch
export TRITON_CACHE_DIR=/mnt/newdisk/xiekunpeng/.cache/triton
export TRANSFORMERS_CACHE=/mnt/newdisk/xiekunpeng/.cache/huggingface/transformers

# Training parameters
DATASET=wikitext-2
MODEL=gpt2
EPOCHS=1
BATCH_SIZE=4
COMPRESSOR=topk
COMPRESSOR_RATIO=0.01
FREQ=100
SAVE_BATCH_FREQ=20
SAVE_DIR=/mnt/newdisk/xiekunpeng/LowDiff/data/lowdiff
RESUME=1                        # 启用恢复模式
PIPELINE_BUFFER_SIZE=2          # 流水线缓冲区大小
NUM_GPUS=4

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

# Log file
LOG_FILE=$SAVE_DIR/pipeline_recovery_$(date +%Y%m%d_%H%M%S).log

# Initialize log file with header
{
    echo "======================================================================"
    echo "LOWDIFF PIPELINE RECOVERY LOG"
    echo "======================================================================"
    echo "Script: pipeline_recovery.sh"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo "Working Directory: $(pwd)"
    echo "======================================================================"
    echo ""
} > $LOG_FILE

# Function to log messages to both console and file
log() {
    echo "$@" | tee -a $LOG_FILE
}

# Display recovery configuration
log "======================================================================"
log "LowDiff Pipeline Recovery Test"
log "======================================================================"
log "Dataset: $DATASET"
log "Model: $MODEL"
log "Batch Size: $BATCH_SIZE per GPU"
log "Number of GPUs: $NUM_GPUS"
log "Save Directory: $SAVE_DIR"
log "Log File: $LOG_FILE"
log "======================================================================"
log ""
log "Recovery Configuration:"
log "  - Compressor: $COMPRESSOR (ratio: $COMPRESSOR_RATIO)"
log "  - Save Batch Freq: $SAVE_BATCH_FREQ"
log "  - Pipeline Buffer Size: $PIPELINE_BUFFER_SIZE"
log "  - Resume: $RESUME"
log ""
log "Starting pipeline recovery at $(date)..."
log "======================================================================"
log ""

# Record system information
{
    echo ""
    echo "=== System Information ==="
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
    echo ""
    echo "Python Version:"
    python --version 2>&1
    echo ""
    echo "PyTorch Version:"
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>&1
    echo ""
    echo "DeepSpeed Version:"
    python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" 2>&1
    echo ""
    echo "======================================================================"
    echo ""
} >> $LOG_FILE

# Build command line arguments
CMD_ARGS="--dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --compressor $COMPRESSOR \
  --compressor_ratio $COMPRESSOR_RATIO \
  --freq $FREQ \
  --save-batch-freq $SAVE_BATCH_FREQ \
  --save-dir $SAVE_DIR \
  --resume $RESUME \
  --diff \
  --pipeline-buffer-size $PIPELINE_BUFFER_SIZE"

# Distributed training with DeepSpeed
log "Starting DeepSpeed pipeline recovery with $NUM_GPUS GPUs..."
deepspeed --num_gpus=$NUM_GPUS ./torch/pipeline_recovery.py $CMD_ARGS 2>&1 | tee -a $LOG_FILE

# Check exit status
EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline recovery completed successfully at $(date)"
    echo "======================================================================"
    echo ""
    echo "To view recovery performance:"
    echo "  grep '\[PipelineRecovery\]' $LOG_FILE"
    echo "  grep '\[PERF\]' $LOG_FILE"
else
    echo "Pipeline recovery failed with exit code $EXIT_CODE"
    echo "======================================================================"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check log file: $LOG_FILE"
    echo "  2. Verify checkpoints exist in: $SAVE_DIR"
    echo "  3. Verify GPU availability: nvidia-smi"
    echo "======================================================================"
    exit $EXIT_CODE
fi

# Append completion status to log file
{
    echo ""
    echo "======================================================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "PIPELINE RECOVERY COMPLETED SUCCESSFULLY"
    else
        echo "PIPELINE RECOVERY FAILED (Exit Code: $EXIT_CODE)"
    fi
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
} >> $LOG_FILE
