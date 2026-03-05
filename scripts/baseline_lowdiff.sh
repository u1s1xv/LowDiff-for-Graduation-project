#!/bin/zsh

################################################################################
# Baseline Training Launcher (No Compression, No Checkpoints)
# Purpose: Run standard GPT-2 training for performance comparison with LowDiff
# Usage: bash scripts/baseline_lowdiff.sh
################################################################################

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Redirect all caches to /mnt/newdisk (avoid root partition full)
export HF_HOME=/mnt/newdisk/xiekunpeng/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/newdisk/xiekunpeng/.cache/huggingface/datasets
export TORCH_HOME=/mnt/newdisk/xiekunpeng/.cache/torch
export TRITON_CACHE_DIR=/mnt/newdisk/xiekunpeng/.cache/triton
export TRANSFORMERS_CACHE=/mnt/newdisk/xiekunpeng/.cache/huggingface/transformers

# Training parameters
DATASET=wikitext-2
MODEL=gpt2-large
EPOCHS=1
BATCH_SIZE=4
NUM_GPUS=4

# Save directory (same as gpt_lowdiff.sh)
SAVE_DIR=/mnt/newdisk/xiekunpeng/LowDiff/data/lowdiff

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

# Log file (same naming convention as gpt_lowdiff.sh)
LOG_FILE=$SAVE_DIR/baseline_lowdiff_$(date +%Y%m%d_%H%M%S).log

# Initialize log file with header
{
    echo "======================================================================"
    echo "BASELINE TRAINING LOG"
    echo "======================================================================"
    echo "Script: baseline_lowdiff.sh"
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

# Display training configuration
log "======================================================================"
log "Baseline Training (No Compression, No Checkpoints)"
log "======================================================================"
log "Dataset: $DATASET"
log "Model: $MODEL"
log "Batch Size: $BATCH_SIZE per GPU"
log "Total Batch Size: $((BATCH_SIZE * NUM_GPUS))"
log "Epochs: $EPOCHS"
log "Number of GPUs: $NUM_GPUS"
log "Save Directory: $SAVE_DIR"
log "Log File: $LOG_FILE"
log "======================================================================"
log ""
log "NOTE: This is a BASELINE training WITHOUT:"
log "  - Gradient compression (no topk_compress)"
log "  - Checkpoint saving (no Communicator)"
log "  - Differential checkpoints"
log ""
log "Starting training at $(date)..."
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

# Distributed training with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS ./torch/baseline_training.py \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  2>&1 | tee -a $LOG_FILE

# Check exit status
EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Baseline training completed successfully at $(date)"
    echo "======================================================================"
    echo ""
    echo "Performance Summary:"
    echo "-------------------------------------------------------------------"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "To view training statistics:"
    echo "  grep 'TRAINING COMPLETED' -A 20 $LOG_FILE"
    echo ""
    echo "To compare with LowDiff training:"
    echo "  1. Run LowDiff: bash scripts/gpt_lowdiff.sh"
    echo "  2. Compare logs in: $SAVE_DIR"
    echo "======================================================================"
else
    echo "❌ Baseline training failed with exit code $EXIT_CODE"
    echo "======================================================================"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check log file: $LOG_FILE"
    echo "  2. Verify GPU availability: nvidia-smi"
    echo "  3. Check DeepSpeed installation: pip show deepspeed"
    echo "======================================================================"
    exit $EXIT_CODE
fi

# Append completion status to log file
{
    echo ""
    echo "======================================================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "TRAINING COMPLETED SUCCESSFULLY"
    else
        echo "TRAINING FAILED (Exit Code: $EXIT_CODE)"
    fi
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
} >> $LOG_FILE

