#!/bin/zsh

################################################################################
# LowDiff Training Launcher (With Compression & Differential Checkpoints)
# Purpose: Run GPT-2 training with gradient compression and differential checkpoints
# Usage: bash scripts/gpt_lowdiff.sh
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
EPOCHS=1                    # 测试：只训练1个epoch
BATCH_SIZE=4
COMPRESSOR=topk
COMPRESSOR_RATIO=0.01
FREQ=100                    # 最大间隔保护（智能检查点会复用此参数）
SAVE_BATCH_FREQ=20          # 保留参数（兼容性）
SAVE_DIR=/mnt/newdisk/xiekunpeng/LowDiff/data/lowdiff
RESUME=0
NUM_GPUS=4
ENABLE_SMART_CHECKPOINT=0  # 测试：禁用智能检查点，专注测试延迟写入

# Optimizer Monitoring (Hardware Fault Detection)
ENABLE_OPTIMIZER_MONITORING=0  # 启用优化器监控（1=启用，0=禁用）
MONITORING_SAFETY_FACTOR=1.0   # 安全系数（>1.0更保守，默认1.0）
INJECT_FAULT=0                 # 注入故障测试（1=启用，0=禁用，仅用于测试）
INJECT_FAULT_AT_BATCH=50       # 在第N个batch注入故障

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

# Log file
LOG_FILE=$SAVE_DIR/gpt_lowdiff_$(date +%Y%m%d_%H%M%S).log

# Initialize log file with header
{
    echo "======================================================================"
    echo "LOWDIFF TRAINING LOG"
    echo "======================================================================"
    echo "Script: gpt_lowdiff.sh"
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
log "LowDiff Training (With Compression & Differential Checkpoints)"
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
log "LowDiff Configuration:"
log "  - Compressor: $COMPRESSOR"
log "  - Compression Ratio: $COMPRESSOR_RATIO"
log "  - Full Checkpoint Frequency: every $FREQ iterations (max interval)"
log "  - Differential Checkpoint Frequency: every $SAVE_BATCH_FREQ iterations"
log "  - Smart Checkpoint: $([ $ENABLE_SMART_CHECKPOINT -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
log ""
log "Fault Tolerance Configuration:"
log "  - Optimizer Monitoring: $([ $ENABLE_OPTIMIZER_MONITORING -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
if [ $ENABLE_OPTIMIZER_MONITORING -eq 1 ]; then
    log "  - Safety Factor: $MONITORING_SAFETY_FACTOR"
    log "  - Fault Injection: $([ $INJECT_FAULT -eq 1 ] && echo 'ENABLED (batch '$INJECT_FAULT_AT_BATCH')' || echo 'DISABLED')"
fi
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
  --diff"

# Add smart checkpoint if enabled
if [ $ENABLE_SMART_CHECKPOINT -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --enable-smart-checkpoint"
fi

# Add optimizer monitoring if enabled
if [ $ENABLE_OPTIMIZER_MONITORING -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --enable-optimizer-monitoring --monitoring-safety-factor $MONITORING_SAFETY_FACTOR"
fi

# Add fault injection if enabled (for testing only)
if [ $INJECT_FAULT -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --inject-fault --inject-fault-at-batch $INJECT_FAULT_AT_BATCH"
fi

# Distributed training with DeepSpeed
log "Starting DeepSpeed training with $NUM_GPUS GPUs..."
deepspeed --num_gpus=$NUM_GPUS ./torch/GPT.py $CMD_ARGS 2>&1 | tee -a $LOG_FILE

# Check exit status
EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ LowDiff training completed successfully at $(date)"
    echo "======================================================================"
    echo ""
    echo "Performance Summary:"
    echo "-------------------------------------------------------------------"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "To view training statistics:"
    echo "  grep 'TRAINING COMPLETED' -A 20 $LOG_FILE"
    echo ""
    echo "To compare with baseline training:"
    echo "  1. Run baseline: bash scripts/baseline_lowdiff.sh"
    echo "  2. Compare logs in: $SAVE_DIR"
    echo "======================================================================"
else
    echo "❌ LowDiff training failed with exit code $EXIT_CODE"
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