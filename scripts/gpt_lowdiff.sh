#!/bin/zsh

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=wikitext-2
MODEL=gpt2-large
EPOCHS=1
BATCH_SIZE=4
COMPRESSOR=topk
COMPRESSOR_RATIO=0.01
FREQ=50
SAVE_BATCH_FREQ=20
SAVE_DIR=/mnt/newdisk/xiekunpeng/LowDiff/data/lowdiff
RESUME=0
NUM_GPUS=4

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

# Log file
LOG_FILE=$SAVE_DIR/gpt_lowdiff_$(date +%Y%m%d_%H%M%S).log

echo "======================================================"
echo "LowDiff GPT-2 Training"
echo "======================================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE per GPU"
echo "Compression Ratio: $COMPRESSOR_RATIO"
echo "Full Checkpoint Frequency: every $FREQ iterations"
echo "Differential Checkpoint Frequency: every $SAVE_BATCH_FREQ iterations"
echo "Number of GPUs: $NUM_GPUS"
echo "Save Directory: $SAVE_DIR"
echo "Log File: $LOG_FILE"
echo "======================================================"
echo ""

# Distributed training with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS ./torch/GPT.py \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --compressor $COMPRESSOR \
  --compressor_ratio $COMPRESSOR_RATIO \
  --diff \
  --freq $FREQ \
  --save-batch-freq $SAVE_BATCH_FREQ \
  --save-dir $SAVE_DIR \
  --resume $RESUME \
  2>&1 | tee -a $LOG_FILE

echo ""
echo "======================================================"
echo "Training completed at $(date)"
echo "Log saved to: $LOG_FILE"
echo "======================================================"
