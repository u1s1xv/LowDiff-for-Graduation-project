"""
Lightweight worker module for parallel checkpoint merging.

This module contains only the minimal dependencies required for merging checkpoints,
avoiding heavy imports like transformers, deepspeed, etc.
"""

import os
import torch

os.environ['DEEPSPEED_DISABLE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


def topk_decompress(values, indices, shape):
    """
    Decompress Top-K compressed gradients back to full gradient tensor.
    Handles both single tensor and list of tensors (multi-GPU case).
    """
    tensor_decompressed = torch.zeros(shape).cuda().view(-1)

    if isinstance(values, list):
        for idx_tensor, val_tensor in zip(indices, values):
            idx_tensor = idx_tensor.cuda() if not idx_tensor.is_cuda else idx_tensor
            val_tensor = val_tensor.cuda() if not val_tensor.is_cuda else val_tensor
            tensor_decompressed.scatter_add_(0, idx_tensor, val_tensor)
    else:
        values = values.cuda() if not values.is_cuda else values
        indices = indices.cuda() if not indices.is_cuda else indices
        tensor_decompressed.scatter_add_(0, indices, values)

    return tensor_decompressed.view(shape)


def merge_two_checkpoints(ckpt1_data, ckpt2_data, compressor_ratio, device_id):
    """
    Merge two checkpoints in a worker process.

    Args:
        ckpt1_data (dict): First checkpoint data {param_name: {values, indices, shape}}
        ckpt2_data (dict): Second checkpoint data {param_name: {values, indices, shape}}
        compressor_ratio (float): Compression ratio for Top-K
        device_id (int): GPU device ID to use for computation

    Returns:
        dict: Merged checkpoint data (on CPU to avoid GPU memory accumulation)
    """
    merged_data = {}

    with torch.cuda.device(device_id):
        for key in ckpt1_data.keys():
            tensor1 = topk_decompress(
                ckpt1_data[key]['values'],
                ckpt1_data[key]['indices'],
                ckpt1_data[key]['shape']
            )
            tensor2 = topk_decompress(
                ckpt2_data[key]['values'],
                ckpt2_data[key]['indices'],
                ckpt2_data[key]['shape']
            )

            merged_tensor = tensor1 + tensor2

            flat_tensor = merged_tensor.view(-1)
            k = int(flat_tensor.numel() * compressor_ratio)
            values, indices = torch.topk(flat_tensor.abs(), k)
            values = flat_tensor[indices]

            merged_data[key] = {
                'values': values.cpu(),
                'indices': indices.cpu(),
                'shape': merged_tensor.shape
            }

            del tensor1, tensor2, merged_tensor, flat_tensor
            torch.cuda.empty_cache()

    return merged_data


def queue_worker(task_queue, result_queue, device_id):
    """
    Worker process that reads tasks from queue, merges checkpoints, and writes results back.

    Args:
        task_queue (multiprocessing.Queue): Input queue for merge tasks
        result_queue (multiprocessing.Queue): Output queue for merge results
        device_id (int): GPU device ID to use for computation
    """
    torch.cuda.set_device(device_id)

    actual_device = torch.cuda.current_device()
    print(f"Worker started: assigned GPU {device_id}, actual GPU {actual_device}")

    if actual_device != device_id:
        print(f"WARNING: Device mismatch! Expected {device_id}, got {actual_device}")

    while True:
        task = task_queue.get()

        if task is None:
            break

        task_id, ckpt1_data, ckpt2_data, compressor_ratio = task

        merged_data = merge_two_checkpoints(
            ckpt1_data,
            ckpt2_data,
            compressor_ratio,
            device_id
        )

        result_queue.put((task_id, merged_data))
