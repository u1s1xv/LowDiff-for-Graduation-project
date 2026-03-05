"""Parallel decompression worker module for checkpoint recovery."""

import os
import time
import torch
import torch.multiprocessing as mp

os.environ['DEEPSPEED_DISABLE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


def topk_decompress(values, indices, shape, device=None):
    """
    Decompress Top-K compressed gradients.

    Args:
        values: Compressed values (CPU or GPU tensor)
        indices: Compressed indices (CPU or GPU tensor)
        shape: Original tensor shape
        device: Target device (e.g., 'cuda:0'), None for default CUDA device

    Returns:
        Decompressed tensor on specified device
    """
    if device is None:
        device = torch.cuda.current_device()

    tensor_decompressed = torch.zeros(shape, device=device).view(-1)

    if isinstance(values, list):
        for idx_tensor, val_tensor in zip(indices, values):
            idx_tensor = idx_tensor.to(device) if idx_tensor.device != device else idx_tensor
            val_tensor = val_tensor.to(device) if val_tensor.device != device else val_tensor
            tensor_decompressed.scatter_add_(0, idx_tensor, val_tensor)
    else:
        values = values.to(device) if values.device != device else values
        indices = indices.to(device) if indices.device != device else indices
        tensor_decompressed.scatter_add_(0, indices, values)

    return tensor_decompressed.view(shape)


def parallel_decompress_worker(file_paths, gpu_id, result_queue):
    """
    Parallel decompression worker for checkpoint recovery.

    Args:
        file_paths: List of checkpoints [(idx, filepath, file_batch_idx), ...]
        gpu_id: GPU device ID
        result_queue: Queue to return decompressed gradients

    Returns:
        Via queue: (gpu_id, decompressed_grads_dict)
    """
    import time
    import sys

    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    worker_start = time.time()
    print(f"[GPU {gpu_id}] Worker started, assigned {len(file_paths)} checkpoints", flush=True)
    sys.stdout.flush()

    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    cuda_init_time = time.time()
    print(f"[GPU {gpu_id}] CUDA initialized (+{cuda_init_time - worker_start:.3f}s)", flush=True)
    sys.stdout.flush()

    load_start = time.time()
    checkpoint_data = {}
    loaded_batch_files = {}

    for idx, filepath, _ in file_paths:
        if filepath not in loaded_batch_files:
            if not os.path.exists(filepath):
                print(f"[GPU {gpu_id}] WARNING: {filepath} not found", flush=True)
                continue

            try:
                file_load_start = time.time()
                loaded_batch_files[filepath] = torch.load(filepath, map_location='cpu')
                file_load_end = time.time()
                print(f"[GPU {gpu_id}] Loaded {filepath} ({file_load_end - file_load_start:.3f}s)", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR loading {filepath}: {e}", flush=True)
                continue

        batch_data = loaded_batch_files[filepath]
        if idx in batch_data:
            checkpoint_data[idx] = batch_data[idx]
        else:
            print(f"[GPU {gpu_id}] WARNING: Checkpoint {idx} not in {filepath}", flush=True)

    load_end = time.time()
    print(f"[GPU {gpu_id}] Loaded {len(checkpoint_data)} checkpoints ({load_end - load_start:.3f}s)", flush=True)
    sys.stdout.flush()

    if len(checkpoint_data) == 0:
        print(f"[GPU {gpu_id}] ERROR: No checkpoints loaded, aborting", flush=True)
        result_queue.put((gpu_id, None))
        return

    decompress_start = time.time()
    decompressed_grads = {}
    checkpoint_keys = sorted(checkpoint_data.keys())

    with torch.no_grad():
        for i, ckpt_idx in enumerate(checkpoint_keys):
            iter_start = time.time()
            ckpt = checkpoint_data[ckpt_idx]
            decompressed_grads[ckpt_idx] = {}

            for param_name in ckpt.keys():
                tensor_gpu = topk_decompress(
                    ckpt[param_name]['values'],
                    ckpt[param_name]['indices'],
                    ckpt[param_name]['shape'],
                    device=device
                )
                decompressed_grads[ckpt_idx][param_name] = tensor_gpu.cpu()

            iter_end = time.time()
            if (i + 1) % 3 == 0 or i == len(checkpoint_keys) - 1:
                print(f"[GPU {gpu_id}] Decompressed {i+1}/{len(checkpoint_keys)} ({iter_end - iter_start:.3f}s)", flush=True)
                sys.stdout.flush()

    decompress_end = time.time()
    print(f"[GPU {gpu_id}] Decompression complete ({decompress_end - decompress_start:.3f}s)", flush=True)
    sys.stdout.flush()

    cleanup_start = time.time()
    del checkpoint_data, loaded_batch_files
    torch.cuda.empty_cache()
    cleanup_end = time.time()

    # Save decompressed gradients to temporary file to avoid Queue transfer issues
    save_start = time.time()
    # Use /tmp/lowdiff instead of /tmp to have dedicated directory
    temp_dir = "/mnt/newdisk/xiekunpeng/LowDiff/data/lowdiff/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = f"{temp_dir}/worker_gpu{gpu_id}_pid{os.getpid()}.pt"

    # Retry logic for save failures
    max_retries = 3
    saved = False

    for attempt in range(max_retries):
        try:
            # Add small delay between GPUs to avoid concurrent write conflicts
            time.sleep(gpu_id * 0.1)

            # Save with atomic write operation
            temp_file_tmp = temp_file + ".tmp"
            torch.save(decompressed_grads, temp_file_tmp)
            os.rename(temp_file_tmp, temp_file)

            save_end = time.time()
            print(f"[GPU {gpu_id}] Saved to temp file: {temp_file} ({save_end - save_start:.3f}s)", flush=True)
            sys.stdout.flush()
            saved = True
            break

        except Exception as e:
            print(f"[GPU {gpu_id}] WARNING: Save attempt {attempt+1}/{max_retries} failed: {e}", flush=True)
            sys.stdout.flush()
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                print(f"[GPU {gpu_id}] ERROR: Failed to save results after {max_retries} attempts", flush=True)
                sys.stdout.flush()

    if saved:
        total_time = time.time() - worker_start
        print(f"[GPU {gpu_id}] Worker finished (total: {total_time:.3f}s)", flush=True)
        print(f"[GPU {gpu_id}]   - CUDA init: {cuda_init_time - worker_start:.3f}s", flush=True)
        print(f"[GPU {gpu_id}]   - Load: {load_end - load_start:.3f}s", flush=True)
        print(f"[GPU {gpu_id}]   - Decompress: {decompress_end - decompress_start:.3f}s", flush=True)
        print(f"[GPU {gpu_id}]   - Save: {save_end - save_start:.3f}s", flush=True)
        print(f"[GPU {gpu_id}]   - Cleanup: {cleanup_end - cleanup_start:.3f}s", flush=True)
        sys.stdout.flush()
        result_queue.put((gpu_id, temp_file))
    else:
        result_queue.put((gpu_id, None))


def parallel_decompress_worker_batch(checkpoint_list, gpu_id, result_queue, done_event):
    """
    Batch decompression worker with CUDA IPC shared memory optimization.

    OPTIMIZATION S2: Uses torch.multiprocessing.Queue + CUDA IPC for zero-copy transfer.
    - Eliminates temporary file I/O (previously ~40s overhead)
    - Decompressed gradients sent directly via CUDA IPC handles
    - Worker process waits for main process to complete before exiting

    Args:
        checkpoint_list: List of [(batch_idx, filepath, file_batch_idx), ...]
        gpu_id: GPU device ID
        result_queue: torch.multiprocessing.Queue for CUDA tensor transfer
        done_event: torch.multiprocessing.Event to signal when main process is done

    Returns:
        Via queue: (gpu_id, {batch_idx: {param_name: cuda_tensor}})
    """
    import sys
    import time

    worker_start = time.time()
    print(f"[GPU {gpu_id}] [SHARED-MEM] Worker started, assigned {len(checkpoint_list)} checkpoints", flush=True)
    sys.stdout.flush()

    cuda_init_start = time.time()
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    cuda_init_time = time.time()
    print(f"[GPU {gpu_id}] CUDA initialized ({cuda_init_time - cuda_init_start:.3f}s)", flush=True)
    sys.stdout.flush()

    # Load checkpoint data from files
    load_start = time.time()
    checkpoint_data_dict = {}
    loaded_batch_files = {}

    is_filepath_format = len(checkpoint_list) > 0 and isinstance(checkpoint_list[0][1], str)

    if is_filepath_format:
        print(f"[GPU {gpu_id}] Loading checkpoints from file paths", flush=True)
        sys.stdout.flush()

        for batch_idx, filepath, _ in checkpoint_list:  # file_batch_idx not used
            if filepath not in loaded_batch_files:
                if not os.path.exists(filepath):
                    print(f"[GPU {gpu_id}] WARNING: {filepath} not found", flush=True)
                    continue

                try:
                    file_load_start = time.time()
                    loaded_batch_files[filepath] = torch.load(filepath, map_location='cpu')
                    file_load_end = time.time()
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"[GPU {gpu_id}] Loaded {os.path.basename(filepath)} ({file_size_mb:.2f} MB in {file_load_end - file_load_start:.3f}s)", flush=True)
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[GPU {gpu_id}] ERROR loading {filepath}: {e}", flush=True)
                    continue

            batch_data = loaded_batch_files[filepath]
            if batch_idx in batch_data:
                checkpoint_data_dict[batch_idx] = batch_data[batch_idx]
            else:
                print(f"[GPU {gpu_id}] WARNING: Checkpoint {batch_idx} not in {filepath}", flush=True)

        load_end = time.time()
        print(f"[GPU {gpu_id}] Loaded {len(checkpoint_data_dict)} checkpoints from {len(loaded_batch_files)} files ({load_end - load_start:.3f}s)", flush=True)
        sys.stdout.flush()
    else:
        print(f"[GPU {gpu_id}] Using legacy format (data preloaded)", flush=True)
        for batch_idx, checkpoint_data in checkpoint_list:
            checkpoint_data_dict[batch_idx] = checkpoint_data
        load_end = load_start

    # Decompression phase - keep tensors on GPU!
    decompress_start = time.time()
    decompressed_grads = {}

    with torch.no_grad():
        for i, batch_idx in enumerate(sorted(checkpoint_data_dict.keys())):
            iter_start = time.time()
            checkpoint_data = checkpoint_data_dict[batch_idx]
            decompressed_grads[batch_idx] = {}

            for param_name in checkpoint_data.keys():
                tensor_gpu = topk_decompress(
                    checkpoint_data[param_name]['values'],
                    checkpoint_data[param_name]['indices'],
                    checkpoint_data[param_name]['shape'],
                    device=device
                )
                # OPTIMIZATION S2: Keep tensor on GPU (no .cpu() call)
                decompressed_grads[batch_idx][param_name] = tensor_gpu

            iter_end = time.time()
            if (i + 1) % 3 == 0 or i == len(checkpoint_data_dict) - 1:
                print(f"[GPU {gpu_id}] Decompressed {i+1}/{len(checkpoint_data_dict)} ({iter_end - iter_start:.3f}s)", flush=True)
                sys.stdout.flush()

    decompress_end = time.time()
    print(f"[GPU {gpu_id}] Decompression complete ({decompress_end - decompress_start:.3f}s)", flush=True)
    sys.stdout.flush()

    # OPTIMIZATION S2: Send CUDA tensors via Queue (uses CUDA IPC automatically)
    queue_send_start = time.time()

    try:
        # PyTorch Queue automatically handles CUDA IPC for GPU tensors
        result_queue.put((gpu_id, decompressed_grads))
        queue_send_end = time.time()

        print(f"[GPU {gpu_id}] [SHARED-MEM] Sent {len(decompressed_grads)} checkpoints via CUDA IPC", flush=True)
        print(f"[GPU {gpu_id}] [SHARED-MEM] Queue send time: {queue_send_end - queue_send_start:.3f}s", flush=True)
        sys.stdout.flush()

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: Failed to send results via queue: {e}", flush=True)
        sys.stdout.flush()
        result_queue.put((gpu_id, None))
        return

    # CRITICAL: Wait for main process to finish using the tensors
    # This prevents premature deallocation of GPU memory
    print(f"[GPU {gpu_id}] [SHARED-MEM] Waiting for main process to complete...", flush=True)
    sys.stdout.flush()

    wait_start = time.time()
    done_event.wait()  # Block until main process signals completion
    wait_end = time.time()

    print(f"[GPU {gpu_id}] [SHARED-MEM] Main process completed, safe to exit", flush=True)
    print(f"[GPU {gpu_id}] [SHARED-MEM] Wait time: {wait_end - wait_start:.3f}s", flush=True)
    sys.stdout.flush()

    # Cleanup
    cleanup_start = time.time()
    del decompressed_grads, checkpoint_data_dict, loaded_batch_files
    torch.cuda.empty_cache()
    cleanup_end = time.time()

    total_time = time.time() - worker_start
    print(f"[GPU {gpu_id}] [SHARED-MEM] Worker finished (total: {total_time:.3f}s)", flush=True)
    print(f"[GPU {gpu_id}]   - CUDA init: {cuda_init_time - cuda_init_start:.3f}s", flush=True)
    if is_filepath_format:
        print(f"[GPU {gpu_id}]   - Load: {load_end - load_start:.3f}s", flush=True)
    print(f"[GPU {gpu_id}]   - Decompress: {decompress_end - decompress_start:.3f}s", flush=True)
    print(f"[GPU {gpu_id}]   - Queue send: {queue_send_end - queue_send_start:.3f}s ⚡", flush=True)
    print(f"[GPU {gpu_id}]   - Wait for main: {wait_end - wait_start:.3f}s", flush=True)
    print(f"[GPU {gpu_id}]   - Cleanup: {cleanup_end - cleanup_start:.3f}s", flush=True)
    sys.stdout.flush()
