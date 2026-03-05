import os
import sys
import time
import argparse
import re
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import deepspeed
from deepspeed import comm as dist

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    set_seed
)

from datasets import load_dataset

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from communicator.lowdiff import Communicator


parser = argparse.ArgumentParser(description='DeepSpeed NLP Training with TopK Compression')
parser.add_argument('--dataset', default='wikitext-2', type=str, help='dataset name')
parser.add_argument('--model', default='gpt2', type=str, help='model architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='batch size per GPU')
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--workers', default=1, type=int, help='data loading workers')
parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
parser.add_argument('--compress_ratio', default=0.01, type=float, help='TopK compression ratio')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
parser.add_argument("--compressor", default="topk", type=str, help='which compressor to use')
parser.add_argument("--compressor_ratio", default=0.01, type=float, help='compress ratio for compressor')
parser.add_argument("--save-dir", default='/data/lowdiff', type=str, help='directory to save checkpoints')
parser.add_argument("--resume", type=int, default=0, help='resume from checkpoint')
parser.add_argument("--diff", action="store_true", help='use differential checkpoint')
parser.add_argument("--freq", default=0, type=int, help='full checkpoint saving frequency')
parser.add_argument("--save-batch-freq", default='1', type=int, help='in-memory batching frequency')
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
args = parser.parse_args()


def main():
    model_path = "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/openai-community/" + args.model

    deepspeed.init_distributed()
    dist.barrier()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    set_seed(42 + rank)
    torch.cuda.set_device(args.local_rank)
    print(f"[Rank {rank}/{world_size}] Initialized DeepSpeed")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_length,
            padding="max_length"
        )

    if args.dataset == 'wikitext-103':
        dataset = load_dataset("/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103",
                        data_files={
                            "train": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103/train.txt",
                            "validation": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103/valid.txt",
                            "test": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103/test.txt"
                        })["train"]
    elif args.dataset == 'wikitext-2':
        dataset = load_dataset("/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-2",
                        data_files={
                            "train": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-2/train.txt",
                            "validation": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-2/valid.txt",
                            "test": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-2/test.txt"
                        })["train"]
    else:
        raise ValueError("Incorrect dataset Name")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=12
    )

    print("Dataset map successfully.")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_sampler = DistributedSampler(
        tokenized_dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=4
    )

    print("Loading model...")
    if args.model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained("/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/openai-community/gpt2")
    elif args.model == 'gpt2-medium':
        model = GPT2LMHeadModel.from_pretrained("/data/dataset/nlp/openai-community/gpt2-medium")
    elif args.model == 'gpt2-large':
        model = GPT2LMHeadModel.from_pretrained("/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/openai-community/gpt2-large")
    else:
        print("Model loaded fail.")
    model.gradient_checkpointing_enable()
    model.cuda()
    print("Model loaded successfully.")

    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-4,
                "weight_decay": 0.01
            }
        },
    }
    model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    resume_epoch = 0
    resume_batch = 0
    last_trained_batch = 0
    if args.resume and dist.get_rank() == 0:
        model, optimizer, resume_epoch, resume_batch = load_base_checkpoint(model, optimizer)

        print(f"Base checkpoint loaded: epoch {resume_epoch}, batch {resume_batch}")
        print(f"Will replay differential checkpoints from batch {resume_batch + 1} onwards")

        if args.save_batch_freq > 1:
            model, optimizer, last_trained_batch = load_batch_differential_checkpoint(model, optimizer, resume_batch)
        else:
            model, optimizer, last_trained_batch = load_differential_checkpoint(model, optimizer, resume_batch)

        print(f"Differential checkpoint replay completed")
        print(f"Last trained batch: {last_trained_batch}")
        print(f"Training will resume from epoch {resume_epoch}, batch {last_trained_batch + 1}")

    deepspeed.enable_backward_allreduce = False

    communicator = Communicator(model, k=args.compress_ratio, save_batch_freq=args.save_batch_freq)
    communicator.register_hooks()

    start_epoch = resume_epoch if args.resume else 0

    training_completed = False
    if args.resume and dist.get_rank() == 0:
        if resume_epoch >= args.epochs - 1:
            if last_trained_batch == resume_batch:
                print(f"No new differential checkpoints found after base checkpoint.")
                print(f"Training appears to be complete at epoch {resume_epoch}, batch {resume_batch}")
                training_completed = True
            else:
                print(f"Starting training from epoch {start_epoch} (total epochs: {args.epochs})")
        else:
            print(f"Starting training from epoch {start_epoch} (total epochs: {args.epochs})")

    if training_completed:
        print("Training already completed. Skipping training loop.")
    else:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_loader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader):
                if args.resume and epoch == resume_epoch and batch_idx <= last_trained_batch:
                    if dist.get_rank() == 0 and batch_idx % 10 == 0:
                        print(f"[Epoch {epoch}] Skipping batch {batch_idx} (already trained)")
                    continue

                end = time.time()
                inputs = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss

                model.backward(loss)
                communicator.decompress_save(args.diff, '{}/{}_{}_{}_{}_{}-{}_batch{}.pth.tar'.format(args.save_dir,args.model,args.dataset,args.compressor,args.compressor_ratio,epoch,batch_idx,args.save_batch_freq), batch_idx)
                model.step()

                if dist.get_rank() == 0:
                    print("[Epoch {}/{}] Batch {}, Loss: {:.3f}, Time: {:.3f}"
                        .format(epoch, args.epochs, batch_idx, loss.item(), time.time() - end))

                if dist.get_rank() == 0 and args.freq > 0 and batch_idx % args.freq == 0:
                            begin_full = time.time()
                            torch.save({
                                'epoch': epoch + 1,
                                'model': model.module.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                            }, '{}/{}_{}_{}_{}_{}_{}_full.pth.tar'.format(args.save_dir,args.model,args.dataset,args.compressor,args.compressor_ratio,epoch,batch_idx))
                            end_full = time.time()
                            print("base checkpoint takes {:.3f}s".format(end_full - begin_full))

                end = time.time()

            print(f"Epoch {epoch} completed.")


def load_base_checkpoint(model, optimizer):
    """Load the latest full checkpoint."""
    start = time.time()
    filedir = args.save_dir
    pattern = r'{}_{}_{}_{}_([0-9]+)_([0-9]+)_full\.pth\.tar'.format(args.model, args.dataset, args.compressor, args.compressor_ratio)
    files = os.listdir(filedir)
    candidates = []
    for f in files:
        m = re.match(pattern, f)
        if m:
            epoch = int(m.group(1))
            batch = int(m.group(2))
            candidates.append((epoch, batch, f))
    if not candidates:
        raise ValueError("No full checkpoint found in {}".format(filedir))
    candidates.sort(key=lambda x: (x[0], x[1]))
    sel_epoch, sel_batch, sel_file = candidates[-1]
    filepath = os.path.join(filedir, sel_file)
    print("Loading {}".format(filepath))
    checkpoint = torch.load(filepath, map_location='cpu')

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.resume = sel_epoch + 1
    end = time.time()
    print("Base checkpoint loaded in {:.3f}s (epoch {}, batch {})".format(end - start, sel_epoch, sel_batch))
    return model, optimizer, sel_epoch, sel_batch


def topk_decompress(values, indices, shape):
    """Decompress Top-K compressed gradients."""
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


def apply_merged_checkpoint(model, optimizer, merged_data):
    """
    Apply merged checkpoint to model and optimizer.
    """
    _parameter_names = {name: param for name, param in model.named_parameters()}
    for key in merged_data.keys():
        tensor = topk_decompress(
            merged_data[key]['values'],
            merged_data[key]['indices'],
            merged_data[key]['shape']
        )
        param = _parameter_names.get(key)
        if param is not None:
            param.grad = tensor
    optimizer.step()

def find_max(base_batch):
    """
    Find the maximum iteration number of differential checkpoints AFTER base_batch.

    Args:
        base_batch (int): The batch number of the base checkpoint.
    """
    files = os.listdir(args.save_dir)
    if args.save_batch_freq > 1:
        pattern = r'{}_{}_{}_{}_{}-(\d+)_batch{}\.pth\.tar'.format(
            args.model, args.dataset, args.compressor, args.compressor_ratio, 
            args.resume-1, args.save_batch_freq
        )
    else:
        pattern = r'{}_{}_{}_{}_{}-(\d+)_batch1\.pth\.tar'.format(
            args.model, args.dataset, args.compressor, args.compressor_ratio, 
            args.resume-1
        )
    
    max_x = -1
    for file in files:
        match = re.match(pattern, file)
        if match:
            x = int(match.group(1))
            if x > base_batch and x > max_x:
                max_x = x
    
    if max_x != -1:
        print("Max diff ckpt at epoch {}, iteration {} (after base batch {})".format(
            args.resume, max_x, base_batch
        ))
    else:
        print("No diff ckpt found after base batch {}".format(base_batch))
    
    return max_x


def assign_checkpoints_to_gpus(checkpoint_list, max_gpus=4):
    """
    Distribute checkpoints evenly across available GPUs.

    Args:
        checkpoint_list: List of checkpoints to distribute
        max_gpus: Maximum number of GPUs to use (default: 4)

    Returns:
        tuple: (gpu_assignments, num_gpus)
            - gpu_assignments: List of checkpoint lists for each GPU
            - num_gpus: Number of GPUs used
    """
    num_gpus = min(torch.cuda.device_count(), max_gpus)
    checkpoints_per_gpu = len(checkpoint_list) // num_gpus
    remainder = len(checkpoint_list) % num_gpus

    gpu_assignments = []
    start_idx = 0

    print(f"\nUsing {num_gpus} GPUs for parallel decompression")

    for gpu_id in range(num_gpus):
        # First 'remainder' GPUs get one extra checkpoint
        count = checkpoints_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + count

        gpu_assignments.append(checkpoint_list[start_idx:end_idx])

        if len(gpu_assignments[gpu_id]) > 0:
            first_batch = checkpoint_list[start_idx][0]
            last_batch = checkpoint_list[end_idx - 1][0]
            print(f"GPU {gpu_id}: {count} checkpoints (batch {first_batch}-{last_batch})")

        start_idx = end_idx

    return gpu_assignments, num_gpus


def run_parallel_workers(worker_func, gpu_assignments, num_gpus):
    """
    Run parallel decompression workers on multiple GPUs.

    Args:
        worker_func: Worker function to execute (parallel_decompress_worker or parallel_decompress_worker_batch)
        gpu_assignments: List of checkpoint assignments for each GPU
        num_gpus: Number of GPUs to use

    Returns:
        tuple: (worker_results, elapsed_time)
            - worker_results: Dictionary mapping gpu_id to decompressed gradients
            - elapsed_time: Time taken for parallel decompression
    """
    mp_ctx = mp.get_context('spawn')
    result_queue = mp_ctx.Queue()
    processes = []
    start_time = time.time()

    # Start workers
    for gpu_id in range(num_gpus):
        if len(gpu_assignments[gpu_id]) == 0:
            continue

        p = mp_ctx.Process(
            target=worker_func,
            args=(gpu_assignments[gpu_id], gpu_id, result_queue)
        )
        p.start()
        processes.append(p)
        print(f"Worker GPU {gpu_id} started (PID: {p.pid})")

    print(f"\nWaiting for {len(processes)} workers to complete...")

    # Collect results
    worker_results = {}
    temp_files = []

    for _ in range(len(processes)):
        try:
            # Add timeout to prevent infinite blocking
            gpu_id, result = result_queue.get(timeout=600)  # 10 minutes timeout

            if result is None:
                print(f"ERROR: GPU {gpu_id} worker failed")
                continue

            # Result is now a file path
            if isinstance(result, str):
                print(f"GPU {gpu_id} completed, loading results from {result}")
                try:
                    decompressed_grads = torch.load(result, map_location='cpu')
                    worker_results[gpu_id] = decompressed_grads
                    temp_files.append(result)
                    print(f"GPU {gpu_id} loaded {len(decompressed_grads)} checkpoints from file")
                except Exception as e:
                    print(f"ERROR: GPU {gpu_id} failed to load results from {result}: {e}")
            else:
                # Fallback for old behavior (direct data transfer)
                worker_results[gpu_id] = result
                print(f"GPU {gpu_id} completed, received {len(result)} checkpoints directly")

        except Exception as e:
            print(f"ERROR: Failed to receive results from worker: {e}")
            import traceback
            traceback.print_exc()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Clean up temporary files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            print(f"WARNING: Failed to remove temp file {temp_file}: {e}")

    elapsed_time = time.time() - start_time
    return worker_results, elapsed_time


def apply_gradients_serially(model, optimizer, worker_results, num_gpus):
    """
    Apply decompressed gradients to model in serial order.

    This function merges gradients from all workers and applies them
    sequentially to ensure correct model state recovery.

    Args:
        model: Model object to apply gradients to
        optimizer: Optimizer object for gradient application
        worker_results: Dictionary of decompressed gradients from workers
        num_gpus: Number of GPUs used

    Returns:
        elapsed_time: Time taken for serial gradient application
    """
    start_time = time.time()

    # Merge all worker results
    all_grads = {}
    for gpu_id in range(num_gpus):
        if gpu_id in worker_results:
            all_grads.update(worker_results[gpu_id])

    print(f"Total checkpoints to apply: {len(all_grads)}")

    # Build parameter name to parameter mapping
    param_name_to_param = {name: param for name, param in model.named_parameters()}

    # Sort checkpoint keys to ensure gradients are applied in correct order
    checkpoint_keys = sorted(all_grads.keys())

    for i, ckpt_idx in enumerate(checkpoint_keys):
        iter_start = time.time()

        # Clear gradients first to prevent gradient accumulation (P0 FIX)
        optimizer.zero_grad()

        # Set gradients from checkpoint
        for param_name, tensor_cpu in all_grads[ckpt_idx].items():
            param = param_name_to_param.get(param_name)
            if param is not None:
                param.grad = tensor_cpu.cuda()

        # Apply gradients
        optimizer.step()

        iter_end = time.time()

        if (i + 1) % 5 == 0 or i == len(checkpoint_keys) - 1:
            print(f"Applied checkpoint {i+1}/{len(checkpoint_keys)} (batch {ckpt_idx}, {iter_end - iter_start:.3f}s)")

    elapsed_time = time.time() - start_time
    return elapsed_time


def load_differential_checkpoint(model, optimizer, base_batch):
    """
    Load differential checkpoints using parallel decompression.

    Args:
        model: Model object
        optimizer: Optimizer object
        base_batch: Base checkpoint batch number

    Returns:
        model, optimizer, last_batch
    """
    print(f"\n{'='*80}")
    print(f"Parallel recovery started (parallel decompression + serial application)")
    print(f"{'='*80}\n")

    recovery_start = time.time()
    filedir = args.save_dir
    max_batch = find_max(base_batch)

    if max_batch == -1:
        print("WARNING: No differential checkpoints found")
        return model, optimizer, base_batch

    print(f"Found checkpoints from batch {base_batch + 1} to {max_batch}")
    print(f"Total checkpoints to recover: {max_batch - base_batch}")

    # Build checkpoint file list
    checkpoint_list = []
    for i in range(base_batch + 1, max_batch + 1):
        filepath = f"{filedir}/{args.model}_{args.dataset}_{args.compressor}_{args.compressor_ratio}_{args.resume-1}-{i}_batch1.pth.tar"
        if os.path.exists(filepath):
            checkpoint_list.append((i, filepath, i))
        else:
            print(f"WARNING: Checkpoint file not found: {filepath}")

    if len(checkpoint_list) == 0:
        print("ERROR: No valid checkpoint files found")
        return model, optimizer, base_batch

    print(f"Valid checkpoint files: {len(checkpoint_list)}")

    # Assign checkpoints to GPUs (P1: Using new helper function)
    gpu_assignments, num_gpus = assign_checkpoints_to_gpus(checkpoint_list)

    # Parallel decompression (P1: Using new helper function)
    from communicator.merge_worker import parallel_decompress_worker
    worker_results, parallel_time = run_parallel_workers(
        parallel_decompress_worker, gpu_assignments, num_gpus
    )
    print(f"\nParallel decompression completed in {parallel_time:.3f}s")

    # Serial gradient application (P1: Using new helper function with P0 fix)
    apply_time = apply_gradients_serially(model, optimizer, worker_results, num_gpus)
    print(f"\nSerial application completed in {apply_time:.3f}s")

    # Summary
    recovery_end = time.time()
    print(f"\n{'='*80}")
    print(f"Parallel recovery completed")
    print(f"{'='*80}")
    print(f"Total time: {recovery_end - recovery_start:.3f}s")
    print(f"  - Parallel decompression: {parallel_time:.3f}s")
    print(f"  - Serial application: {apply_time:.3f}s")
    print(f"{'='*80}\n")

    return model, optimizer, max_batch


def load_batch_differential_checkpoint(model, optimizer, base_batch):
    """
    Load batched differential checkpoints using parallel decompression.

    Args:
        model: Model object
        optimizer: Optimizer object
        base_batch: Base checkpoint batch number

    Returns:
        model, optimizer, last_batch
    """
    print(f"\n{'='*80}")
    print(f"Batch parallel recovery started (parallel decompression + serial application)")
    print(f"{'='*80}\n")

    recovery_start = time.time()
    filedir = args.save_dir
    max_batch = find_max(base_batch)

    if max_batch == -1:
        print("WARNING: No differential checkpoints found")
        return model, optimizer, base_batch

    print(f"Found checkpoints from batch {base_batch + 1} to {max_batch}")

    first_batch = ((base_batch // args.save_batch_freq) + 1) * args.save_batch_freq - 1
    load_start = time.time()

    batch_files = []
    for i in range(first_batch, max_batch + 1, args.save_batch_freq):
        filepath = f"{filedir}/{args.model}_{args.dataset}_{args.compressor}_{args.compressor_ratio}_{args.resume-1}-{i}_batch{args.save_batch_freq}.pth.tar"
        if os.path.exists(filepath):
            batch_files.append((i, filepath))
        else:
            print(f"WARNING: Batch file not found: {filepath}")

    if len(batch_files) == 0:
        print("ERROR: No valid batch files found")
        return model, optimizer, base_batch

    print(f"Found {len(batch_files)} batch files to load")

    checkpoint_list = []
    batch_files_loaded = 0

    for file_batch_idx, filepath in batch_files:
        tensor_compressed = torch.load(filepath, map_location='cpu')
        batch_files_loaded += 1

        for j in range(file_batch_idx - args.save_batch_freq + 1, file_batch_idx + 1):
            if j <= base_batch or j not in tensor_compressed:
                continue
            checkpoint_list.append((j, tensor_compressed[j]))

    load_end = time.time()
    print(f"Loaded {batch_files_loaded} batch files with {len(checkpoint_list)} checkpoints in {load_end - load_start:.3f}s")

    if len(checkpoint_list) == 0:
        print("ERROR: No valid checkpoints extracted from batch files")
        return model, optimizer, base_batch

    # Assign checkpoints to GPUs (P1: Using new helper function)
    gpu_assignments, num_gpus = assign_checkpoints_to_gpus(checkpoint_list)

    # Parallel decompression (P1: Using new helper function)
    from communicator.merge_worker import parallel_decompress_worker_batch
    worker_results, parallel_time = run_parallel_workers(
        parallel_decompress_worker_batch, gpu_assignments, num_gpus
    )
    print(f"\nParallel decompression completed in {parallel_time:.3f}s")

    # Serial gradient application (P1: Using new helper function with P0 fix)
    apply_time = apply_gradients_serially(model, optimizer, worker_results, num_gpus)
    print(f"\nSerial application completed in {apply_time:.3f}s")

    # Summary
    recovery_end = time.time()
    print(f"\n{'='*80}")
    print(f"Batch parallel recovery completed")
    print(f"{'='*80}")
    print(f"Total time: {recovery_end - recovery_start:.3f}s")
    print(f"  - Batch file loading: {load_end - load_start:.3f}s")
    print(f"  - Parallel decompression: {parallel_time:.3f}s")
    print(f"  - Serial application: {apply_time:.3f}s")
    print(f"{'='*80}\n")

    return model, optimizer, max_batch


if __name__ == '__main__':
    main()                                                                                                            