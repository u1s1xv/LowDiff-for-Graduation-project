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
    print("loading {}".format(filepath))
    checkpoint = torch.load(filepath, map_location='cpu')

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.resume = sel_epoch + 1
    end = time.time()
    print("load base checkpoint takes {:.3f}s (epoch {}, batch {})".format(end - start, sel_epoch, sel_batch))
    return model, optimizer, sel_epoch, sel_batch


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


def tree_merge_checkpoints(diff_data):
    """
    Merge differential checkpoints using tree-based parallel strategy.
    Uses multiprocessing to parallelize merge operations within each round.

    Returns:
        tuple: (merged_data, merge_time, merge_round)
    """
    merge_start = time.time()
    current_level = sorted(diff_data.keys()) if isinstance(list(diff_data.keys())[0], int) else list(diff_data.keys())
    merge_round = 0

    num_gpus = torch.cuda.device_count()
    num_gpus = max(num_gpus, 1)

    num_workers = min(mp.cpu_count() // 2, num_gpus, 8)
    num_workers = max(num_workers, 1)

    print(f"Using {num_workers} worker processes and {num_gpus} GPUs for parallel merging (IPC mode)")

    from communicator.merge_worker import queue_worker

    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    workers = []
    for i in range(num_workers):
        device_id = i % num_gpus
        p = ctx.Process(target=queue_worker, args=(task_queue, result_queue, device_id))
        p.start()
        workers.append(p)

    try:
        while len(current_level) > 1:
            merge_round += 1
            next_level = []

            merge_tasks = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    ckpt1_idx = current_level[i]
                    ckpt2_idx = current_level[i + 1]
                    merge_tasks.append((
                        ckpt1_idx,
                        ckpt2_idx,
                        diff_data[ckpt1_idx],
                        diff_data[ckpt2_idx]
                    ))
                else:
                    next_level.append(current_level[i])

            if len(merge_tasks) > 0:
                round_start = time.time()

                for idx, (ckpt1_idx, ckpt2_idx, ckpt1_data, ckpt2_data) in enumerate(merge_tasks):
                    task_queue.put((idx, ckpt1_data, ckpt2_data, args.compressor_ratio))

                results = {}
                for _ in range(len(merge_tasks)):
                    task_id, merged_data = result_queue.get()
                    results[task_id] = merged_data

                round_end = time.time()

                for idx, (ckpt1_idx, ckpt2_idx, _, _) in enumerate(merge_tasks):
                    merged_idx = f"merged_{ckpt1_idx}_{ckpt2_idx}"
                    diff_data[merged_idx] = results[idx]
                    next_level.append(merged_idx)

                    del diff_data[ckpt1_idx]
                    del diff_data[ckpt2_idx]

                print(f"Round {merge_round}: merged {len(merge_tasks)} pairs in {round_end - round_start:.3f}s, {len(next_level)} checkpoints remaining")

            current_level = next_level

    finally:
        for _ in range(num_workers):
            task_queue.put(None)

        for p in workers:
            p.join()

    merge_end = time.time()
    merge_time = merge_end - merge_start
    final_merged_idx = current_level[0]
    return diff_data[final_merged_idx], merge_time, merge_round

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

def load_differential_checkpoint(model, optimizer, base_batch):
    """
    Load differential checkpoints using tree-based parallel merging strategy.
    """
    begin = time.time()
    filedir = args.save_dir
    iterations = find_max(base_batch)

    if iterations == -1:
        return model, optimizer, base_batch

    diff_checkpoints = list(range(base_batch + 1, iterations + 1))
    num_diffs = len(diff_checkpoints)
    print(f"Loading {num_diffs} differential checkpoints (batch {base_batch + 1} to {iterations})")

    diff_data = {}

    load_start = time.time()
    for i in diff_checkpoints:
        filepath = filedir + '/{}_{}_{}_{}_{}-{}_batch1.pth.tar'.format(
            args.model, args.dataset, args.compressor, args.compressor_ratio,
            args.resume-1, i
        )
        if not os.path.exists(filepath):
            continue
        diff_data[i] = torch.load(filepath, map_location='cpu')
    load_end = time.time()
    print(f"Loaded {len(diff_data)} checkpoints in {load_end - load_start:.3f}s")

    merged_data, merge_time, merge_round = tree_merge_checkpoints(diff_data)
    print(f"Tree-based merging completed in {merge_time:.3f}s ({merge_round} rounds)")

    apply_merged_checkpoint(model, optimizer, merged_data)

    end = time.time()
    print("parallel recovery takes {:.3f}s".format(end - begin))
    return model, optimizer, iterations

def load_batch_differential_checkpoint(model, optimizer, base_batch):
    """
    Load batched differential checkpoints using tree-based parallel merging strategy.
    """
    begin = time.time()
    filedir = args.save_dir
    iterations = find_max(base_batch)

    if iterations == -1:
        return model, optimizer, base_batch

    first_batch = ((base_batch // args.save_batch_freq) + 1) * args.save_batch_freq - 1
    diff_data = {}

    load_start = time.time()
    batch_files_loaded = 0
    for i in range(first_batch, iterations + 1, args.save_batch_freq):
        filepath = filedir + '/{}_{}_{}_{}_{}-{}_batch{}.pth.tar'.format(
            args.model, args.dataset, args.compressor, args.compressor_ratio,
            args.resume-1, i, args.save_batch_freq
        )
        if not os.path.exists(filepath):
            continue
        tensor_compressed = torch.load(filepath, map_location='cpu')
        batch_files_loaded += 1
        for j in range(i - args.save_batch_freq + 1, i + 1):
            if j <= base_batch or j not in tensor_compressed:
                continue
            diff_data[j] = tensor_compressed[j]
    load_end = time.time()
    print(f"Loaded {batch_files_loaded} batch files containing {len(diff_data)} checkpoints in {load_end - load_start:.3f}s")

    if len(diff_data) == 0:
        return model, optimizer, base_batch

    merged_data, merge_time, merge_round = tree_merge_checkpoints(diff_data)
    print(f"Tree-based merging completed in {merge_time:.3f}s ({merge_round} rounds)")

    apply_merged_checkpoint(model, optimizer, merged_data)

    end = time.time()
    print("parallel batch recovery takes {:.3f}s".format(end - begin))
    return model, optimizer, iterations


if __name__ == '__main__':
    main()