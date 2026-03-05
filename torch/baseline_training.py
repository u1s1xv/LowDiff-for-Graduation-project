#!/usr/bin/env python3
# Minimal baseline training script - no compression, no checkpoints

import os
import sys
import time
import argparse
from pathlib import Path

import torch
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


parser = argparse.ArgumentParser(description="Baseline Training (No Compression, No Checkpoints)")
parser.add_argument("--dataset", default="wikitext-2", type=str)
parser.add_argument("--model", default="gpt2", type=str)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--seq-length", type=int, default=512)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()


def main():
    model_path = "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/openai-community/" + args.model

    deepspeed.init_distributed()
    dist.barrier()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    set_seed(42 + rank)
    torch.cuda.set_device(args.local_rank)
    
    if rank == 0:
        print(f"Baseline Training: {world_size} GPUs, Model: {args.model}, Dataset: {args.dataset}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_length,
            padding="max_length"
        )

    if args.dataset == "wikitext-103":
        dataset = load_dataset("/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103",
                        data_files={
                            "train": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103/train.txt",
                            "validation": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103/valid.txt",
                            "test": "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/transformer/wikitext-103/test.txt"
                        })["train"]
    elif args.dataset == "wikitext-2":
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
        num_proc=12,
        keep_in_memory=True,  # 使用内存缓存，避免磁盘空间不足
        load_from_cache_file=False  # 禁用磁盘缓存
    )

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

    if args.model == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/openai-community/gpt2")
    elif args.model == "gpt2-medium":
        model = GPT2LMHeadModel.from_pretrained("/data/dataset/nlp/openai-community/gpt2-medium")
    elif args.model == "gpt2-large":
        model = GPT2LMHeadModel.from_pretrained("/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/openai-community/gpt2-large")
    else:
        raise ValueError("Model not supported")
    
    model.gradient_checkpointing_enable()
    model.cuda()

    world_size = dist.get_world_size()
    ds_config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-4,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * len(train_loader),
                "warmup_min_lr": 0,
                "warmup_max_lr": 5e-4,
                "warmup_num_steps": 100
            }
        }
    }
    model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            
            inputs = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss

            model.backward(loss)
            model.step()

            if dist.get_rank() == 0:
                print("[Epoch {}/{}] Batch {}, Loss: {:.3f}, Time: {:.3f}"
                    .format(epoch, args.epochs, batch_idx, loss.item(), time.time() - start_time))

        if rank == 0:
            print(f"Epoch {epoch} completed.")


if __name__ == "__main__":
    main()
