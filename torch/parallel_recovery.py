import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from deepspeed import comm as dist
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))
from communicator.lowdiff import Communicator
import re
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    set_seed
)

# Argument parsing
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
parser.add_argument("--compressor_ratio", default=0.01, type=float, help='choose compress ratio for compressor')
parser.add_argument("--save-dir", default='/data/lowdiff', type=str, help='directory to save checkpoints')
parser.add_argument("--resume", type=int, default=0, help='resume from checkpoint')
parser.add_argument("--diff", action="store_true", help='whether to use differential checkpoint')
parser.add_argument("--freq", default=0, type=int, help='how many iteration to save a full checkpoint')
parser.add_argument("--save-batch-freq", default='1', type=int, help='in-memory batching frequency')
parser.add_argument("--seq_length", type=int, default=512)  
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
args = parser.parse_args()


def main():
    # Initialize argument parsing
    model_path = "/mnt/newdisk/xiekunpeng/LowDiff/data/dataset/nlp/openai-community/" + args.model

    # Initialize DeepSpeed distributed training
    deepspeed.init_distributed()
    dist.barrier()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    set_seed(42 + rank)  # Set deterministic seed
    torch.cuda.set_device(args.local_rank)
    print(f"[Rank {rank}/{world_size}] Initialized DeepSpeed")

    # Load dataset and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_length,
            padding="max_length"
        )

    # Load and process wikitext-103 dataset
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
    # Data collator (automatically generate labels)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Use causal language modeling
    )

    # Distributed sampler
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

    # Initialize model (enable gradient checkpointing to save memory)
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

    # Optionally resume from a checkpoint at rank 0, then broadcast weights to other workers
    resume_epoch = 0
    resume_batch = 0
    last_trained_batch = 0  # 记录最后训练到的 batch
    if args.resume and dist.get_rank() == 0:
        # 加载基准检查点，并获取其 epoch 和 batch 编号
        model, optimizer, resume_epoch, resume_batch = load_base_checkpoint(model, optimizer)

        print(f"Base checkpoint loaded: epoch {resume_epoch}, batch {resume_batch}")
        print(f"Will replay differential checkpoints from batch {resume_batch + 1} onwards")

        # 根据批处理频率选择恢复方法，并传递 base_batch 参数
        if args.save_batch_freq > 1:
            model, optimizer, last_trained_batch = load_batch_differential_checkpoint(model, optimizer, resume_batch)
        else:
            model, optimizer, last_trained_batch = load_differential_checkpoint(model, optimizer, resume_batch)

        print(f"Differential checkpoint replay completed")
        print(f"Last trained batch: {last_trained_batch}")
        print(f"Training will resume from epoch {resume_epoch}, batch {last_trained_batch + 1}")

    model.cuda()
    
    # Initialize DeepSpeed
    deepspeed.enable_backward_allreduce = False
    
    # Use the Communicator class
    communicator = Communicator(model, k=args.compress_ratio, save_batch_freq=args.save_batch_freq)
    communicator.register_hooks()

    # Training loop
    # 如果是恢复训练，从恢复的 epoch 开始；否则从 0 开始
    start_epoch = resume_epoch if args.resume else 0

    # 判断训练是否已经完成
    training_completed = False
    if args.resume and dist.get_rank() == 0:
        # 检查是否已经训练完成
        # 如果恢复的 epoch 已经是最后一个 epoch，且没有更多的 batch 需要训练
        if resume_epoch >= args.epochs - 1:
            # 在最后一个 epoch 中，检查是否还有 batch 需要训练
            # 如果 last_trained_batch 就是恢复点，说明没有新的差分检查点，训练可能已完成
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
                # 只在恢复的 epoch 中跳过已训练的 batch
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
    # 保存格式: {save_dir}/{model}_{dataset}_{compressor}_{compressor_ratio}_{epoch}_{batch_idx}_full.pth.tar
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
    # 选择 epoch 最大，若同 epoch 则选择 batch 最大
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
    # 更新 args.resume，使后续差分恢复使用正确基准
    args.resume = sel_epoch + 1
    end = time.time()
    print("load base checkpoint takes {:.3f}s (epoch {}, batch {})".format(end - start, sel_epoch, sel_batch))
    return model, optimizer, sel_epoch, sel_batch

def topk_decompress(values, indices, shape):
    """
    Decompress Top-K compressed gradients back to full gradient tensor.
    """
    tensor_decompressed = torch.zeros(shape).cuda().view(-1)
    for idx, val in zip(indices, values):
        tensor_decompressed = tensor_decompressed.scatter_add_(0, idx, val)
    return tensor_decompressed.view(shape)

def find_max(base_batch):  # 新增参数：base_batch（基准检查点的 batch 编号）
    """
    Find the maximum iteration number of differential checkpoints AFTER base_batch.
    
    Args:
        base_batch (int): The batch number of the base checkpoint.
                         Only find diff checkpoints after this batch.
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
            # 只考虑 base_batch 之后的差分检查点
            if x > base_batch and x > max_x:
                max_x = x
    
    if max_x != -1:
        print("Max diff ckpt at epoch {}, iteration {} (after base batch {})".format(
            args.resume, max_x, base_batch
        ))
    else:
        print("No diff ckpt found after base batch {}".format(base_batch))
    
    return max_x

if __name__ == '__main__':
    main()