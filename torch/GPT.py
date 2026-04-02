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
parser.add_argument("--enable-smart-checkpoint", action="store_true", help='enable smart checkpoint management')
parser.add_argument("--enable-optimizer-monitoring", action="store_true", help='enable hardware fault detection via optimizer state monitoring')
parser.add_argument("--monitoring-safety-factor", type=float, default=1.0, help='safety factor for detection thresholds (>1.0 = more conservative)')
parser.add_argument("--inject-fault", action="store_true", help='inject hardware fault for testing (development only)')
parser.add_argument("--inject-fault-at-batch", type=int, default=50, help='batch index to inject fault')
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

    # 确定max_full_interval：如果--freq=0，使用默认值30
    max_full_interval = args.freq if args.freq > 0 else 30

    # Use the Communicator class with smart checkpoint support
    communicator = Communicator(
        model,
        k=args.compress_ratio,
        save_batch_freq=args.save_batch_freq,
        enable_smart_checkpoint=args.enable_smart_checkpoint,
        save_dir=args.save_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        compressor_name=args.compressor,
        compressor_ratio=args.compressor_ratio,
        max_full_interval=max_full_interval
    )
    communicator.register_hooks()

    # Initialize optimizer anomaly detector (if enabled)
    anomaly_detector = None
    if args.enable_optimizer_monitoring:
        from communicator.optimizer_anomaly_detector import OptimizerAnomalyDetector
        global_batch_size = args.batch_size * dist.get_world_size()
        anomaly_detector = OptimizerAnomalyDetector(
            batch_size=global_batch_size,
            seq_length=args.seq_length,
            num_layers=12,  # GPT-2 has 12 layers
            learning_rate=args.lr,
            buffer_size=2,
            safety_factor=args.monitoring_safety_factor
        )

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

                # 差分检查点：始终保存（不能跳过）
                communicator.decompress_save(args.diff, '{}/{}_{}_{}_{}_{}-{}_batch{}.pth.tar'.format(args.save_dir,args.model,args.dataset,args.compressor,args.compressor_ratio,epoch,batch_idx,args.save_batch_freq), batch_idx)

                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 故障注入（仅用于测试）
                if args.inject_fault and batch_idx == args.inject_fault_at_batch:
                    if dist.get_rank() == 0:
                        print(f"\n[FaultInjection] Injecting hardware fault at batch {batch_idx}")
                    # 模拟硬件位翻转：向梯度注入大扰动
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad += torch.randn_like(p.grad) * 10.0

                # 检查优化器状态（在step()之前）
                if anomaly_detector is not None:
                    rolled_back, reason = anomaly_detector.check_and_rollback(
                        model.module if hasattr(model, 'module') else model,
                        model.optimizer,
                        batch_idx
                    )

                    if rolled_back:
                        # 回滚成功，跳过本次更新，重新执行该batch
                        continue

                model.step()

                if dist.get_rank() == 0:
                    print("[Epoch {}/{}] Batch {}, Loss: {:.3f}, Time: {:.3f}"
                        .format(epoch, args.epochs, batch_idx, loss.item(), time.time() - end))

                # 测试：查询buffer状态（每10个batch）
                # 全量检查点保存逻辑
                if dist.get_rank() == 0:
                    should_save_full = False
                    save_reason = ''

                    if args.enable_smart_checkpoint:
                        # 智能检查点决策
                        should_save_full, save_reason = communicator.should_save_full_checkpoint(
                            batch_idx, epoch, loss.item()
                        )
                    elif args.freq > 0 and batch_idx % args.freq == 0:
                        # 传统定期保存
                        should_save_full = True
                        save_reason = 'periodic'

                    if should_save_full:
                        # 先清理过期差分检查点，再保存全量检查点，尽早释放内存
                        communicator.clear_buffer_before(batch_idx)

                        begin_full = time.time()
                        torch.save({
                            'epoch': epoch + 1,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, '{}/{}_{}_{}_{}_{}_{}_full.pth.tar'.format(
                            args.save_dir, args.model, args.dataset, args.compressor,
                            args.compressor_ratio, epoch, batch_idx
                        ))
                        end_full = time.time()
                        print("[SmartCkpt] Saved full checkpoint at batch {} (reason: {}, time: {:.3f}s)".format(
                            batch_idx, save_reason, end_full - begin_full
                        ))

                        # 清理旧的检查点
                        if args.enable_smart_checkpoint:
                            communicator.cleanup_old_checkpoints(batch_idx, epoch)
                            communicator.cleanup_old_full_checkpoints()

                end = time.time()

            print(f"Epoch {epoch} completed.")

        # 训练结束后输出统计信息
        if anomaly_detector is not None and dist.get_rank() == 0:
            stats = anomaly_detector.get_statistics()
            print(f"\n{'='*60}")
            print(f"[OptimizerMonitor] Training completed - Final statistics")
            print(f"{'='*60}")
            print(f"  Anomalies detected: {stats['anomaly_count']}")
            print(f"  Rollbacks performed: {stats['rollback_count']}")
            print(f"  Buffer size: {stats['buffer_size']}")
            print(f"{'='*60}\n")

def load_base_checkpoint(model, optimizer):
    """
    Load base checkpoint with detailed performance logging.
    """
    print(f"\n{'='*80}")
    print(f"[PERF] ========== Loading Base Checkpoint ==========")
    print(f"{'='*80}\n")

    start = time.time()
    filedir = args.save_dir

    # ========== PERF: File Discovery Stage ==========
    discovery_start = time.time()
    print(f"[PERF] Scanning directory: {filedir}")

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

    discovery_end = time.time()
    file_size_mb = os.path.getsize(filepath) / (1024**2)

    print(f"[PERF] Discovery time: {discovery_end - discovery_start:.3f}s")
    print(f"[PERF] Found checkpoint: {sel_file}")
    print(f"[PERF] Checkpoint location: epoch {sel_epoch}, batch {sel_batch}")
    print(f"[PERF] File size: {file_size_mb:.2f} MB\n")

    # ========== PERF: Checkpoint Loading Stage ==========
    load_start = time.time()
    print(f"[PERF] Loading checkpoint from disk...")
    checkpoint = torch.load(filepath, map_location='cpu')
    load_end = time.time()
    load_time = load_end - load_start

    print(f"[PERF] Load time: {load_time:.3f}s")
    if load_time > 0:
        print(f"[PERF] Load speed: {file_size_mb/load_time:.2f} MB/s\n")

    # ========== PERF: State Dict Restoration Stage ==========
    restore_start = time.time()
    print(f"[PERF] Restoring model and optimizer states...")

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    restore_end = time.time()
    restore_time = restore_end - restore_start

    print(f"[PERF] Restore time: {restore_time:.3f}s\n")

    # 更新 args.resume，使后续差分恢复使用正确基准
    args.resume = sel_epoch + 1

    end = time.time()
    total_time = end - start

    # ========== PERF: Summary ==========
    print(f"[PERF] ========== Base Checkpoint Summary ==========")
    print(f"[PERF] Total time: {total_time:.3f}s")
    print(f"[PERF] Time breakdown:")
    if total_time > 0:
        print(f"[PERF]   - Discovery:  {discovery_end - discovery_start:.3f}s ({(discovery_end - discovery_start)/total_time*100:.1f}%)")
        print(f"[PERF]   - Loading:    {load_time:.3f}s ({load_time/total_time*100:.1f}%)")
        print(f"[PERF]   - Restoring:  {restore_time:.3f}s ({restore_time/total_time*100:.1f}%)")
    print(f"[PERF] ==================================================")
    print(f"{'='*80}\n")

    return model, optimizer, sel_epoch, sel_batch

def topk_decompress(values, indices, shape):
    """
    Decompress Top-K compressed gradients back to full gradient tensor.
    """
    tensor_decompressed = torch.zeros(shape).cuda().view(-1)

    # Move tensors to CUDA if they are on CPU
    if isinstance(values, list):
        for idx_tensor, val_tensor in zip(indices, values):
            idx_tensor = idx_tensor.cuda() if not idx_tensor.is_cuda else idx_tensor
            val_tensor = val_tensor.cuda() if not val_tensor.is_cuda else val_tensor
            tensor_decompressed = tensor_decompressed.scatter_add_(0, idx_tensor, val_tensor)
    else:
        values = values.cuda() if not values.is_cuda else values
        indices = indices.cuda() if not indices.is_cuda else indices
        tensor_decompressed = tensor_decompressed.scatter_add_(0, indices, values)

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

def load_differential_checkpoint(model, optimizer, base_batch):  # 新增参数
    """
    Load differential checkpoints iteratively and apply them to the model (SERIAL version).

    Args:
        base_batch (int): The batch number of the base checkpoint.
                         Start replaying from base_batch + 1.
    """
    print(f"\n{'='*80}")
    print(f"[PERF] ========== Serial Differential Checkpoint Recovery ==========")
    print(f"{'='*80}\n")

    recovery_start = time.time()
    filedir = args.save_dir
    _parameter_names = {name: param for name, param in model.named_parameters()}

    # ========== PERF: Find Checkpoints ==========
    find_start = time.time()
    iterations = find_max(base_batch)
    find_end = time.time()

    if iterations == -1:
        print("[PERF] WARNING: No differential checkpoints found")
        return model, optimizer, base_batch

    num_checkpoints = iterations - base_batch
    print(f"[PERF] Found {num_checkpoints} differential checkpoints (batch {base_batch + 1} to {iterations})")
    print(f"[PERF] Checkpoint discovery time: {find_end - find_start:.3f}s\n")

    # ========== PERF: Serial Recovery Loop ==========
    recovery_times = []
    load_times = []
    decompress_times = []
    apply_times = []

    print(f"[PERF] ========== Starting Serial Recovery Loop ==========\n")

    # 从 base_batch + 1 开始回放
    for i in range(base_batch + 1, iterations + 1):  # iterations + 1 确保包含最后一个
        iter_start = time.time()
        filepath = filedir + '/{}_{}_{}_{}_{}-{}_batch1.pth.tar'.format(
            args.model, args.dataset, args.compressor, args.compressor_ratio,
            args.resume-1, i
        )

        # 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"[PERF] WARNING: Diff checkpoint {filepath} not found, skipping...")
            continue

        # Load checkpoint
        load_start = time.time()
        tensor_compressed = torch.load(filepath, map_location='cpu')
        load_end = time.time()
        load_time = load_end - load_start
        load_times.append(load_time)

        # Decompress gradients
        decompress_start = time.time()
        for key in tensor_compressed.keys():
            tensor = topk_decompress(
                tensor_compressed[key]['values'],
                tensor_compressed[key]['indices'],
                tensor_compressed[key]['shape']
            )
            param = _parameter_names.get(key)
            if param is not None:
                param.grad = tensor
        decompress_end = time.time()
        decompress_time = decompress_end - decompress_start
        decompress_times.append(decompress_time)

        # Apply gradients
        apply_start = time.time()
        optimizer.step()
        apply_end = time.time()
        apply_time = apply_end - apply_start
        apply_times.append(apply_time)

        iter_end = time.time()
        iter_time = iter_end - iter_start
        recovery_times.append(iter_time)

        # Log every 5 checkpoints
        if (i - base_batch) % 5 == 0 or i == iterations:
            print(f"[PERF] Checkpoint {i - base_batch}/{num_checkpoints} (batch {i})")
            print(f"       Load: {load_time:.4f}s | Decompress: {decompress_time:.4f}s | Apply: {apply_time:.4f}s | Total: {iter_time:.4f}s")

    recovery_end = time.time()
    total_recovery_time = recovery_end - recovery_start

    # ========== PERF: Summary ==========
    print(f"\n[PERF] ========== Serial Recovery Summary ==========")
    print(f"[PERF] Total recovery time: {total_recovery_time:.3f}s")
    print(f"[PERF] Checkpoints processed: {len(recovery_times)}")

    if recovery_times:
        total_load = sum(load_times)
        total_decompress = sum(decompress_times)
        total_apply = sum(apply_times)

        print(f"\n[PERF] Time breakdown:")
        if total_recovery_time > 0:
            print(f"[PERF]   - Total load time:       {total_load:.3f}s ({total_load/total_recovery_time*100:.1f}%)")
            print(f"[PERF]   - Total decompress time: {total_decompress:.3f}s ({total_decompress/total_recovery_time*100:.1f}%)")
            print(f"[PERF]   - Total apply time:      {total_apply:.3f}s ({total_apply/total_recovery_time*100:.1f}%)")

        if len(recovery_times) > 0:
            print(f"\n[PERF] Average per checkpoint:")
            print(f"[PERF]   - Load:       {total_load/len(recovery_times):.4f}s")
            print(f"[PERF]   - Decompress: {total_decompress/len(recovery_times):.4f}s")
            print(f"[PERF]   - Apply:      {total_apply/len(recovery_times):.4f}s")
            print(f"[PERF]   - Total:      {sum(recovery_times)/len(recovery_times):.4f}s")

        print(f"\n[PERF] Performance stats:")
        print(f"[PERF]   - Min checkpoint time: {min(recovery_times):.4f}s")
        print(f"[PERF]   - Max checkpoint time: {max(recovery_times):.4f}s")

    print(f"[PERF] ==================================================")
    print(f"{'='*80}\n")

    # 返回最后恢复到的 batch 编号
    last_batch = iterations if iterations != -1 else base_batch
    return model, optimizer, last_batch

def load_batch_differential_checkpoint(model, optimizer, base_batch):  # 新增参数
    """
    Load batched differential checkpoints for more efficient recovery (SERIAL version).

    Args:
        base_batch (int): The batch number of the base checkpoint.
                         Start replaying from the first batch after base_batch.
    """
    print(f"\n{'='*80}")
    print(f"[PERF] ========== Serial Batch Differential Checkpoint Recovery ==========")
    print(f"{'='*80}\n")

    recovery_start = time.time()
    filedir = args.save_dir
    _parameter_names = {name: param for name, param in model.named_parameters()}

    # ========== PERF: Find Checkpoints ==========
    find_start = time.time()
    iterations = find_max(base_batch)
    find_end = time.time()

    if iterations == -1:
        print("[PERF] WARNING: No differential checkpoints found")
        return model, optimizer, base_batch

    print(f"[PERF] Found checkpoints up to batch {iterations}")
    print(f"[PERF] Base batch: {base_batch}")
    print(f"[PERF] Checkpoint discovery time: {find_end - find_start:.3f}s\n")

    # 计算第一个需要加载的批次
    # 找到 base_batch 之后的第一个批次边界
    first_batch = ((base_batch // args.save_batch_freq) + 1) * args.save_batch_freq - 1

    # ========== PERF: Data Preparation ==========
    data_prep_start = time.time()
    print(f"[PERF] ========== Data Preparation Stage ==========")
    print(f"[PERF] Batch frequency: {args.save_batch_freq}")
    print(f"[PERF] First batch to load: {first_batch}\n")

    batch_files = []
    total_data_size = 0

    for i in range(first_batch, iterations + 1, args.save_batch_freq):
        filepath = filedir + '/{}_{}_{}_{}_{}-{}_batch{}.pth.tar'.format(
            args.model, args.dataset, args.compressor, args.compressor_ratio,
            args.resume-1, i, args.save_batch_freq
        )

        if os.path.exists(filepath):
            file_size_mb = os.path.getsize(filepath) / (1024**2)
            total_data_size += file_size_mb
            batch_files.append((i, filepath, file_size_mb))
            print(f"[PERF] Found batch file: {os.path.basename(filepath)} ({file_size_mb:.2f} MB)")
        else:
            print(f"[PERF] WARNING: Batch file not found: {os.path.basename(filepath)}")

    data_prep_end = time.time()
    data_prep_time = data_prep_end - data_prep_start

    print(f"\n[PERF] Data preparation completed in {data_prep_time:.3f}s")
    print(f"[PERF] Total batch files: {len(batch_files)}")
    print(f"[PERF] Total data size: ~{total_data_size:.2f} MB\n")

    if len(batch_files) == 0:
        print("[PERF] ERROR: No batch files found")
        return model, optimizer, base_batch

    # ========== PERF: Serial Recovery Loop ==========
    print(f"[PERF] ========== Starting Serial Batch Recovery Loop ==========\n")

    batch_recovery_times = []
    file_load_times = []
    checkpoint_apply_times = []
    checkpoints_processed = 0

    # 从 first_batch 开始，每隔 save_batch_freq 加载一次
    for idx, (file_batch_idx, filepath, file_size_mb) in enumerate(batch_files):
        batch_start = time.time()

        # Load batch file
        load_start = time.time()
        print(f"[PERF] Loading batch file {idx+1}/{len(batch_files)}: {os.path.basename(filepath)}")
        tensor_compressed = torch.load(filepath, map_location='cpu')
        load_end = time.time()
        load_time = load_end - load_start
        file_load_times.append(load_time)

        if load_time > 0:
            print(f"       File size: {file_size_mb:.2f} MB | Load time: {load_time:.3f}s | Speed: {file_size_mb/load_time:.2f} MB/s")
        else:
            print(f"       File size: {file_size_mb:.2f} MB | Load time: {load_time:.3f}s")

        # Process checkpoints in this batch
        ckpts_in_batch = 0
        for j in range(file_batch_idx - args.save_batch_freq + 1, file_batch_idx + 1):
            # 只回放 base_batch 之后的差分检查点
            if j <= base_batch:
                continue

            if j not in tensor_compressed:
                print(f"       WARNING: Iteration {j} not found in batch checkpoint, skipping...")
                continue

            # Decompress and apply
            apply_start = time.time()
            for key in tensor_compressed[j].keys():
                tensor = topk_decompress(
                    tensor_compressed[j][key]['values'],
                    tensor_compressed[j][key]['indices'],
                    tensor_compressed[j][key]['shape']
                )
                param = _parameter_names.get(key)
                if param is not None:
                    param.grad = tensor
            optimizer.step()
            apply_end = time.time()

            checkpoint_apply_times.append(apply_end - apply_start)
            ckpts_in_batch += 1
            checkpoints_processed += 1

        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_recovery_times.append(batch_time)

        print(f"       Checkpoints in batch: {ckpts_in_batch}")
        print(f"       Batch processing time: {batch_time:.3f}s\n")

    recovery_end = time.time()
    total_recovery_time = recovery_end - recovery_start

    # ========== PERF: Final Summary ==========
    print(f"[PERF] ========== FINAL PERFORMANCE SUMMARY ==========")
    print(f"[PERF] Total recovery time: {total_recovery_time:.3f}s")
    print(f"[PERF] ")
    print(f"[PERF] Time breakdown:")
    if total_recovery_time > 0:
        print(f"[PERF]   1. Data preparation:      {data_prep_time:.3f}s ({data_prep_time/total_recovery_time*100:.1f}%)")
        print(f"[PERF]   2. File loading:          {sum(file_load_times):.3f}s ({sum(file_load_times)/total_recovery_time*100:.1f}%)")
        print(f"[PERF]   3. Checkpoint application: {sum(checkpoint_apply_times):.3f}s ({sum(checkpoint_apply_times)/total_recovery_time*100:.1f}%)")
    print(f"[PERF] ")
    print(f"[PERF] Data statistics:")
    print(f"[PERF]   - Batch files loaded: {len(batch_files)}")
    print(f"[PERF]   - Checkpoints processed: {checkpoints_processed}")
    print(f"[PERF]   - Total data size: ~{total_data_size:.2f} MB")
    if total_recovery_time > 0:
        print(f"[PERF]   - Effective throughput: {total_data_size/total_recovery_time:.2f} MB/s")

    if checkpoint_apply_times:
        print(f"\n[PERF] Per-checkpoint statistics:")
        if len(checkpoint_apply_times) > 0:
            print(f"[PERF]   - Average apply time: {sum(checkpoint_apply_times)/len(checkpoint_apply_times):.4f}s")
            print(f"[PERF]   - Min apply time: {min(checkpoint_apply_times):.4f}s")
            print(f"[PERF]   - Max apply time: {max(checkpoint_apply_times):.4f}s")

    if batch_recovery_times:
        print(f"\n[PERF] Per-batch-file statistics:")
        if len(batch_recovery_times) > 0:
            print(f"[PERF]   - Average batch time: {sum(batch_recovery_times)/len(batch_recovery_times):.3f}s")
            print(f"[PERF]   - Min batch time: {min(batch_recovery_times):.3f}s")
            print(f"[PERF]   - Max batch time: {max(batch_recovery_times):.3f}s")

    print(f"[PERF] ==================================================")
    print(f"{'='*80}\n")

    # 返回最后恢复到的 batch 编号
    last_batch = iterations if iterations != -1 else base_batch
    return model, optimizer, last_batch

if __name__ == '__main__':
    main()