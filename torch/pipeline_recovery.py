import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
import torch
import deepspeed
from deepspeed import comm as dist
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))
from communicator.lowdiff import Communicator
from communicator.pipeline_recovery import PipelineCheckpointRecovery
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
parser.add_argument("--save-batch-freq", default=1, type=int, help='in-memory batching frequency')
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--enable-smart-checkpoint", action="store_true", help='enable smart checkpoint management')
parser.add_argument("--enable-optimizer-monitoring", action="store_true", help='enable hardware fault detection via optimizer state monitoring')
parser.add_argument("--monitoring-safety-factor", type=float, default=1.0, help='safety factor for detection thresholds (>1.0 = more conservative)')
parser.add_argument("--inject-fault", action="store_true", help='inject hardware fault for testing (development only)')
parser.add_argument("--inject-fault-at-batch", type=int, default=50, help='batch index to inject fault')
# Fault Injection (Poisson-based)
parser.add_argument("--enable-fault-injection", action="store_true", help='enable Poisson-based fault injection for testing')
parser.add_argument("--fault-mtbf", type=int, default=1000, help='Mean Time Between Failures (in batches)')
parser.add_argument("--fault-min-batches", type=int, default=50, help='minimum batches before allowing crash')
# SDC Injection
parser.add_argument("--enable-sdc-injection", action="store_true", help='enable SDC (Silent Data Corruption) injection')
parser.add_argument("--sdc-inject-prob", type=float, default=0.02, help='per-batch injection probability')
parser.add_argument("--sdc-param-fraction", type=float, default=0.1, help='fraction of params to corrupt per injection')
parser.add_argument("--sdc-inj-type", type=str, default="rbflip",
                    choices=["rbflip"],
                    help='SDC injection type: rbflip')
parser.add_argument("--sdc-positions-per-param", type=int, default=1,
                    help='how many positions to inject in each selected parameter tensor')
parser.add_argument("--sdc-min-batch", type=int, default=20,
                    help='minimum batch index before SDC injection is allowed')
parser.add_argument("--sdc-target-param", type=str, default=None,
                    help='only inject parameters whose name contains this substring')
parser.add_argument("--pipeline-buffer-size", type=int, default=2,
                    help='pipeline buffer size (number of checkpoints to prefetch)')
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
        # 恢复逻辑全部委托给 PipelineCheckpointRecovery 组件，
        # 此处只负责创建实例并选择流水线恢复路径（批量 or 逐条）
        recovery = PipelineCheckpointRecovery(
            model=model,
            optimizer=optimizer,
            save_dir=args.save_dir,
            model_name=args.model,
            dataset_name=args.dataset,
            compressor_name=args.compressor,
            compressor_ratio=args.compressor_ratio,
            save_batch_freq=args.save_batch_freq,
            buffer_size=args.pipeline_buffer_size,
            rank=dist.get_rank(),
        )

        # 加载基准检查点，并获取其 epoch 和 batch 编号
        model, optimizer, resume_epoch, resume_batch = load_base_checkpoint(model, optimizer, recovery)

        print(f"Base checkpoint loaded: epoch {resume_epoch}, batch {resume_batch}")
        print(f"Will replay differential checkpoints from batch {resume_batch + 1} onwards")

        # 使用流水线恢复
        if args.save_batch_freq > 1:
            last_trained_batch = recovery.replay_batched_differential_checkpoints(resume_epoch, resume_batch)
        else:
            last_trained_batch = recovery.replay_differential_checkpoints(resume_epoch, resume_batch)

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

    # Initialize fault injector (if enabled)
    fault_injector = None
    if args.enable_fault_injection:
        from communicator.fault_injector import FaultInjector
        fault_injector = FaultInjector(
            mtbf_batches=args.fault_mtbf,
            enable=True,
            min_batches_before_crash=args.fault_min_batches,
            crash_log_path="crash_history.log",
            state_save_path="current_epoch.txt"
        )

    # Initialize SDC injector (if enabled)
    sdc_injector = None
    if args.enable_sdc_injection:
        from communicator.sdc_injector import SDCInjector
        sdc_injector = SDCInjector(
            inject_prob=args.sdc_inject_prob,
            param_fraction=args.sdc_param_fraction,
            min_batch=args.sdc_min_batch,
            enable=True,
            log_path=os.path.join(args.save_dir, "sdc_injection.log"),
            inj_type=args.sdc_inj_type,
            positions_per_param=args.sdc_positions_per_param,
            target_param_pattern=args.sdc_target_param,
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

                # SDC 故障注入（在 step() 之前注入梯度扰动）
                sdc_injected = False
                if sdc_injector is not None:
                    sdc_injected = sdc_injector.maybe_inject(
                        model.module if hasattr(model, 'module') else model,
                        batch_idx, epoch
                    )
                # 兼容旧的单次注入参数
                if args.inject_fault and batch_idx == args.inject_fault_at_batch:
                    if dist.get_rank() == 0:
                        print(f"\n[FaultInjection] Injecting hardware fault at batch {batch_idx}")
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad += torch.randn_like(p.grad) * 10.0
                    sdc_injected = True

                # 阻断机制：检查是否需要跳过本次 step()
                step_blocked = False
                if anomaly_detector is not None:
                    step_blocked, block_reason = anomaly_detector.should_block_step()
                    if step_blocked and dist.get_rank() == 0:
                        print(f"[AnomalyDetector] BLOCKED step at batch {batch_idx}: {block_reason}")

                if not step_blocked:
                    model.step()

                    # step() 之后检查优化器状态是否越界
                    if anomaly_detector is not None:
                        anomaly, reason = anomaly_detector.check_after_step(
                            model.module if hasattr(model, 'module') else model,
                            model.optimizer,
                            batch_idx,
                            injected=sdc_injected
                        )
                        if anomaly and dist.get_rank() == 0:
                            # 异常触发后保存全量检查点
                            begin_full = time.time()
                            torch.save({
                                'epoch': epoch + 1,
                                'model': model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                            }, '{}/{}_{}_{}_{}_{}_{}_full.pth.tar'.format(
                                args.save_dir, args.model, args.dataset, args.compressor,
                                args.compressor_ratio, epoch, batch_idx
                            ))
                            print(f"[AnomalyDetector] Emergency full checkpoint saved ({time.time()-begin_full:.3f}s)")
                else:
                    # 被阻断时仍需调用 step() 以保持 DeepSpeed 内部状态一致，
                    # 但先将梯度清零使更新量为零
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    model.step()

                # 泊松过程故障注入（在step()之后，确保梯度已应用）
                if fault_injector is not None and batch_idx > args.fault_min_batches:
                    fault_injector.check_and_crash_distributed(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        save_dir=args.save_dir
                    )

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
                        if communicator.smart_ckpt_manager is not None:
                            should_save_full, save_reason = communicator.smart_ckpt_manager.should_save_full_checkpoint(
                                batch_idx, epoch, loss.item()
                            )
                        else:
                            should_save_full, save_reason = False, 'smart_checkpoint_disabled'
                    elif args.freq > 0 and batch_idx % args.freq == 0:
                        # 传统定期保存
                        should_save_full = True
                        save_reason = 'periodic'

                    if should_save_full:
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
                            if communicator.smart_ckpt_manager is not None:
                                communicator.smart_ckpt_manager.cleanup_old_diff_checkpoints(batch_idx, epoch)
                                communicator.smart_ckpt_manager.cleanup_old_full_checkpoints()

                end = time.time()

            print(f"Epoch {epoch} completed.")

        # 训练结束后输出统计信息
        if dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"Training completed - Final statistics")
            print(f"{'='*60}")
            if anomaly_detector is not None:
                stats = anomaly_detector.get_statistics()
                print(f"  [AnomalyDetector] Anomalies: {stats['anomaly_count']}, Blocks: {stats['block_count']}")
            if sdc_injector is not None:
                stats = sdc_injector.get_statistics()
                print(f"  [SDCInjector] Injections: {stats['inject_count']}, Checks: {stats['total_checks']}")
            print(f"{'='*60}\n")

def load_base_checkpoint(model, optimizer, recovery):
    """
    加载基准全量检查点。核心逻辑委托 PipelineCheckpointRecovery，
    本函数仅做 PERF 计时包装，保持与 GPT.py 统一的日志输出格式。
    """
    print(f"\n{'='*80}")
    print(f"[PERF] ========== Loading Base Checkpoint ==========")
    print(f"{'='*80}\n")

    start = time.time()

    model, optimizer, sel_epoch, sel_batch = recovery.load_latest_full_checkpoint(verbose=True)

    total_time = time.time() - start

    # 更新 args.resume，使后续差分恢复使用正确基准
    args.resume = sel_epoch + 1

    print(f"\n[PERF] ========== Base Checkpoint Summary ==========")
    print(f"[PERF] Total time: {total_time:.3f}s")
    print(f"[PERF] ==================================================")
    print(f"{'='*80}\n")

    return model, optimizer, sel_epoch, sel_batch


if __name__ == '__main__':
    main()
