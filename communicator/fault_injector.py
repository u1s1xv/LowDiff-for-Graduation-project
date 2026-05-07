#!/usr/bin/env python3
"""
基于泊松过程的故障注入模块（Fault Injection Module）

基于 MTBF（Mean Time Between Failures）和泊松分布的科学概率模型，
用于测试分布式训练系统的容错与恢复能力。
"""

import os
import time
import random
import math
import torch
try:
    from deepspeed import comm as dist
except ImportError:
    import torch.distributed as dist


class FaultInjector:
    """
    基于泊松过程的故障注入器
    
    理论基础：
    - 硬件故障通常遵循泊松过程（Poisson Process）
    - 故障间隔时间服从指数分布：P(T > t) = exp(-λt)
    - λ = 1/MTBF（故障率）
    
    使用场景：
    - 测试检查点恢复机制
    - 验证分布式一致性
    - 压力测试容错能力
    """
    
    def __init__(self, mtbf_batches=1000, enable=False, min_batches_before_crash=50,
                 crash_log_path=None, state_save_path=None):
        """
        初始化故障注入器

        Args:
            mtbf_batches: 平均故障间隔（以batch为单位），默认1000
            enable: 是否启用故障注入
            min_batches_before_crash: 最小训练步数后才允许崩溃（避免过早崩溃）
            crash_log_path: 崩溃日志路径
            state_save_path: 状态保存路径

        理论依据：
            - 泊松过程：硬件故障间隔服从指数分布
            - 累积故障概率：P(T ≤ t) = 1 - exp(-λt)
            - 参考：Google "Fail at Scale" (OSDI 2015)
        """
        self.enable = enable
        self.mtbf_batches = mtbf_batches
        self.min_batches_before_crash = min_batches_before_crash
        self.crash_log_path = crash_log_path or "crash_history.log"
        self.state_save_path = state_save_path or "current_epoch.txt"

        # 计算故障率 λ = 1/MTBF
        self.lambda_rate = 1.0 / mtbf_batches

        # 统计信息
        self.crash_count = 0

        # 初始化随机种子（使用时间戳+进程ID，确保每次运行不同）
        seed = int(time.time() * 1000) + os.getpid()
        random.seed(seed)

        if self.enable and dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"[FaultInjector] Initialized with Poisson Process Model")
            print(f"{'='*60}")
            print(f"  MTBF (batches):        {mtbf_batches}")
            print(f"  Failure rate (λ):      {self.lambda_rate:.6f}")
            print(f"  Min batches before crash: {min_batches_before_crash}")
            print(f"  Random seed:           {seed}")
            print(f"  Crash log:             {self.crash_log_path}")
            print(f"  State save path:       {self.state_save_path}")
            print(f"{'='*60}\n")
    
    def should_crash(self, batch_idx, epoch):
        """
        基于泊松过程决策是否触发故障

        理论：泊松过程中，故障间隔时间服从指数分布
              累积故障概率：P(T ≤ t) = 1 - exp(-λt)
              其中 t 是自上次故障以来经过的时间步数

        Args:
            batch_idx: 当前batch索引
            epoch: 当前epoch

        Returns:
            (should_crash: bool, reason: str)
        """
        if not self.enable:
            return False, 'disabled'

        # 仅在rank 0做决策
        if dist.get_rank() != 0:
            return False, 'not_rank_0'

        # 最小步数保护
        if batch_idx < self.min_batches_before_crash:
            return False, 'too_early'

        # 修正后的泊松过程：直接使用batch_idx作为时间累积
        # P(T ≤ t) = 1 - exp(-λt)
        crash_prob = 1 - math.exp(-self.lambda_rate * batch_idx)

        # 限制最大概率为50%，避免必然崩溃
        crash_prob = min(crash_prob, 0.5)

        # 随机决策
        if random.random() < crash_prob:
            self.crash_count += 1
            reason = f'poisson_λ={self.lambda_rate:.6f}_Δt={batch_idx}_prob={crash_prob:.4f}'
            return True, reason

        return False, 'no_crash'
    
    def trigger_crash(self, epoch, batch_idx, reason, save_dir=None):
        """
        触发故障崩溃（仅在rank 0执行记录，所有进程同步退出）
        
        Args:
            epoch: 当前epoch
            batch_idx: 当前batch索引
            reason: 崩溃原因
            save_dir: 检查点保存目录
        """
        if dist.get_rank() == 0:
            # 1. 记录崩溃日志
            crash_msg = (
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Simulated Crash #{self.crash_count} | "
                f"Epoch {epoch} | Batch {batch_idx} | "
                f"Reason: {reason}"
            )
            
            print(f"\n{'!'*60}")
            print(f"[FaultInjector] TRIGGERING SIMULATED CRASH")
            print(f"{'!'*60}")
            print(crash_msg)
            print(f"{'!'*60}\n")
            
            # 写入崩溃历史日志
            log_path = os.path.join(save_dir, self.crash_log_path) if save_dir else self.crash_log_path
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
            with open(log_path, "a") as f:
                f.write(crash_msg + "\n")
            
            # 2. 保存当前训练进度
            state_path = os.path.join(save_dir, self.state_save_path) if save_dir else self.state_save_path
            with open(state_path, "w") as f:
                f.write(f"{epoch},{batch_idx}\n")
            
            # 3. 确保日志刷新
            time.sleep(2)
        
        # 4. 同步所有进程（确保所有节点同时崩溃）
        dist.barrier()
        
        # 5. 模拟硬崩溃（强制退出，不执行清理）
        os._exit(1)

    def check_and_crash_distributed(self, epoch, batch_idx, save_dir=None):
        """
        分布式一致性故障注入（所有进程同步决策）

        工作流程：
        1. Rank 0 基于泊松过程做决策
        2. 通过 broadcast 同步决策到所有进程
        3. 所有进程同时执行崩溃或继续训练

        Args:
            epoch: 当前epoch
            batch_idx: 当前batch索引
            save_dir: 检查点保存目录

        Returns:
            None（如果触发崩溃则直接退出）
        """
        if not self.enable:
            return

        # 初始化崩溃标志位（所有进程）
        crash_flag = torch.tensor([0], dtype=torch.int, device='cuda')

        # Rank 0 做决策
        if dist.get_rank() == 0:
            should_crash, reason = self.should_crash(batch_idx, epoch)
            if should_crash:
                crash_flag[0] = 1
                # 预先记录原因（在broadcast前）
                self._crash_reason = reason

        # 同步决策到所有进程
        dist.broadcast(crash_flag, src=0)

        # 所有进程根据决策执行
        if crash_flag.item() == 1:
            reason = getattr(self, '_crash_reason', 'unknown') if dist.get_rank() == 0 else 'synced_from_rank0'
            self.trigger_crash(epoch, batch_idx, reason, save_dir)

    def get_statistics(self):
        """返回统计信息"""
        return {
            'crash_count': self.crash_count,
            'mtbf_batches': self.mtbf_batches,
            'lambda_rate': self.lambda_rate,
            'last_crash_batch': self.last_crash_batch if self.last_crash_batch != -float('inf') else None
        }

