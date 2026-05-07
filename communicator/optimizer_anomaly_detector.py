#!/usr/bin/env python3
"""
基于优化器状态监控的硬件故障检测器（阻断机制）

基于论文 "Understanding and Mitigating Hardware Failures in Deep Learning Training Systems"
使用Algorithm 1的数学严格推导来检测硬件故障。

检测策略：在 optimizer.step() 之后检查 m_t 是否越界，
若越界则标记异常，下一个 batch 跳过 step() 阻止污染继续扩散。
"""

import math
import torch
try:
    from deepspeed import comm as dist
except ImportError:
    import torch.distributed as dist


class OptimizerAnomalyDetector:
    """
    检测Adam优化器的梯度历史值(m_t)是否超过理论边界。

    采用阻断机制（而非回滚）：检测到异常后，在后续迭代中跳过 optimizer.step()，
    阻止被污染的梯度继续更新模型参数。被污染的优化器状态会通过 Adam 的
    指数移动平均机制在后续正常训练中逐渐稀释。
    """

    def __init__(self, batch_size, seq_length, num_layers=12,
                 learning_rate=1e-4, buffer_size=2, safety_factor=1.0):
        """
        Args:
            batch_size: 全局batch size（所有GPU总和）
            seq_length: 序列长度
            num_layers: 模型层数
            learning_rate: 学习率η
            buffer_size: 连续阻断的最大迭代数（默认2）
            safety_factor: 安全系数（>1.0更保守）
        """
        n_l = batch_size * seq_length
        m = batch_size

        self.m_threshold = 20 * math.sqrt(n_l / (m ** 2)) * safety_factor
        # 阻断状态：检测到异常后连续跳过的剩余次数
        self.block_remaining = 0
        self.max_block_steps = buffer_size  # 最多连续阻断几步

        # 统计
        self.anomaly_count = 0
        self.block_count = 0  # 实际阻断（跳过step）的总次数

        if dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"[AnomalyDetector] Initialized (block mechanism)")
            print(f"{'='*60}")
            print(f"  m_threshold           = {self.m_threshold:.6f}")
            print(f"  max_block_steps       = {self.max_block_steps}")
            print(f"  safety_factor         = {safety_factor}")
            print(f"{'='*60}\n")

    def should_block_step(self):
        """
        在 optimizer.step() 之前调用，判断是否应跳过本次更新。

        Returns:
            (should_block: bool, reason: str)
        """
        if self.block_remaining > 0:
            self.block_remaining -= 1
            self.block_count += 1
            return True, f'blocking_update (remaining={self.block_remaining})'
        return False, ''

    def check_after_step(self, model, optimizer, iteration, injected=False):
        """
        在 optimizer.step() 之后调用，检查优化器状态是否越界（仅 m_t）。
        若越界，设置阻断标志，后续 batch 将跳过 step()。

        Returns:
            (anomaly_detected: bool, reason: str)
        """
        name_by_param = {p: n for n, p in model.named_parameters()}
        max_m, max_m_name = self._extract_optimizer_states(optimizer, name_by_param)

        # 周期性日志
        if iteration > 0 and iteration % 100 == 0 and dist.get_rank() == 0:
            print(f"\n[AnomalyDetector] iter {iteration}: "
                f"max_m={max_m:.6f}/{self.m_threshold:.6f} "
                f"anomalies={self.anomaly_count} blocks={self.block_count}")

        if injected and dist.get_rank() == 0:
            print(f"[AnomalyDetector] Injected step {iteration}: "
                  f"max_m={max_m:.6f} param={max_m_name}")

        reason = ''
        if max_m > self.m_threshold:
            reason = f'max_m={max_m:.4f} > {self.m_threshold:.4f}'
        

        if reason:
            self.anomaly_count += 1
            self.block_remaining = self.max_block_steps

            if dist.get_rank() == 0:
                print(f"\n{'!'*60}")
                print(f"[AnomalyDetector] ANOMALY #{self.anomaly_count} at iter {iteration}")
                print(f"  Reason: {reason}")
                print(f"  Will block next {self.max_block_steps} step(s)")
                print(f"{'!'*60}\n")

            return True, reason

        return False, ''

    def _extract_optimizer_states(self, optimizer, name_by_param):
        max_m = 0.0
        max_name = None

        # DeepSpeed/FP16 wrappers may store the real optimizer on `.optimizer`/`._optim`.
        opt = optimizer
        for attr in ('optimizer', '_optim', 'optim'):
            if hasattr(opt, attr):
                opt = getattr(opt, attr)

        # Prefer scanning optimizer.state directly (more robust for wrapper optimizers)
        state = getattr(opt, 'state', None)
        if isinstance(state, dict) and state:
            for param, st in state.items():
                if 'exp_avg' in st:
                    local_max = st['exp_avg'].abs().max().item()
                    if local_max > max_m:
                        max_m = local_max
                        max_name = name_by_param.get(param)
            return max_m, max_name

        # Fallback: iterate param_groups if state is empty
        for group in getattr(opt, 'param_groups', []):
            for p in group.get('params', []):
                if p.grad is None:
                    continue
                st = opt.state.get(p, {}) if hasattr(opt, 'state') else {}
                if 'exp_avg' in st:
                    local_max = st['exp_avg'].abs().max().item()
                    if local_max > max_m:
                        max_m = local_max
                        max_name = name_by_param.get(p)
        return max_m, max_name

    def get_statistics(self):
        return {
            'anomaly_count': self.anomaly_count,
            'block_count': self.block_count,
        }

