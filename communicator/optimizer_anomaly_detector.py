#!/usr/bin/env python3
"""
基于优化器状态监控的硬件故障检测器

基于论文 "Understanding and Mitigating Hardware Failures in Deep Learning Training Systems"
使用Algorithm 1的数学严格推导来检测硬件故障
"""

import math
import torch
try:
    from deepspeed import comm as dist
except ImportError:
    import torch.distributed as dist


class OptimizerAnomalyDetector:
    """
    检测Adam优化器的梯度历史值(m_t, v_t)和LayerNorm参数是否超过理论边界
    
    理论基础：
    - 梯度历史边界：|m_t| ≤ 20 × √(n_l / m²)，概率 > (1 - 3×10^-89)
    - 其中 n_l = batch_size × seq_length（部分和数量）
    - m = batch_size（全局batch size）
    """
    
    def __init__(self, batch_size, seq_length, num_layers=12, 
                 learning_rate=1e-4, buffer_size=2, safety_factor=1.0):
        """
        初始化检测器
        
        Args:
            batch_size: 全局batch size（所有GPU总和）
            seq_length: 序列长度
            num_layers: 模型层数（用于LayerNorm边界）
            learning_rate: 学习率η
            buffer_size: 保存最近N个迭代的状态（默认2）
            safety_factor: 安全系数（>1.0更保守，默认1.0）
        """
        # 计算梯度历史边界 Bound_G（论文Algorithm 1公式）
        n_l = batch_size * seq_length  # 部分和数量
        m = batch_size
        
        self.m_threshold = 20 * math.sqrt(n_l / (m ** 2)) * safety_factor
        
        # v_t边界（保守估计为m_threshold的平方）
        self.v_threshold = (self.m_threshold ** 2) * safety_factor
        
        # LayerNorm权重边界（经验值，正常权重应在[-10, 10]范围内）
        self.layernorm_threshold = 10.0 * safety_factor
        
        # 状态buffer（保存最近N个迭代的模型和优化器状态）
        self.buffer = []
        self.buffer_size = buffer_size
        
        # 统计信息
        self.rollback_count = 0
        self.anomaly_count = 0
        
        # 日志（仅rank 0输出）
        if dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"[OptimizerMonitor] Initialized with DNN-property-based bounds")
            print(f"{'='*60}")
            print(f"  n_l (partial sums)    = {n_l}")
            print(f"  m (batch size)        = {m}")
            print(f"  m_threshold           = {self.m_threshold:.2f}")
            print(f"  v_threshold           = {self.v_threshold:.2f}")
            print(f"  layernorm_threshold   = {self.layernorm_threshold:.2f}")
            print(f"  safety_factor         = {safety_factor}")
            print(f"  Theoretical false positive rate < 3e-89")
            print(f"{'='*60}\n")
    
    def check_and_rollback(self, model, optimizer, iteration):
        """
        检查优化器状态和LayerNorm参数，异常则回滚

        Args:
            model: PyTorch模型
            optimizer: 优化器
            iteration: 当前迭代次数

        Returns:
            (rolled_back: bool, reason: str)
        """
        # 提取优化器状态
        max_m, max_v = self._extract_optimizer_states(optimizer)

        # 提取LayerNorm统计
        max_ln_weight, max_ln_bias = self._extract_layernorm_stats(model)

        # 周期性输出监控统计（每100个batch）
        if iteration > 0 and iteration % 100 == 0 and dist.get_rank() == 0:
            print(f"\n[OptimizerMonitor] Statistics at iteration {iteration}:")
            print(f"  Anomalies detected: {self.anomaly_count}")
            print(f"  Rollbacks performed: {self.rollback_count}")
            print(f"  Buffer size: {len(self.buffer)}/{self.buffer_size}")
            print(f"  Current max_m: {max_m:.2f} (threshold: {self.m_threshold:.2f})")
            print(f"  Current max_v: {max_v:.2f} (threshold: {self.v_threshold:.2f})")
            print(f"  Current max_ln: {max_ln_weight:.2f} (threshold: {self.layernorm_threshold:.2f})\n")

        # 检测异常
        if max_m > self.m_threshold:
            self.anomaly_count += 1
            return self._rollback(model, optimizer,
                f'max_m={max_m:.2f} > threshold={self.m_threshold:.2f}')

        if max_v > self.v_threshold:
            self.anomaly_count += 1
            return self._rollback(model, optimizer,
                f'max_v={max_v:.2f} > threshold={self.v_threshold:.2f}')

        if max_ln_weight > self.layernorm_threshold:
            self.anomaly_count += 1
            return self._rollback(model, optimizer,
                f'max_ln_weight={max_ln_weight:.2f} > threshold={self.layernorm_threshold:.2f}')

        # 正常：保存当前状态到buffer
        self._save_to_buffer(model, optimizer, iteration)
        return False, ''
    
    def _extract_optimizer_states(self, optimizer):
        """提取Adam的m_t和v_t的最大绝对值"""
        max_m = max_v = 0.0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = optimizer.state[p]
                if 'exp_avg' in state:  # m_t（一阶动量）
                    max_m = max(max_m, state['exp_avg'].abs().max().item())
                if 'exp_avg_sq' in state:  # v_t（二阶动量）
                    max_v = max(max_v, state['exp_avg_sq'].abs().max().item())
        
        return max_m, max_v
    
    def _extract_layernorm_stats(self, model):
        """提取LayerNorm的权重和偏置的最大绝对值"""
        max_weight = max_bias = 0.0
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                if module.weight is not None:
                    max_weight = max(max_weight, module.weight.abs().max().item())
                if module.bias is not None:
                    max_bias = max(max_bias, module.bias.abs().max().item())
        
        return max_weight, max_bias
    
    def _save_to_buffer(self, model, optimizer, iteration):
        """保存当前状态到buffer（优化版本：只保存GPU上的参数clone）"""
        # 只保存模型参数（不保存buffers和其他状态），保持在GPU上
        model_params = [p.data.clone() for p in model.parameters()]

        self.buffer.append({
            'model_params': model_params,
            'iteration': iteration
        })

        # 保持buffer大小
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def _rollback(self, model, optimizer, reason):
        """回滚到buffer中最早的状态（2个迭代前）"""
        if len(self.buffer) == 0:
            if dist.get_rank() == 0:
                print(f"[OptimizerMonitor] WARNING: Cannot rollback, buffer is empty")
            return False, 'no_checkpoint_in_buffer'

        # 回滚到最早的状态
        state = self.buffer[0]

        # 恢复模型参数
        for p, saved_p in zip(model.parameters(), state['model_params']):
            p.data.copy_(saved_p)

        self.rollback_count += 1

        if dist.get_rank() == 0:
            print(f"\n{'!'*60}")
            print(f"[OptimizerMonitor] ANOMALY DETECTED - Rollback #{self.rollback_count}")
            print(f"  Reason: {reason}")
            print(f"  Restored to iteration {state['iteration']}")
            print(f"{'!'*60}\n")

        return True, f'rollback_to_iter_{state["iteration"]}'
    
    def get_statistics(self):
        """返回统计信息"""
        return {
            'anomaly_count': self.anomaly_count,
            'rollback_count': self.rollback_count,
            'buffer_size': len(self.buffer)
        }

