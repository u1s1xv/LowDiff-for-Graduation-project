#!/usr/bin/env python3
"""
静默数据损坏（SDC）故障注入模块

该模块在训练过程中对梯度做受控扰动，模拟硬件静默故障（SDC），
用于验证 `OptimizerAnomalyDetector` 等防护机制的有效性。

设计定位：
- 仅做“梯度级”注入（不做前向/反向层级 hook 注入）。
- 保留比特翻转（`rbflip`）作为唯一故障类型。
"""

import random
import time
import os
import torch
try:
    from deepspeed import comm as dist
except ImportError:
    import torch.distributed as dist


class SDCInjector:
    """
    静默数据损坏注入器

    调用时机：在 `backward()` 之后、`optimizer.step()` 之前。

    分布式语义：默认仅在 rank-0 注入，模拟“单节点被污染”的故障场景。

    注入模式：
    - `rbflip`：位置级指数位翻转
    """

    SUPPORTED_INJ_TYPES = {"rbflip"}

    def __init__(self, inject_prob=0.02, param_fraction=0.1,
                 min_batch=20, enable=False, log_path=None,
                 inj_type="rbflip", positions_per_param=1,
                 target_param_pattern=None):
        """
        Args:
            inject_prob: 每个 batch 触发注入的概率（默认 2%）
            param_fraction: 每次注入时受影响的参数比例（0.0~1.0）
            min_batch: 最小 batch 数后才允许注入
            enable: 是否启用
            log_path: 注入日志路径
            inj_type: 注入类型
                      - rbflip: 对选定位置做随机指数位翻转（两位）
            positions_per_param: 每个参数张量注入的位置数
            target_param_pattern: 仅对名称包含该子串的参数进行注入（None 表示不限制）
        """
        self.inject_prob = inject_prob
        self.param_fraction = param_fraction
        self.min_batch = min_batch
        self.enable = enable
        self.log_path = log_path or "sdc_injection.log"
        self.inj_type = (inj_type or "rbflip").lower()
        self.positions_per_param = max(1, int(positions_per_param))
        self.target_param_pattern = target_param_pattern

        if self.inj_type not in self.SUPPORTED_INJ_TYPES:
            raise ValueError(f"Unsupported inj_type: {self.inj_type}. "
                             f"Supported: {sorted(self.SUPPORTED_INJ_TYPES)}")

        # 统计
        self.inject_count = 0
        self.total_checks = 0

        seed = int(time.time() * 1000) + os.getpid()
        random.seed(seed)

        if self.enable and dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"[SDCInjector] Initialized")
            print(f"  inj_type:        {self.inj_type}")
            print(f"  inject_prob:     {inject_prob}")
            print(f"  param_fraction:  {param_fraction}")
            print(f"  positions/param: {self.positions_per_param}")
            if self.target_param_pattern:
                print(f"  target_param_pattern: {self.target_param_pattern}")
            print(f"  min_batch:       {min_batch}")
            print(f"{'='*60}\n")

    def maybe_inject(self, model, batch_idx, epoch):
        """
        按门控条件决定是否执行注入。

        建议在 `backward()` 之后、`optimizer.step()` 之前调用。

        门控顺序：
        1) `enable` 开关
        2) `batch_idx >= min_batch`
        3) 随机概率 `inject_prob`
        4) rank 限制（仅 rank-0）

        Returns:
            injected (bool): 是否注入了故障
        """
        if not self.enable:
            return False

        self.total_checks += 1

        if batch_idx < self.min_batch:
            return False

        if random.random() >= self.inject_prob:
            return False

        # 只在 rank 0 做注入（分布式场景下模拟单节点故障）
        if dist.get_rank() != 0:
            return False

        return self._inject(model, batch_idx, epoch)

    def _inject(self, model, batch_idx, epoch):
        """
        执行一次注入事件。

        流程：
        - 从有梯度的参数中随机采样 `param_fraction`
        - 对每个目标参数执行对应 `inj_type` 注入
        - 记录注入统计与日志
        """
        params_with_grad = [(n, p) for n, p in model.named_parameters()
                            if p.grad is not None]
        if self.target_param_pattern:
            params_with_grad = [
                (n, p) for n, p in params_with_grad
                if self.target_param_pattern in n
            ]
        if not params_with_grad:
            return False

        # 随机选择部分参数
        num_to_inject = max(1, int(len(params_with_grad) * self.param_fraction))
        targets = random.sample(params_with_grad, num_to_inject)

        injected_names = []
        injected_positions_total = 0
        for name, param in targets:
            # 简化 ISCA 风格：位置级指数位翻转
            injected_positions_total += self._inject_isca_style(param.grad.data)
            injected_names.append(name)

        self.inject_count += 1

        if dist.get_rank() == 0:
            print(f"\n[SDCInjector] INJECTED at epoch {epoch} batch {batch_idx}")
            print(f"  Affected params: {num_to_inject}/{len(params_with_grad)}")
            print(f"  Type: {self.inj_type}")
            print(f"  Positions/param: {self.positions_per_param}")
            print(f"  Total injected positions: {injected_positions_total}")
            print(f"  Injection #{self.inject_count}")
            for n in injected_names[:3]:
                print(f"    - {n}")
            if len(injected_names) > 3:
                print(f"    ... and {len(injected_names)-3} more")

            # 写入日志
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Injection #{self.inject_count} | "
                        f"Epoch {epoch} Batch {batch_idx} | "
                        f"Type: {self.inj_type} | "
                        f"Params: {num_to_inject} | "
                    f"Positions: {injected_positions_total}\n")

        return True

    def _inject_isca_style(self, grad_tensor):
        """
        在梯度张量上做“位置级指数位翻转”。

        注入语义：
        - `rbflip`: 对当前位置值做 2 个指数位随机翻转

        Returns:
            int: 实际修改的位置个数
        """
        flat = grad_tensor.view(-1)
        numel = flat.numel()
        if numel == 0:
            return 0

        k = min(self.positions_per_param, numel)
        chosen = random.sample(range(numel), k)

        with torch.no_grad():
            for idx in chosen:
                old_val = float(flat[idx].item())
                new_val = self._flip_exponent_bits(old_val, flat.dtype)

                # 关键：保证注入值可表示为目标 dtype（如 fp16），避免赋值时溢出崩溃
                new_val = self._sanitize_for_dtype(new_val, flat)
                flat[idx] = new_val

        return k

    @staticmethod
    def _flip_exponent_bits(value, dtype):
        """对浮点数的指数位做随机 bit 翻转并返回新值（两位）。"""
        if dtype == torch.float64:
            int_dtype = torch.int64
            exp_bits = range(52, 63)  # 11 bits
        elif dtype == torch.float32:
            int_dtype = torch.int32
            exp_bits = range(23, 31)  # 8 bits
        elif dtype == torch.float16:
            int_dtype = torch.int16
            exp_bits = range(10, 15)  # 5 bits
        else:
            int_dtype = torch.int32
            exp_bits = range(23, 31)

        bits = torch.tensor(value, dtype=dtype, device='cpu').view(int_dtype)
        flip_bits = random.sample(tuple(exp_bits), k=2)
        for bit in flip_bits:
            bits ^= 1 << bit
        return bits.view(dtype).item()

    @staticmethod
    def _sanitize_for_dtype(value, tensor):
        """
        将注入标量限制到目标张量 dtype 可表示范围内，避免赋值时 overflow。
        """
        # NaN check
        if value != value:
            value = 0.0

        dtype = tensor.dtype
        if dtype.is_floating_point:
            finfo = torch.finfo(dtype)
            max_val = float(finfo.max)
            min_val = -max_val

            if value == float('inf'):
                return max_val
            if value == float('-inf'):
                return min_val
            if value > max_val:
                return max_val
            if value < min_val:
                return min_val
            return float(value)

        # 其他 dtype（理论上梯度一般是浮点），这里做保底处理
        return float(value)

    def get_statistics(self):
        """返回注入器运行统计。"""
        return {
            'inject_count': self.inject_count,
            'total_checks': self.total_checks,
            'inj_type': self.inj_type,
        }

