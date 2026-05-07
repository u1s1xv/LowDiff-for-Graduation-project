
# LowDiff: AI Agent Coding Guide

## 项目概览
LowDiff 是一个高性能分布式训练系统，支持高频（每迭代）检查点，利用梯度压缩实现极低 (<3.1%) 性能开销。主要用于多 GPU/多节点环境下的深度学习训练，具备容错与故障注入能力。

## 架构与核心组件
- **communicator/lowdiff.py**：Top-K 梯度压缩与异步分发、批量检查点写盘、分布式同步。
- **communicator/smart_checkpoint.py**：智能检查点管理，控制全量/差分频率与垃圾回收，仅 rank-0 执行 I/O。
- **communicator/pipeline_recovery.py**：流水线式差分检查点恢复（加载/解压/应用并行）。
- **communicator/fault_injector.py**：泊松过程模拟硬件故障，分布式同步崩溃，避免与梯度相关。
- **communicator/sdc_injector.py**：概率性梯度扰动（SDC）注入，当前仅 rank-0，未来可扩展 ISCA 风格。
- **communicator/optimizer_anomaly_detector.py**：优化器状态异常检测（阻断 step 机制）。
- **communicator/merge_worker.py**：并行解压缩检查点，多 GPU 分担恢复负载。
- **torch/**：训练驱动脚本（cv.py、GPT.py、pipeline_recovery.py 等），均集成 Communicator 钩子。

## 关键数据流
**训练/检查点流程**：
1. 反向传播后注册钩子，触发梯度 Top-K 压缩
2. 异步 all_gather 汇总梯度
3. rank-0 后台进程批量写盘（save_batch_freq 控制）
4. 智能管理器决定是否全量检查点、清理旧文件

**恢复流程**：
1. 读取最新全量检查点
2. 串行或流水线回放差分检查点
3. scatter_add_ 还原梯度
4. 恢复模型参数，继续训练

## 开发者工作流
- 推荐通过 `scripts/cv_lowdiff.sh` 或 `scripts/gpt_lowdiff.sh` 启动训练，环境变量可配置数据集、模型、压缩率等
- 分布式训练统一用 DeepSpeed (`--hostfile`)，多节点需编辑 `scripts/hostfile`
- 仅做流水线恢复时使用 `torch/pipeline_recovery.py`
- 仅做串行恢复时使用 `torch/GPT.py`
- 基线性能对比用 `torch/baseline_training.py`（无压缩/检查点）
- 结果分析脚本见 `scripts/plot_*.py`

## 项目约定与模式
- **分布式 I/O**：所有写盘操作仅 rank-0 执行，需 `dist.barrier()` 保证同步
- **压缩格式**：梯度以 (indices, values) tuple 存储，解压用 scatter_add_ 还原
- **检查点命名**：
	- 全量：`{model}_{dataset}_{compressor}_{ratio}_{epoch}_{batch}_full.pth.tar`
	- 差分：`{model}_{dataset}_{compressor}_{ratio}_{epoch}-{batch}_batch{save_batch_freq}.pth.tar`
- **错误处理**：任一 rank 出错需广播并全体退出，故障注入用 `os._exit(1)`
- **内存管理**：压缩数据存于 self.compression_data，解压后及时清理
- **SDC 注入**：仅本地扰动梯度，勿与 crash 注入混用，参数需实验日志记录
- **智能检查点**：启动时重建全量检查点历史，避免恢复后间隔判断失效

## 依赖与集成点
- **DeepSpeed**：分布式训练与 NCCL 通信
- **PyTorch**：模型与分布式原语
- **Transformers/Datasets**：NLP 任务数据与模型
- **cross-component**：训练脚本统一注册 Communicator 钩子，merge_worker 仅恢复时调用

## 不可修改的接口（保持兼容性）
- communicator/lowdiff.py::Communicator.topk_compress
- communicator/merge_worker.py::topk_decompress
- communicator/fault_injector.py::should_crash

## 常见调试建议
- 检查点未保存：确认 `save_batch_freq` 与文件名后缀一致，且 rank-0 后台进程存活
- 分布式死锁：检查 barrier/broadcast 调用对称
- 解压失败：scatter_add_ 目标 tensor 需与 indices/values 同设备
- SDC 导致溢出/NaN：降低 `param_fraction` 或注入概率
- 智能清理不生效：检查保存目录中文件名是否匹配正则（epoch/batch 格式）

---
**最后更新**：2026-05-05
**适用范围**：分布式深度学习训练/容错研究
