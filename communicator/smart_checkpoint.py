"""
智能检查点管理模块
实现检查点清理功能
"""
import os


class SmartCheckpointManager:
    """智能检查点管理器：负责清理过期的全量检查点和差分检查点"""

    def __init__(self, save_dir, model_name, dataset_name, compressor, ratio,
                 max_full_interval=100, keep_full_checkpoints=3):
        """
        Args:
            save_dir: 检查点保存目录
            model_name: 模型名称
            dataset_name: 数据集名称
            compressor: 压缩器名称
            ratio: 压缩比例
            max_full_interval: 全量检查点最大间隔（保底策略）
            keep_full_checkpoints: 保留最近N个全量检查点
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.compressor = compressor
        self.ratio = ratio

        self.max_full_interval = max_full_interval
        self.last_full_checkpoint = 0
        self.keep_full_checkpoints = keep_full_checkpoints

        self.full_checkpoint_history = []  # 记录全量检查点的(epoch, iteration)

    def should_save_full_checkpoint(self, iteration, epoch, loss=None):
        """
        决策是否保存全量检查点（仅基于固定间隔）

        Args:
            iteration: 当前训练迭代次数
            epoch: 当前epoch
            loss: 当前loss值（保留参数以兼容旧代码，但不使用）

        Returns:
            (should_save, reason): 是否保存及原因
        """
        # 规则：最大间隔保护（保底策略）
        if iteration - self.last_full_checkpoint >= self.max_full_interval:
            self.last_full_checkpoint = iteration
            self.full_checkpoint_history.append((epoch, iteration))
            return True, 'max_interval'

        return False, 'skip'

    def cleanup_old_full_checkpoints(self):
        """
        清理旧的全量检查点（保留最近N个）
        """
        if len(self.full_checkpoint_history) <= self.keep_full_checkpoints:
            return  # 不需要清理

        # 计算需要删除的检查点
        checkpoints_to_delete = self.full_checkpoint_history[:-self.keep_full_checkpoints]

        cleanup_count = 0
        for epoch, batch in checkpoints_to_delete:
            full_path = self._get_full_checkpoint_path(epoch, batch)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    cleanup_count += 1
                except Exception as e:
                    print(f"[SmartCkpt] Warning: Failed to remove {full_path}: {e}")

        # 更新历史记录
        self.full_checkpoint_history = self.full_checkpoint_history[-self.keep_full_checkpoints:]

        if cleanup_count > 0:
            print(f"[SmartCkpt] Cleaned up {cleanup_count} old full checkpoints (keeping {self.keep_full_checkpoints} most recent)")
    
    def cleanup_old_diff_checkpoints(self, current_iteration, epoch):
        """
        清理旧的差分检查点（保留最近两个全量检查点之间的差分）

        Args:
            current_iteration: 当前迭代次数
            epoch: 当前epoch
        """
        if len(self.full_checkpoint_history) < 2:
            return  # 至少需要2个全量检查点才能清理

        # 找到倒数第二个全量检查点
        second_last_epoch, second_last_batch = self.full_checkpoint_history[-2]

        # 如果是第一个epoch，清理从0到second_last_batch的所有差分检查点
        # 否则清理从上一个epoch的所有差分检查点
        cleanup_count = 0

        # 扫描保存目录中的所有差分检查点文件
        try:
            for filename in os.listdir(self.save_dir):
                # 匹配差分检查点文件名格式：model_dataset_compressor_ratio_epoch-batch_batchN.pth.tar
                if filename.endswith('_batch1.pth.tar') or filename.endswith(f'_batch{20}.pth.tar'):
                    # 解析文件名获取epoch和batch信息
                    parts = filename.replace('.pth.tar', '').split('_')
                    if len(parts) >= 5:
                        try:
                            # 提取epoch-batch部分
                            epoch_batch_str = parts[-2]  # 例如 "0-100"
                            if '-' in epoch_batch_str:
                                file_epoch, file_batch = map(int, epoch_batch_str.split('-'))

                                # 删除倒数第二个全量检查点之前的差分检查点
                                if (file_epoch < second_last_epoch) or \
                                   (file_epoch == second_last_epoch and file_batch < second_last_batch):
                                    filepath = os.path.join(self.save_dir, filename)
                                    os.remove(filepath)
                                    cleanup_count += 1
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"[SmartCkpt] Warning: Failed to scan directory {self.save_dir}: {e}")

        if cleanup_count > 0:
            print(f"[SmartCkpt] Cleaned up {cleanup_count} old diff checkpoints before epoch {second_last_epoch} batch {second_last_batch}")

    def _get_diff_checkpoint_path(self, epoch, batch):
        """构造差分检查点文件路径"""
        filename = f"{self.model_name}_{self.dataset_name}_{self.compressor}_{self.ratio}_{epoch}-{batch}_batch1.pth.tar"
        return os.path.join(self.save_dir, filename)

    def _get_full_checkpoint_path(self, epoch, batch):
        """构造全量检查点文件路径"""
        filename = f"{self.model_name}_{self.dataset_name}_{self.compressor}_{self.ratio}_{epoch}_{batch}_full.pth.tar"
        return os.path.join(self.save_dir, filename)

