import os
import re
import time
import threading
import queue
import torch


class PipelineCheckpointRecovery:
    """
    流水线检查点恢复器，提供四组恢复方法：

    流水线模式（供 pipeline_recovery.py 使用）：
      - replay_differential_checkpoints       3级流水线：load线程 → decompress线程 → 主线程apply
      - replay_batched_differential_checkpoints  同上，处理批量打包的差分文件

    串行模式（供 GPT.py 对照实验使用）：
      - replay_serial_differential_checkpoints       单线程 load→decompress→step
      - replay_serial_batched_differential_checkpoints  同上，批量版

    流水线设计的核心前提：load(磁盘DMA)、decompress(PCIe DMA)、step(GPU SM)
    三阶段使用独立硬件引擎，通过多线程实现重叠执行。
    解压线程内不使用 ThreadPoolExecutor——_topk_decompress 内部 CUDA 操作
    会释放 GIL，参数级线程池的调度开销反而大于收益。
    """

    def __init__(
        self,
        model,
        optimizer,
        save_dir,
        model_name,
        dataset_name,
        compressor_name,
        compressor_ratio,
        save_batch_freq=1,
        buffer_size=2,
        rank=0,
        device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.compressor_name = compressor_name
        self.compressor_ratio = compressor_ratio
        self.save_batch_freq = int(save_batch_freq)
        self.buffer_size = buffer_size
        self.rank = rank
        self.device = device or next(self.model.parameters()).device
        self._parameter_names = {name: param for name, param in self.model.named_parameters()}

    def load_latest_full_checkpoint(self, verbose=False):
        if verbose:
            discovery_start = time.time()

        pattern = r"{}_{}_{}_{}_([0-9]+)_([0-9]+)_full\.pth\.tar".format(
            self.model_name,
            self.dataset_name,
            self.compressor_name,
            self.compressor_ratio,
        )
        files = os.listdir(self.save_dir)
        candidates = []
        for f in files:
            m = re.match(pattern, f)
            if m:
                epoch = int(m.group(1))
                batch = int(m.group(2))
                candidates.append((epoch, batch, f))

        if not candidates:
            raise ValueError("No full checkpoint found in {}".format(self.save_dir))

        candidates.sort(key=lambda x: (x[0], x[1]))
        sel_epoch, sel_batch, sel_file = candidates[-1]
        filepath = os.path.join(self.save_dir, sel_file)

        if self.rank == 0:
            print(f"[PipelineRecovery] Loading base checkpoint: {sel_file}")
            if verbose:
                file_size_mb = os.path.getsize(filepath) / (1024**2)
                print(f"[PipelineRecovery] File size: {file_size_mb:.2f} MB")
                print(f"[PipelineRecovery] Discovery time: {time.time() - discovery_start:.3f}s")

        if verbose:
            load_start = time.time()
        checkpoint = torch.load(filepath, map_location="cpu")
        if verbose and self.rank == 0:
            load_time = time.time() - load_start
            print(f"[PipelineRecovery] Load time: {load_time:.3f}s")

        if verbose:
            restore_start = time.time()
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if verbose and self.rank == 0:
            restore_time = time.time() - restore_start
            print(f"[PipelineRecovery] Restore time: {restore_time:.3f}s")

        return self.model, self.optimizer, sel_epoch, sel_batch

    def _diff_checkpoint_pattern(self, base_epoch):
        return r"{}_{}_{}_{}_{}-(\d+)_batch{}\.pth\.tar".format(
            self.model_name,
            self.dataset_name,
            self.compressor_name,
            self.compressor_ratio,
            base_epoch,
            self.save_batch_freq,
        )

    def _parse_diff_filename(self, filename):
        pattern = r"^{}_{}_{}_{}_(\d+)-(\d+)_batch(\d+)\.pth\.tar$".format(
            self.model_name,
            self.dataset_name,
            self.compressor_name,
            self.compressor_ratio,
        )
        match = re.match(pattern, filename)
        if not match:
            return None
        epoch = int(match.group(1))
        batch = int(match.group(2))
        batch_freq = int(match.group(3))
        return epoch, batch, batch_freq

    def _list_diff_checkpoint_files(self, base_epoch, base_batch):
        files = []
        for filename in os.listdir(self.save_dir):
            parsed = self._parse_diff_filename(filename)
            if not parsed:
                continue
            epoch, batch, _ = parsed
            if epoch != base_epoch or batch <= base_batch:
                continue
            files.append((batch, os.path.join(self.save_dir, filename)))
        files.sort(key=lambda x: x[0])
        return files

    def _is_param_diff_dict(self, obj):
        """检查对象是否为 Top-K 压缩格式的参数字典：{'param_name': {'values':..., 'indices':..., 'shape':...}}"""
        if not isinstance(obj, dict) or not obj:
            return False
        sample = next(iter(obj.values()))
        return isinstance(sample, dict) and {"values", "indices", "shape"}.issubset(sample.keys())

    def _iter_checkpoint_entries(self, checkpoint, fallback_batch):
        """
        从检查点文件中提取 (batch_idx, diff_dict) 条目。

        兼容两种存储格式：
        1. 批量格式：{iter_idx: diff_dict, ...}——key 为整数
        2. 单条格式：直接是 diff_dict——用 fallback_batch 作为索引

        注意 value 可能被包装在单元素 tuple 中（某些序列化路径的副作用），
        需要解包后再传给 _is_param_diff_dict 判断。
        """
        entries = []
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            if keys and all(isinstance(k, int) for k in keys):
                for iter_idx, value in checkpoint.items():
                    diff = value[0] if isinstance(value, tuple) else value
                    if self._is_param_diff_dict(diff):
                        entries.append((iter_idx, diff))
            elif self._is_param_diff_dict(checkpoint):
                entries.append((fallback_batch, checkpoint))
        entries.sort(key=lambda x: x[0])
        return entries

    def find_max_diff_checkpoint(self, base_epoch, base_batch):
        files = self._list_diff_checkpoint_files(base_epoch, base_batch)
        if not files:
            return -1
        return files[-1][0]

    def _diff_checkpoint_path(self, base_epoch, batch_idx):
        return os.path.join(
            self.save_dir,
            "{}_{}_{}_{}_{}-{}_batch{}.pth.tar".format(
                self.model_name,
                self.dataset_name,
                self.compressor_name,
                self.compressor_ratio,
                base_epoch,
                batch_idx,
                self.save_batch_freq,
            ),
        )

    def _topk_decompress(self, values, indices, shape):
        """
        将 Top-K 压缩的稀疏梯度还原为完整稠密张量。

        non_blocking=True 是关键：CPU→GPU 的 DMA 传输异步发起后立即返回，
        CPU 可继续处理下一个参数，DMA 引擎在后台搬运数据。
        scatter_add_ 提交 GPU kernel 后同样释放 GIL，
        这使得解压线程与主线程的 optimizer.step() 可以在不同硬件引擎上并行。
        """
        tensor_decompressed = torch.zeros(shape, device=self.device).view(-1)
        if isinstance(values, list):
            for idx_tensor, val_tensor in zip(indices, values):
                idx_tensor = idx_tensor.to(self.device, non_blocking=True)
                if idx_tensor.dtype != torch.int64:
                    idx_tensor = idx_tensor.to(torch.int64)
                val_tensor = val_tensor.to(self.device, non_blocking=True)
                if val_tensor.dtype != tensor_decompressed.dtype:
                    val_tensor = val_tensor.to(tensor_decompressed.dtype)
                tensor_decompressed = tensor_decompressed.scatter_add_(0, idx_tensor, val_tensor)
        else:
            values = values.to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True)
            if indices.dtype != torch.int64:
                indices = indices.to(torch.int64)
            if values.dtype != tensor_decompressed.dtype:
                values = values.to(tensor_decompressed.dtype)
            tensor_decompressed = tensor_decompressed.scatter_add_(0, indices, values)
        return tensor_decompressed.view(shape)

    def replay_differential_checkpoints(self, base_epoch, base_batch):
        total_start = time.time()
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"[PipelineRecovery] Pipeline Diff Recovery")
            print(f"[PipelineRecovery] Buffer size: {self.buffer_size}")
            print(f"{'='*60}")

        discovery_start = time.time()
        file_list = self._list_diff_checkpoint_files(base_epoch, base_batch)
        discovery_time = time.time() - discovery_start

        if not file_list:
            if self.rank == 0:
                print("[PipelineRecovery] No diff checkpoint files found")
            return base_batch

        num_checkpoints = len(file_list)
        if self.rank == 0:
            print(f"[PipelineRecovery] Found {num_checkpoints} diff checkpoints (discovery: {discovery_time:.3f}s)")

        load_queue = queue.Queue(maxsize=4)
        decompress_queue = queue.Queue(maxsize=2)
        # 用列表包装异常，使闭包内的赋值对外层可见（Python 闭包限制）
        exception_holder = [None]

        def loader_thread():
            try:
                for batch_idx, filepath in file_list:
                    # map_location="cpu" 避免加载时占用 GPU 显存，解压时再按需搬运
                    checkpoint = torch.load(filepath, map_location="cpu")
                    load_queue.put((batch_idx, checkpoint))
                load_queue.put(None)
            except Exception as e:
                exception_holder[0] = e
                load_queue.put(None)

        def decompress_thread():
            try:
                while True:
                    item = load_queue.get()
                    if item is None:
                        decompress_queue.put(None)
                        break
                    batch_idx, checkpoint = item
                    entries = self._iter_checkpoint_entries(checkpoint, batch_idx)
                    for iter_idx, diff in entries:
                        decompressed = {}
                        for key in diff.keys():
                            decompressed[key] = self._topk_decompress(
                                diff[key]["values"],
                                diff[key]["indices"],
                                diff[key]["shape"],
                            )
                        decompress_queue.put((iter_idx, decompressed))
            except Exception as e:
                exception_holder[0] = e
                decompress_queue.put(None)

        loader = threading.Thread(target=loader_thread, daemon=True)
        decompressor = threading.Thread(target=decompress_thread, daemon=True)

        loader.start()
        decompressor.start()

        last_batch = base_batch
        processed = 0

        while True:
            if exception_holder[0]:
                raise exception_holder[0]

            item = decompress_queue.get()
            if item is None:
                break

            batch_idx, decompressed = item
            for key, tensor in decompressed.items():
                param = self._parameter_names.get(key)
                if param is not None:
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    param.grad = tensor
            self.optimizer.step()

            last_batch = batch_idx
            processed += 1
            if self.rank == 0 and (processed % 5 == 0 or processed == num_checkpoints):
                print(f"[PipelineRecovery] Progress: {processed}/{num_checkpoints} (batch {batch_idx})")

        loader.join(timeout=5)
        decompressor.join(timeout=5)

        total_time = time.time() - total_start
        if self.rank == 0:
            print(f"[PipelineRecovery] {'='*60}")
            print(f"[PipelineRecovery] Diff Recovery Summary:")
            print(f"[PipelineRecovery]   Checkpoints processed: {processed}")
            print(f"[PipelineRecovery]   Total recovery time: {total_time:.3f}s")
            if processed > 0:
                print(f"[PipelineRecovery]   Avg per checkpoint: {total_time/processed:.4f}s")
            print(f"[PipelineRecovery]   Last batch: {last_batch}")
            print(f"[PipelineRecovery] {'='*60}")

        return last_batch

    def replay_batched_differential_checkpoints(self, base_epoch, base_batch):
        total_start = time.time()
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"[PipelineRecovery] Pipeline Batch Recovery")
            print(f"[PipelineRecovery] Buffer size: {self.buffer_size}, Batch freq: {self.save_batch_freq}")
            print(f"{'='*60}")

        discovery_start = time.time()
        batch_files = self._list_diff_checkpoint_files(base_epoch, base_batch)
        discovery_time = time.time() - discovery_start

        if not batch_files:
            if self.rank == 0:
                print("[PipelineRecovery] No batch checkpoint files found")
            return base_batch

        num_batch_files = len(batch_files)
        if self.rank == 0:
            print(f"[PipelineRecovery] Found {num_batch_files} batch files (discovery: {discovery_time:.3f}s)")

        load_queue = queue.Queue(maxsize=self.buffer_size)
        decompress_queue = queue.Queue(maxsize=self.buffer_size * self.save_batch_freq)
        # 用列表包装异常，使闭包内的赋值对外层可见
        exception_holder = [None]

        def loader_thread():
            try:
                for file_batch_idx, filepath in batch_files:
                    # map_location="cpu" 避免加载时占用 GPU 显存
                    checkpoint = torch.load(filepath, map_location="cpu")
                    load_queue.put((file_batch_idx, checkpoint))
                load_queue.put(None)
            except Exception as e:
                exception_holder[0] = e
                load_queue.put(None)

        def decompress_thread():
            try:
                while True:
                    item = load_queue.get()
                    if item is None:
                        decompress_queue.put(None)
                        break

                    file_batch_idx, checkpoint = item
                    entries = self._iter_checkpoint_entries(checkpoint, file_batch_idx)
                    for iter_idx, diff in entries:
                        decompressed = {}
                        for key in diff.keys():
                            decompressed[key] = self._topk_decompress(
                                diff[key]["values"],
                                diff[key]["indices"],
                                diff[key]["shape"],
                            )
                        decompress_queue.put((iter_idx, decompressed))
            except Exception as e:
                exception_holder[0] = e
                decompress_queue.put(None)

        loader = threading.Thread(target=loader_thread, daemon=True)
        decompressor = threading.Thread(target=decompress_thread, daemon=True)

        loader.start()
        decompressor.start()

        last_batch = base_batch
        processed_entries = 0

        while True:
            if exception_holder[0]:
                raise exception_holder[0]

            item = decompress_queue.get()
            if item is None:
                break

            iter_idx, decompressed = item
            if iter_idx <= base_batch:
                continue

            for key, tensor in decompressed.items():
                param = self._parameter_names.get(key)
                if param is not None:
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    param.grad = tensor
            self.optimizer.step()

            last_batch = iter_idx
            processed_entries += 1

            if self.rank == 0 and processed_entries % 5 == 0:
                print(f"[PipelineRecovery] Progress: {processed_entries} entries (up to batch {iter_idx})")

        loader.join(timeout=5)
        decompressor.join(timeout=5)

        total_time = time.time() - total_start
        if self.rank == 0:
            print(f"[PipelineRecovery] {'='*60}")
            print(f"[PipelineRecovery] Batch Recovery Summary:")
            print(f"[PipelineRecovery]   Batch files processed: {num_batch_files}")
            print(f"[PipelineRecovery]   Entries replayed: {processed_entries}")
            print(f"[PipelineRecovery]   Total recovery time: {total_time:.3f}s")
            if processed_entries > 0:
                print(f"[PipelineRecovery]   Avg per entry: {total_time/processed_entries:.4f}s")
            print(f"[PipelineRecovery]   Last batch: {last_batch}")
            print(f"[PipelineRecovery] {'='*60}")

        return last_batch

    def replay_serial_differential_checkpoints(self, base_epoch, base_batch):
        """Serial recovery: single-threaded load→decompress→step, for baseline comparison."""
        total_start = time.time()
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"[PipelineRecovery] Serial Diff Recovery (baseline)")
            print(f"{'='*60}")

        discovery_start = time.time()
        file_list = self._list_diff_checkpoint_files(base_epoch, base_batch)
        discovery_time = time.time() - discovery_start

        if not file_list:
            if self.rank == 0:
                print("[PipelineRecovery] No diff checkpoint files found")
            return base_batch

        num_checkpoints = len(file_list)
        if self.rank == 0:
            print(f"[PipelineRecovery] Found {num_checkpoints} diff checkpoints (discovery: {discovery_time:.3f}s)")

        recovery_times = []
        load_times = []
        decompress_times = []
        apply_times = []
        last_batch = base_batch

        for idx, (batch_idx, filepath) in enumerate(file_list):
            iter_start = time.time()

            load_start = time.time()
            checkpoint = torch.load(filepath, map_location="cpu")
            load_time = time.time() - load_start
            load_times.append(load_time)

            entries = self._iter_checkpoint_entries(checkpoint, batch_idx)
            for iter_idx, diff in entries:
                decompress_start = time.time()
                for key in diff.keys():
                    tensor = self._topk_decompress(
                        diff[key]["values"], diff[key]["indices"], diff[key]["shape"])
                    param = self._parameter_names.get(key)
                    if param is not None:
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        param.grad = tensor
                decompress_time = time.time() - decompress_start
                decompress_times.append(decompress_time)

                apply_start = time.time()
                self.optimizer.step()
                apply_time = time.time() - apply_start
                apply_times.append(apply_time)

                last_batch = iter_idx

            iter_time = time.time() - iter_start
            recovery_times.append(iter_time)

            if self.rank == 0 and (idx % 5 == 0 or idx == num_checkpoints - 1):
                print(f"[PipelineRecovery] Checkpoint {idx+1}/{num_checkpoints} (batch {batch_idx})")
                print(f"       Load: {load_time:.4f}s | Decompress: {decompress_time:.4f}s | "
                      f"Apply: {apply_time:.4f}s | Total: {iter_time:.4f}s")

        total_time = time.time() - total_start
        if self.rank == 0:
            print(f"[PipelineRecovery] {'='*60}")
            print(f"[PipelineRecovery] Serial Diff Recovery Summary:")
            print(f"[PipelineRecovery]   Checkpoints processed: {num_checkpoints}")
            print(f"[PipelineRecovery]   Total recovery time: {total_time:.3f}s")
            if recovery_times:
                print(f"[PipelineRecovery]   Avg per checkpoint: {sum(recovery_times)/len(recovery_times):.4f}s")
            print(f"[PipelineRecovery]   Last batch: {last_batch}")
            print(f"[PipelineRecovery] {'='*60}")

        return last_batch

    def replay_serial_batched_differential_checkpoints(self, base_epoch, base_batch):
        """Serial batch recovery: single-threaded, for baseline comparison."""
        total_start = time.time()
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"[PipelineRecovery] Serial Batch Recovery (baseline)")
            print(f"[PipelineRecovery] Batch freq: {self.save_batch_freq}")
            print(f"{'='*60}")

        discovery_start = time.time()
        batch_files = self._list_diff_checkpoint_files(base_epoch, base_batch)
        discovery_time = time.time() - discovery_start

        if not batch_files:
            if self.rank == 0:
                print("[PipelineRecovery] No batch checkpoint files found")
            return base_batch

        num_batch_files = len(batch_files)
        if self.rank == 0:
            print(f"[PipelineRecovery] Found {num_batch_files} batch files (discovery: {discovery_time:.3f}s)")

        batch_times = []
        load_times = []
        apply_times = []
        entries_processed = 0
        last_batch = base_batch

        for idx, (file_batch_idx, filepath) in enumerate(batch_files):
            batch_start = time.time()

            load_start = time.time()
            checkpoint = torch.load(filepath, map_location="cpu")
            load_time = time.time() - load_start
            load_times.append(load_time)

            entries = self._iter_checkpoint_entries(checkpoint, file_batch_idx)
            ckpts_in_batch = 0
            for iter_idx, diff in entries:
                if iter_idx <= base_batch:
                    continue

                apply_start = time.time()
                for key in diff.keys():
                    tensor = self._topk_decompress(
                        diff[key]["values"], diff[key]["indices"], diff[key]["shape"])
                    param = self._parameter_names.get(key)
                    if param is not None:
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        param.grad = tensor
                self.optimizer.step()
                apply_time = time.time() - apply_start
                apply_times.append(apply_time)

                last_batch = iter_idx
                ckpts_in_batch += 1
                entries_processed += 1

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if self.rank == 0:
                print(f"[PipelineRecovery] Batch file {idx+1}/{num_batch_files} "
                      f"(entries: {ckpts_in_batch}, time: {batch_time:.3f}s)")

        total_time = time.time() - total_start
        if self.rank == 0:
            print(f"[PipelineRecovery] {'='*60}")
            print(f"[PipelineRecovery] Serial Batch Recovery Summary:")
            print(f"[PipelineRecovery]   Batch files processed: {num_batch_files}")
            print(f"[PipelineRecovery]   Entries replayed: {entries_processed}")
            print(f"[PipelineRecovery]   Total recovery time: {total_time:.3f}s")
            if entries_processed > 0:
                print(f"[PipelineRecovery]   Avg per entry: {total_time/entries_processed:.4f}s")
            print(f"[PipelineRecovery]   Last batch: {last_batch}")
            print(f"[PipelineRecovery] {'='*60}")

        return last_batch
