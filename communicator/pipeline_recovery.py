import os
import re
import time
import threading
import queue
import torch


class PipelineCheckpointRecovery:
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

    def load_latest_full_checkpoint(self):
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

        checkpoint = torch.load(filepath, map_location="cpu")
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

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
        if not isinstance(obj, dict) or not obj:
            return False
        sample = next(iter(obj.values()))
        return isinstance(sample, dict) and {"values", "indices", "shape"}.issubset(sample.keys())

    def _iter_checkpoint_entries(self, checkpoint, fallback_batch):
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
        tensor_decompressed = torch.zeros(shape, device=self.device).view(-1)
        if isinstance(values, list):
            for idx_tensor, val_tensor in zip(indices, values):
                # 优化：异步传输，不阻塞流水线
                idx_tensor = idx_tensor.to(self.device, non_blocking=True)
                if idx_tensor.dtype != torch.int64:
                    idx_tensor = idx_tensor.to(torch.int64)
                val_tensor = val_tensor.to(self.device, non_blocking=True)
                if val_tensor.dtype != tensor_decompressed.dtype:
                    val_tensor = val_tensor.to(tensor_decompressed.dtype)
                tensor_decompressed = tensor_decompressed.scatter_add_(0, idx_tensor, val_tensor)
        else:
            # 优化：异步传输
            values = values.to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True)
            if indices.dtype != torch.int64:
                indices = indices.to(torch.int64)
            if values.dtype != tensor_decompressed.dtype:
                values = values.to(tensor_decompressed.dtype)
            tensor_decompressed = tensor_decompressed.scatter_add_(0, indices, values)
        return tensor_decompressed.view(shape)

    def replay_differential_checkpoints(self, base_epoch, base_batch):
        if self.rank == 0:
            print(f"[PipelineRecovery] Start pipeline diff recovery (buffer={self.buffer_size})")

        file_list = self._list_diff_checkpoint_files(base_epoch, base_batch)
        if not file_list:
            if self.rank == 0:
                print("[PipelineRecovery] No diff checkpoint files found")
            return base_batch

        num_checkpoints = len(file_list)
        if self.rank == 0:
            print(f"[PipelineRecovery] Found {num_checkpoints} diff checkpoint files")

        load_queue = queue.Queue(maxsize=4)  # 优化：增加缓冲区
        decompress_queue = queue.Queue(maxsize=2)
        exception_holder = [None]

        def loader_thread():
            try:
                for batch_idx, filepath in file_list:
                    checkpoint = torch.load(filepath, map_location="cpu")
                    load_queue.put((batch_idx, checkpoint))
                load_queue.put(None)
            except Exception as e:
                exception_holder[0] = e
                load_queue.put(None)

        def decompress_thread():
            import concurrent.futures
            try:
                # 优化：使用线程池并行解压多个参数
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    while True:
                        item = load_queue.get()
                        if item is None:
                            decompress_queue.put(None)
                            break
                        batch_idx, checkpoint = item
                        entries = self._iter_checkpoint_entries(checkpoint, batch_idx)
                        for iter_idx, diff in entries:
                            # 并行解压所有参数
                            futures = {}
                            for key in diff.keys():
                                future = executor.submit(
                                    self._topk_decompress,
                                    diff[key]["values"],
                                    diff[key]["indices"],
                                    diff[key]["shape"],
                                )
                                futures[key] = future

                            # 等待所有参数解压完成
                            decompressed = {key: future.result() for key, future in futures.items()}
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
        start = time.time()

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
            if self.rank == 0 and (processed % 5 == 0 or processed == len(file_list)):
                print(f"[PipelineRecovery] Processed {processed}/{len(file_list)} (batch {batch_idx})")

        loader.join(timeout=5)
        decompressor.join(timeout=5)

        if self.rank == 0:
            print(f"[PipelineRecovery] Done. Processed {processed} checkpoints in {time.time()-start:.3f}s")

        return last_batch

    def replay_batched_differential_checkpoints(self, base_epoch, base_batch):
        if self.rank == 0:
            print(f"[PipelineRecovery] Start pipeline batch recovery (buffer={self.buffer_size})")

        batch_files = self._list_diff_checkpoint_files(base_epoch, base_batch)
        if not batch_files:
            if self.rank == 0:
                print("[PipelineRecovery] No batch checkpoint files found")
            return base_batch

        load_queue = queue.Queue(maxsize=self.buffer_size)
        exception_holder = [None]

        def loader_thread():
            try:
                for file_batch_idx, filepath in batch_files:
                    checkpoint = torch.load(filepath, map_location="cpu")
                    load_queue.put((file_batch_idx, checkpoint))
                load_queue.put(None)
            except Exception as e:
                exception_holder[0] = e
                load_queue.put(None)

        loader = threading.Thread(target=loader_thread, daemon=True)
        loader.start()

        last_batch = base_batch
        processed_files = 0
        start = time.time()

        while True:
            if exception_holder[0]:
                raise exception_holder[0]

            item = load_queue.get()
            if item is None:
                break

            file_batch_idx, checkpoint = item
            processed_files += 1

            entries = self._iter_checkpoint_entries(checkpoint, file_batch_idx)
            for iter_idx, diff in entries:
                if iter_idx <= base_batch:
                    continue
                for key in diff.keys():
                    tensor = self._topk_decompress(
                        diff[key]["values"],
                        diff[key]["indices"],
                        diff[key]["shape"],
                    )
                    param = self._parameter_names.get(key)
                    if param is not None:
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        param.grad = tensor
                self.optimizer.step()
                last_batch = iter_idx

            if self.rank == 0:
                print(
                    f"[PipelineRecovery] Processed batch file {processed_files}/{len(batch_files)} "
                    f"(up to batch {file_batch_idx})"
                )

        loader.join(timeout=5)

        if self.rank == 0:
            print(
                f"[PipelineRecovery] Done. Processed {processed_files} batch files in {time.time()-start:.3f}s"
            )

        return last_batch
