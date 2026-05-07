import os
import sys
import torch
from deepspeed import comm as dist
# import torch.distributed as dist
import concurrent.futures
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import datetime
import time
import atexit

class Communicator:
    def __init__(self, model, k=0.01, num_threads=None, save_batch_freq=1,
                 enable_smart_checkpoint=False, save_dir=None, model_name=None, dataset_name=None,
                 compressor_name=None, compressor_ratio=None, max_full_interval=30, keep_full_checkpoints=3):
        """
        Initialize the Communicator for Top-K gradient compression with async all_gather.

        Args:
            model (nn.Module): The PyTorch model.
            k (float): Compression ratio (top-k percentage of gradient to keep).
            num_threads (int, optional): Number of threads for decompression.
                                          Defaults to half of CPU cores.
            save_batch_freq (int): In-memory batching frequency for saving compressed gradients.
            enable_smart_checkpoint (bool): Whether to enable smart checkpoint management.
            save_dir (str): Directory to save checkpoints (required if enable_smart_checkpoint=True).
            model_name (str): Model name for checkpoint naming.
            dataset_name (str): Dataset name for checkpoint naming.
            compressor_name (str): Compressor name for checkpoint naming.
            compressor_ratio (float): Compressor ratio for checkpoint naming.
            max_full_interval (int): Maximum interval between full checkpoints (from --freq parameter).
            keep_full_checkpoints (int): Number of recent full checkpoints to keep.
        """
        self.k = k
        self.model = model
        self.compression_data = {}  # Store async work handles and gathered results

        # Smart Checkpoint Management
        self.enable_smart_checkpoint = enable_smart_checkpoint
        self.smart_ckpt_manager = None
        if enable_smart_checkpoint and dist.get_rank() == 0:
            from communicator.smart_checkpoint import SmartCheckpointManager
            if not all([save_dir, model_name, dataset_name, compressor_name, compressor_ratio]):
                raise ValueError("Smart checkpoint requires save_dir, model_name, dataset_name, compressor_name, and compressor_ratio")
            self.smart_ckpt_manager = SmartCheckpointManager(
                save_dir=save_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                compressor=compressor_name,
                ratio=compressor_ratio,
                max_full_interval=max_full_interval,
                keep_full_checkpoints=keep_full_checkpoints,
                save_batch_freq=save_batch_freq
            )
            print(f"[SmartCkpt] Smart checkpoint management ENABLED (max_interval={max_full_interval}, keep={keep_full_checkpoints})")

        # Get the number of available CPU threads (default to half of total cores, max 32)
        if num_threads is None:
            num_threads = int(os.cpu_count() / 2)

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)  # Thread pool
        self.param_dict = dict(self.model.named_parameters())

        print(f"Using {num_threads} threads for gradient decompression.")

        if dist.get_rank() == 0:
            self.save_batch_freq = save_batch_freq
            self.diff_ckpt = {}
            self.queue = mp.Queue()
            # 传递内存监控参数到后台进程
            self.save_process = mp.Process(target=diff_ckpt_saver,
                                args=(self.queue, self.save_batch_freq))
            self.save_process.start()
            print("save process start!")

            # Register cleanup handler
            atexit.register(self._cleanup_save_process)
        
        
    def topk_compress(self, tensor):
        """
        Compress the gradient into Top-K format.
        """
        num_elements = tensor.numel()
        k_elements = max(1, int(num_elements * self.k))

        values, indices = torch.topk(tensor.view(-1).abs(), k_elements, sorted=False)
        values = tensor.view(-1).gather(0, indices)

        return indices, values
        
    def async_send(self, grad, param_name):
        """
        Hook function for gradient compression.
        """
        world_size = dist.get_world_size()

        # Compress the gradient
        indices, values = self.topk_compress(grad)

        gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
        gathered_values = [torch.zeros_like(values) for _ in range(world_size)]

        # Perform async all_gather
        work_indices = dist.all_gather(gathered_indices, indices, async_op=True)
        work_values = dist.all_gather(gathered_values, values, async_op=True)

        # Store work handles and gathered buffers
        self.compression_data[param_name] = {
            "work_indices": work_indices,
            "work_values": work_values,
            "gathered_indices": gathered_indices,
            "gathered_values": gathered_values,
            "grad_shape": grad.shape
        }

        if dist.get_rank() == 0:
            # Store references (will be moved to CPU in decompress_save)
            self.diff_ckpt[param_name] = {'values': gathered_values, 'indices': gathered_indices, 'shape': grad.shape}

        return None  # Do not modify grad immediately
    
    def register_hooks(self):
        """
        Register Top-K compression hooks for model parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.async_send(grad, name))
    
    def get_gradient_norm(self):
        """计算当前所有参数的梯度范数（用于智能检查点决策）"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def decompress_save(self, diff, filename, i):
        """
        Parallel gradient restoration.
        """
        def process_gradient(param_name, param, data):
            data["work_indices"].wait()
            data["work_values"].wait()

            restored_grad = torch.zeros(data["grad_shape"], device=data["gathered_values"][0].device).view(-1)

            for indices, values in zip(data["gathered_indices"], data["gathered_values"]):
                restored_grad.scatter_add_(0, indices, values)

            restored_grad = restored_grad.view(data["grad_shape"])
            param.grad = restored_grad

        # Submit tasks to the thread pool and wait for completion
        futures = [
            self.executor.submit(process_gradient, name, self.param_dict[name], data)
            for name, data in self.compression_data.items()
        ]
        concurrent.futures.wait(futures)

        # Clear stored data
        self.compression_data.clear()

        # Send the compressed gradients to the save process
        if diff and dist.get_rank() == 0:
            if not self.save_process.is_alive():
                print("[Communicator] WARNING: saver process is dead, skipping checkpoint save")
            else:
                # 在主进程做 CPU 转换，确保后台进程收到的全是 CPU tensor
                # 避免后台进程在 CUDA driver 关闭后仍试图访问 GPU
                diff_cpu = _to_cpu(self.diff_ckpt)
                self.queue.put((diff_cpu, filename, i))
            # 保持与论文原始实现一致：每次发送后清空当前 diff_ckpt
            self.diff_ckpt = {}

    def _cleanup_save_process(self):
        """
        Cleanup save process on exit.
        """
        if hasattr(self, 'save_process') and self.save_process.is_alive():
            try:
                self.queue.put(None, timeout=1)
                self.save_process.join(timeout=5)
                if self.save_process.is_alive():
                    self.save_process.terminate()
                    self.save_process.join(timeout=2)
            except Exception:
                pass

    def __del__(self):
        """
        Ensure the thread pool is properly shut down on object destruction.
        """
        self.executor.shutdown(wait=True)
        self._cleanup_save_process()

def diff_ckpt_saver(queue, save_batch_freq):
    """
    Background process that saves compressed gradients to disk.

    Args:
        queue (mp.Queue): Queue receiving data to be saved.
        save_batch_freq (int): Save frequency in terms of batch steps.
    """
    import sys
    import signal

    batch_buffer = {}

    sys.stderr.write("[DiffCkpt] Background saver started\n")
    sys.stderr.flush()

    def flush_buffer():
        """Flush current batch_buffer to disk (best effort)."""
        nonlocal batch_buffer
        if not batch_buffer:
            return
        try:
            begin = time.time()
            iterations = sorted(batch_buffer.keys())
            filename = batch_buffer[iterations[0]][1]
            torch.save(batch_buffer, filename)
            sys.stderr.write(
                f"[DiffCkpt] Flushed {len(batch_buffer)} checkpoints: {os.path.basename(filename)} ({time.time()-begin:.3f}s)\n"
            )
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[DiffCkpt] ERROR: Failed to flush buffer: {e}\n")
            sys.stderr.flush()
        finally:
            batch_buffer = {}

    def emergency_flush():
        """Emergency flush on signal or exception"""
        if batch_buffer:
            sys.stderr.write(f"[DiffCkpt] Emergency flush: saving {len(batch_buffer)} checkpoints...\n")
            sys.stderr.flush()
            flush_buffer()

    def signal_handler(signum, frame):
        """Handle termination signals"""
        sys.stderr.write(f"[DiffCkpt] Received signal {signum}, performing emergency flush...\n")
        sys.stderr.flush()
        emergency_flush()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            data = queue.get()

            # Termination signal
            if data is None:
                sys.stderr.write("[DiffCkpt] Termination signal received, exiting...\n")
                sys.stderr.flush()
                break

            # Flush command: write all buffered checkpoints to disk
            if isinstance(data, str) and data == 'FLUSH':
                flush_buffer()
                continue

            # Clear command: remove checkpoints before specified iteration
            if isinstance(data, str) and data.startswith('CLEAR_BEFORE:'):
                try:
                    parts = data.split(':')
                    clear_iteration = int(parts[1])
                    before_count = len(batch_buffer)
                    batch_buffer = {k: v for k, v in batch_buffer.items() if k >= clear_iteration}
                    cleared = before_count - len(batch_buffer)
                    if cleared > 0:
                        sys.stderr.write(
                            f"[DiffCkpt] Cleared {cleared} buffered checkpoints before iter {clear_iteration}\n"
                        )
                        sys.stderr.flush()
                except Exception as e:
                    sys.stderr.write(f"[DiffCkpt] ERROR: Failed to clear buffer: {e}\n")
                    sys.stderr.flush()
                continue

            # Normal checkpoint data
            diff, filename, i = data
            diff = _compress_diff(diff)

            if save_batch_freq == 1:
                begin = time.time()
                torch.save(diff, filename)
                end = time.time()
                now = datetime.datetime.now()
                sys.stderr.write(f"[DiffCkpt] Saved {os.path.basename(filename)} ({end - begin:.3f}s) at {now}\n")
                sys.stderr.flush()
            else:
                batch_buffer[i] = diff
                if i % save_batch_freq == save_batch_freq - 1:
                    begin = time.time()
                    torch.save(batch_buffer, filename)
                    end = time.time()
                    sys.stderr.write(f"[DiffCkpt] Saved {os.path.basename(filename)} ({end - begin:.3f}s)\n")
                    sys.stderr.flush()
                    batch_buffer = {}



    except KeyboardInterrupt:
        sys.stderr.write("[DiffCkpt] KeyboardInterrupt detected, performing emergency flush...\n")
        sys.stderr.flush()
        emergency_flush()
    except Exception as e:
        sys.stderr.write(f"[DiffCkpt] Unexpected error: {e}, performing emergency flush...\n")
        sys.stderr.flush()
        emergency_flush()
    finally:
        # Final flush on exit
        if batch_buffer:
            sys.stderr.write(f"[DiffCkpt] Final cleanup: flushing {len(batch_buffer)} remaining checkpoints...\n")
            sys.stderr.flush()
            flush_buffer()
        sys.stderr.write("[DiffCkpt] Background process terminated\n")
        sys.stderr.flush()

def _to_cpu(data):
    """
    Move tensor to CPU and return
    """
    if hasattr(data, 'cpu'):
        cpu_data = data.cpu().clone()
        return cpu_data
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    else:
        return data

def _compress_diff(data):
    """
    Compress diff_ckpt types to reduce memory usage:
    - float32 values  -> float16 (halves memory)
    - int64  indices  -> int32   (halves memory, safe for GPT-2 layer sizes)
    - moves tensors to CPU in the process
    Operates recursively on dict / list / tuple.
    Falls back to plain CPU copy if CUDA is unavailable (e.g. driver shutting down).
    """
    if isinstance(data, torch.Tensor):
        try:
            if data.dtype == torch.float32:
                return data.half().cpu()
            elif data.dtype == torch.int64:
                return data.to(torch.int32).cpu()
            else:
                return data.cpu()
        except Exception:
            # CUDA driver may be shutting down; return as-is and let torch.save handle it
            return data
    elif isinstance(data, dict):
        return {k: _compress_diff(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_compress_diff(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_compress_diff(v) for v in data)
    else:
        return data

