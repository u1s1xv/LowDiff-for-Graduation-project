import os
import torch
from deepspeed import comm as dist
# import torch.distributed as dist
import concurrent.futures
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import datetime
import time

class Communicator:
    def __init__(self, model, k=0.01, num_threads=None, save_batch_freq=1, use_error_feedback=True,
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
            use_error_feedback (bool): Whether to use error feedback mechanism for gradient compensation.
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

        # Error Feedback Mechanism
        self.use_error_feedback = use_error_feedback
        self.error_feedback = {}  # Store error tensors for each parameter
        self.error_clipped_count = 0  # Track how many times error was clipped
        self.error_reset_count = 0  # Track how many times error was reset due to NaN/Inf

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
                keep_full_checkpoints=keep_full_checkpoints
            )
            print(f"[SmartCkpt] Smart checkpoint management ENABLED (max_interval={max_full_interval}, keep={keep_full_checkpoints})")

        # Get the number of available CPU threads (default to half of total cores, max 32)
        if num_threads is None:
            num_threads = int(os.cpu_count() / 2)

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)  # Thread pool
        self.param_dict = dict(self.model.named_parameters())

        print(f"Using {num_threads} threads for gradient decompression.")
        if self.use_error_feedback:
            print("Error Feedback mechanism is ENABLED.")
        else:
            print("Error Feedback mechanism is DISABLED.")

        if dist.get_rank() == 0:
            self.save_batch_freq = save_batch_freq
            self.diff_ckpt = {}
            self.queue = mp.Queue()
            self.save_process = mp.Process(target=diff_ckpt_saver, args=(self.queue,self.save_batch_freq))
            self.save_process.start()
            print("save process start!")
        
        
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
        Hook function for gradient compression with error feedback.
        """
        world_size = dist.get_world_size()

        # 保存原始梯度用于自适应裁剪
        original_grad = grad.clone() if self.use_error_feedback else None

        # Error Feedback: Compensate gradient with accumulated error
        if self.use_error_feedback:
            if param_name not in self.error_feedback:
                # Initialize error buffer for this parameter
                self.error_feedback[param_name] = torch.zeros_like(grad)

            # Compensate gradient: g'_t = g_t + e_{t-1}
            compensated_grad = grad + self.error_feedback[param_name]
        else:
            compensated_grad = grad

        # Compress the compensated gradient
        indices, values = self.topk_compress(compensated_grad)

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
            "grad_shape": grad.shape,
            "original_grad": original_grad,
            "compensated_grad": compensated_grad.clone() if self.use_error_feedback else None
        }

        if dist.get_rank() == 0:
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

    def should_save_full_checkpoint(self, iteration, epoch, loss):
        """
        决策是否保存全量检查点（智能检查点管理）

        Args:
            iteration: 当前batch索引
            epoch: 当前epoch
            loss: 当前loss值

        Returns:
            (should_save, reason): 是否保存及原因
        """
        if not self.enable_smart_checkpoint or self.smart_ckpt_manager is None:
            return False, 'smart_checkpoint_disabled'

        # 注意：grad_norm参数已被移除，不再基于梯度范数做决策
        return self.smart_ckpt_manager.should_save_full_checkpoint(iteration, epoch, loss)

    def cleanup_old_full_checkpoints(self):
        """清理旧的全量检查点"""
        if self.enable_smart_checkpoint and self.smart_ckpt_manager is not None:
            self.smart_ckpt_manager.cleanup_old_full_checkpoints()

    def cleanup_old_checkpoints(self, current_iteration, epoch):
        """清理旧的差分检查点"""
        if self.enable_smart_checkpoint and self.smart_ckpt_manager is not None:
            self.smart_ckpt_manager.cleanup_old_diff_checkpoints(current_iteration, epoch)

    def decompress_save(self, diff, filename, i):
        """
        Parallel gradient restoration with error feedback update.
        """
        def process_gradient(param_name, param, data):
            data["work_indices"].wait()
            data["work_values"].wait()

            restored_grad = torch.zeros(data["grad_shape"], device=data["gathered_values"][0].device).view(-1)

            for indices, values in zip(data["gathered_indices"], data["gathered_values"]):
                restored_grad.scatter_add_(0, indices, values)  # This remains a CPU/GPU task

            restored_grad = restored_grad.view(data["grad_shape"])

            # Error Feedback: Update error buffer with momentum
            # e_t = compensated_grad - restored_grad
            if self.use_error_feedback and data["compensated_grad"] is not None:
                error = data["compensated_grad"] - restored_grad

                # Critical: Check for NaN/Inf and clip error to prevent numerical instability
                if torch.isnan(error).any() or torch.isinf(error).any():
                    error = torch.zeros_like(error)
                    self.error_reset_count += 1
                else:
                    # 自适应误差裁剪
                    error_norm = error.norm()
                    grad_norm = data["original_grad"].norm() if data["original_grad"] is not None else restored_grad.norm()
                    max_error_norm = max(10.0, grad_norm.item() * 0.5)
                    if error_norm > max_error_norm:
                        error = error * (max_error_norm / error_norm)
                        self.error_clipped_count += 1

                # 误差动量：加速补偿收敛
                momentum = 0.9
                if param_name in self.error_feedback:
                    error = momentum * self.error_feedback[param_name] + error

                self.error_feedback[param_name] = error.detach()

            param.grad = restored_grad  # Direct assignment

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
            self.queue.put((self.diff_ckpt,filename,i))

    def get_error_norm(self):
        """
        Get the total norm of all error tensors (for monitoring).

        Returns:
            float: Total L2 norm of all error tensors.
        """
        if not self.use_error_feedback:
            return 0.0

        total_norm = 0.0
        for error in self.error_feedback.values():
            total_norm += error.norm().item() ** 2
        return total_norm ** 0.5

    def reset_error_feedback(self):
        """
        Reset all error buffers to zero (for debugging).
        """
        if not self.use_error_feedback:
            return

        for param_name in self.error_feedback:
            self.error_feedback[param_name].zero_()

        if dist.get_rank() == 0:
            print("Error feedback buffers have been reset to zero.")

    def get_error_stats(self):
        """
        Get detailed statistics of error tensors (for analysis).

        Returns:
            dict: Statistics including mean, std, max, min of errors.
        """
        if not self.use_error_feedback or len(self.error_feedback) == 0:
            return {}

        all_errors = torch.cat([e.flatten() for e in self.error_feedback.values()])

        stats = {
            'error_norm': self.get_error_norm(),
            'error_mean': all_errors.mean().item(),
            'error_std': all_errors.std().item(),
            'error_max': all_errors.max().item(),
            'error_min': all_errors.min().item(),
            'num_params': len(self.error_feedback)
        }

        return stats

    def __del__(self):
        """
        Ensure the thread pool is properly shut down on object destruction.
        """
        self.executor.shutdown(wait=True)
        self.queue.put(None)
        self.save_process.join()

def diff_ckpt_saver(queue,save_batch_freq):
    """
    Background process that saves compressed gradients to disk.
    
    Args:
        queue (mp.Queue): Queue receiving data to be saved.
        save_batch_freq (int): Save frequency in terms of batch steps.
    """
    
    batch_buffer = {}
    print("batching freq = {}".format(save_batch_freq))
    
    while True:
        data = queue.get()
        
        if data is None:
            break
        diff, filename, i = data
        data = _to_cpu(data)
    
        if save_batch_freq == 1 :
            begin = time.time()
            torch.save(diff, filename)
            end = time.time()
            now = datetime.datetime.now()
            print("Saved {} time: {:.3f}s at {}".format(filename, end - begin, now))
        
        else: 
            batch_buffer[i] = diff
            if i % save_batch_freq == save_batch_freq-1:
                begin = time.time()
                torch.save(batch_buffer, filename)
                end = time.time()
                print("Saved {} time: {:.3f}s".format(filename, end - begin))
                batch_buffer={}

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