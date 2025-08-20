import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import math
import os
from typing import List, Tuple, Dict
import comfy.model_management as mm
from .SUPIR.utils.tilevae import VAEHook, GroupNormParam, clone_task_queue, build_task_queue, crop_valid_region, get_var_mean, custom_group_norm
import gc
from tqdm import tqdm
import comfy.utils
from einops import rearrange


def distributed_worker_function(rank, network_info, world_size, input_data, is_decoder, fast_mode, z_original):
    """Global function for distributed worker - avoids self serialization issues"""
    # Fix module path for spawned processes
    import sys
    import os
    
    # Add ComfyUI paths to sys.path
    comfy_path = '/workspace/MyDeptEDS/yangli/projects/ComfyUI'
    custom_nodes_path = '/workspace/MyDeptEDS/yangli/projects/ComfyUI/custom_nodes'
    supir_path = '/workspace/MyDeptEDS/yangli/projects/ComfyUI/custom_nodes/ComfyUI-SUPIR-Parallel'
    
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    if custom_nodes_path not in sys.path:
        sys.path.insert(0, custom_nodes_path)
    if supir_path not in sys.path:
        sys.path.insert(0, supir_path)
    
    # Now we can import the required modules
    try:
        from SUPIR.utils.tilevae import build_task_queue, crop_valid_region, get_var_mean, custom_group_norm
        import torch
        import torch.distributed as dist
        import copy
        
        # Import the DistributedGroupNormSync class from current module
        # We need to redefine it here since it's not easily importable in spawned process
        class DistributedGroupNormSync:
            def __init__(self, rank: int, world_size: int):
                self.rank = rank
                self.world_size = world_size
                
            def collect_and_sync_group_norm_batch(self, tile_batch: torch.Tensor, norm_layer):
                device = tile_batch.device
                batch_size = tile_batch.shape[0]
                
                var_list = []
                mean_list = []
                pixel_counts = []
                
                for i in range(batch_size):
                    tile = tile_batch[i:i+1]
                    var, mean = get_var_mean(tile, 32)
                    pixel_count = tile.shape[2] * tile.shape[3]
                    
                    var_list.append(var)
                    mean_list.append(mean)
                    pixel_counts.append(pixel_count)
                
                total_pixels = sum(pixel_counts)
                weights = [count / total_pixels for count in pixel_counts]
                
                local_mean = sum(w * m for w, m in zip(weights, mean_list))
                local_var = sum(w * v for w, v in zip(weights, var_list))
                
                local_stats = torch.stack([
                    local_mean.flatten(),
                    local_var.flatten(),
                    torch.tensor(total_pixels, device=device, dtype=local_mean.dtype).expand_as(local_mean.flatten())
                ], dim=0)
                
                all_stats = [torch.zeros_like(local_stats) for _ in range(self.world_size)]
                dist.all_gather(all_stats, local_stats)
                
                total_pixels_global = sum(stats[2].sum().item() for stats in all_stats)
                
                global_mean = torch.zeros_like(local_mean)
                global_var = torch.zeros_like(local_var)
                
                for stats in all_stats:
                    weight = stats[2].sum().item() / total_pixels_global
                    global_mean += weight * stats[0].view_as(local_mean)
                    global_var += weight * stats[1].view_as(local_var)
                
                weight = norm_layer.weight.to(device) if hasattr(norm_layer, 'weight') and norm_layer.weight is not None else None
                bias = norm_layer.bias.to(device) if hasattr(norm_layer, 'bias') and norm_layer.bias is not None else None
                
                normalized_batch = []
                for i in range(batch_size):
                    tile = tile_batch[i:i+1]
                    normalized_tile = custom_group_norm(
                        tile, 32, global_mean, global_var, weight, bias
                    )
                    normalized_batch.append(normalized_tile)
                
                normalized_tile_batch = torch.cat(normalized_batch, dim=0)
                return normalized_tile_batch
        
    except ImportError as e:
        print(f"--------Worker {rank}: Import error: {e}-----------")
        return
    
    # Now rank is directly 0, 1, 2, 3... (no adjustment needed)
    actual_rank = rank
    
    print(f'--------Worker {rank} (actual_rank {actual_rank}) starting-----------')
    
    try:
        # Initialize distributed environment
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=actual_rank,
            world_size=world_size
        )
        
        device = f'cuda:{actual_rank}'
        torch.cuda.set_device(actual_rank)
        
        # Extract serializable data (no CUDA tensors here)
        z_shape = input_data['z_shape']
        z_dtype = input_data['z_dtype']
        in_bboxes = input_data['in_bboxes']
        out_bboxes = input_data['out_bboxes']
        
        # Rank 0 creates the tensor, others receive it
        if actual_rank == 0:
            # Rank 0: Create tensor from original input
            z_local = z_original.to(device)
        else:
            # Other ranks: Create empty tensor to receive broadcast
            z_local = torch.zeros(z_shape, dtype=z_dtype, device=device)
        
        # Broadcast input tensor from rank 0 to all ranks
        dist.broadcast(z_local, src=0)
        
        # TODO: Implement actual network recreation and tile processing
        # For now, create a dummy result tensor to test the multiprocessing framework
        
        # Initialize result tensor on this GPU
        result_shape = (z_shape[0], z_shape[1], 
                       z_shape[2] * 8 if is_decoder else z_shape[2] // 8, 
                       z_shape[3] * 8 if is_decoder else z_shape[3] // 8)
        local_result = torch.zeros(result_shape, dtype=z_dtype, device=device)
        
        # Use NCCL reduce to send result to rank 0 (main process)
        dist.reduce(local_result, dst=0, op=dist.ReduceOp.SUM)
                
        torch.cuda.empty_cache()
        
        # Rank 0 returns the final result
        if actual_rank == 0:
            return local_result
        
    except Exception:
        raise
    finally:
        try:
            dist.destroy_process_group()
        except:
            pass


class DistributedGroupNormSync:
    """
    Distributed synchronization for SDXL VAE Group Normalization
    Uses PyTorch distributed collective operations for efficient GPU communication
    """
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
    def collect_and_sync_group_norm_batch(self, tile_batch: torch.Tensor, norm_layer):
        """
        Collect group norm statistics from tile batch and synchronize across all processes
        tile_batch shape: [batch_size, C, H, W] where batch_size = tiles per GPU
        Returns normalized tile batch using global statistics
        """
        device = tile_batch.device
        batch_size = tile_batch.shape[0]
        
        # 1. Compute local batch statistics (aggregate across all tiles in this batch)
        var_list = []
        mean_list = []
        pixel_counts = []
        
        for i in range(batch_size):
            tile = tile_batch[i:i+1]  # Keep batch dim: [1, C, H, W]
            var, mean = get_var_mean(tile, 32)  # SDXL uses 32 groups
            pixel_count = tile.shape[2] * tile.shape[3]
            
            var_list.append(var)
            mean_list.append(mean)
            pixel_counts.append(pixel_count)
        
        # 2. Aggregate statistics for this process's batch
        total_pixels = sum(pixel_counts)
        weights = [count / total_pixels for count in pixel_counts]
        
        # Weighted average across tiles in this process's batch
        local_mean = sum(w * m for w, m in zip(weights, mean_list))
        local_var = sum(w * v for w, v in zip(weights, var_list))
        
        # 3. Create tensors for distributed communication
        # Pack mean, var, and pixel count for collective operations
        local_stats = torch.stack([
            local_mean.flatten(),
            local_var.flatten(),
            torch.tensor(total_pixels, device=device, dtype=local_mean.dtype).expand_as(local_mean.flatten())
        ], dim=0)  # [3, num_groups]
        
        # 4. Gather statistics from all processes
        all_stats = [torch.zeros_like(local_stats) for _ in range(self.world_size)]
        dist.all_gather(all_stats, local_stats)
        
        # 5. Compute global weighted statistics
        total_pixels_global = sum(stats[2].sum().item() for stats in all_stats)
        
        global_mean = torch.zeros_like(local_mean)
        global_var = torch.zeros_like(local_var)
        
        for stats in all_stats:
            weight = stats[2].sum().item() / total_pixels_global
            global_mean += weight * stats[0].view_as(local_mean)
            global_var += weight * stats[1].view_as(local_var)
        
        # 6. Apply group norm to entire batch
        weight = norm_layer.weight.to(device) if hasattr(norm_layer, 'weight') and norm_layer.weight is not None else None
        bias = norm_layer.bias.to(device) if hasattr(norm_layer, 'bias') and norm_layer.bias is not None else None
        
        normalized_batch = []
        for i in range(batch_size):
            tile = tile_batch[i:i+1]  # [1, C, H, W]
            normalized_tile = custom_group_norm(
                tile, 32, global_mean, global_var, weight, bias
            )
            normalized_batch.append(normalized_tile)
        
        normalized_tile_batch = torch.cat(normalized_batch, dim=0)  # [batch_size, C, H, W]
        return normalized_tile_batch


class DistributedVAEHook(VAEHook):
    """
    Distributed Multi-GPU VAE Hook using PyTorch multiprocessing
    Eliminates GIL bottleneck with process-based parallelization
    """
    
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, num_gpus=None, to_gpu=True):
        super().__init__(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu)
        
        # Auto-detect available GPUs if not specified
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
            
        self.device_ids = list(range(self.num_gpus))
        
        print(f"[Distributed VAE]: Using {self.num_gpus} GPUs for SDXL VAE: {self.device_ids}")
    
    def __call__(self, x):
        H, W = x.shape[2], x.shape[3]
        original_device = next(self.net.parameters()).device
        
        try:
            if max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Distributed VAE]: Input size is small, using single GPU")
                return self.net.original_forward(x)
            elif self.num_gpus <= 1:
                print("[Distributed VAE]: Only 1 GPU available, falling back to single GPU")
                return super().vae_tile_forward(x)
            else:
                return self.distributed_vae_forward(x)
        finally:
            self.net.to(original_device)
    
    def distributed_vae_forward(self, z):
        """Main distributed VAE forward pass using NCCL broadcast"""
        height, width = z.shape[2], z.shape[3]
        
        print(f'[Distributed VAE]: input_size: {z.shape}, tile_size: {self.tile_size}')
        
        # Split into tiles
        in_bboxes, out_bboxes = self.split_tiles(height, width)
        num_tiles = len(in_bboxes)
        
        print(f"[Distributed VAE]: Processing {num_tiles} tiles across {self.num_gpus} GPUs")
        
        # Prepare data for multiprocessing - only small serializable data
        input_data = {
            'z_shape': z.shape,
            'z_dtype': z.dtype,
            'in_bboxes': in_bboxes,
            'out_bboxes': out_bboxes,
            'is_decoder': self.is_decoder
        }

        try:
            # Setup distributed environment for ALL processes including main
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            # Fix multiprocessing module compatibility
            mp.set_start_method('spawn', force=True)
            
            # Convert file path to proper Python module name for serialization
            current_module_name = __name__
            if current_module_name.startswith('/') and 'ComfyUI-SUPIR-Parallel' in current_module_name:
                parts = current_module_name.split('ComfyUI-SUPIR-Parallel')
                if len(parts) > 1:
                    module_suffix = parts[1].lstrip('.')
                    proper_module_name = f"ComfyUI-SUPIR-Parallel.{module_suffix}"
                    distributed_worker_function.__module__ = proper_module_name
                    
                    # Add parent directory to sys.path for child processes
                    import sys
                    parent_dir = parts[0].rstrip('/')
                    if not parent_dir.endswith('custom_nodes'):
                        parent_dir = parent_dir + '/custom_nodes'
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    
                    # Import the function properly to fix object identity
                    try:
                        import importlib
                        module = importlib.import_module(proper_module_name)
                        test_function = getattr(module, 'distributed_worker_function')
                    except Exception:
                        test_function = distributed_worker_function
                else:
                    test_function = distributed_worker_function
            else:
                test_function = distributed_worker_function
            
            # Spawn workers across all GPUs
            results = mp.spawn(
                test_function,
                args=(None, self.num_gpus, input_data, self.is_decoder, self.fast_mode, z),
                nprocs=self.num_gpus,
                join=True
            )
            
            # Extract result from rank 0
            if results and len(results) > 0 and results[0] is not None:
                return results[0].to(z.device)
            
            # Fallback if no results
            return super().vae_tile_forward(z)
            
        except Exception:
            # Fallback to single GPU on any error
            return super().vae_tile_forward(z)



def create_distributed_vae_hook(net, tile_size, is_decoder, fast_decoder=False, fast_encoder=False, color_fix=False, num_gpus=None):
    """Factory function to create DistributedVAEHook"""
    return DistributedVAEHook(
        net=net,
        tile_size=tile_size, 
        is_decoder=is_decoder,
        fast_decoder=fast_decoder,
        fast_encoder=fast_encoder,
        color_fix=color_fix,
        num_gpus=num_gpus,
        to_gpu=True
    )

# Backward compatibility
create_multi_gpu_vae_hook = create_distributed_vae_hook