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
        
        # Recreate network on this GPU instead of using passed network object
        print(f"--------Worker {rank}: Recreating network from state dict-----------")
        # For now, we'll use a simplified approach - just create a dummy network
        # In practice, you'd recreate the actual network based on network_info
        net_replica = None  # We'll handle this differently
        
        # Skip task queue for now since we don't have network
        # task_queue_template = build_task_queue(net_replica, is_decoder)
        print(f"--------Worker {rank}: Skipping task queue build for test-----------")
        
        # Distribute tiles to this process (using actual_rank for distribution)
        tiles_for_rank = []
        bboxes_for_rank = []
        
        for i, (in_bbox, out_bbox) in enumerate(zip(in_bboxes, out_bboxes)):
            if i % world_size == actual_rank:  # Round-robin distribution
                tile = z_local[:, :, in_bbox[2]:in_bbox[3], in_bbox[0]:in_bbox[1]]
                tiles_for_rank.append(tile)
                bboxes_for_rank.append((in_bbox, out_bbox))
        
        if len(tiles_for_rank) == 0:
            print(f'--------Worker {rank}: No tiles assigned-----------')
            return
        
        # Create distributed sync for group norm
        group_norm_sync = DistributedGroupNormSync(actual_rank, world_size)
        
        # Initialize result tensor on this GPU
        result_shape = (z_shape[0], z_shape[1], 
                       z_shape[2] * 8 if is_decoder else z_shape[2] // 8, 
                       z_shape[3] * 8 if is_decoder else z_shape[3] // 8)
        local_result = torch.zeros(result_shape, dtype=z_dtype, device=device)
        
        # Skip actual processing for now - just test if workers start
        print(f"--------Worker {rank}: Skipping tile processing for test-----------")
        # for tile, (in_bbox, out_bbox) in zip(tiles_for_rank, bboxes_for_rank):
        #     processed_tile = execute_distributed_task_queue_static(
        #         tile.unsqueeze(0), task_queue_template, net_replica, group_norm_sync, fast_mode
        #     )
        #     
        #     processed_tile = crop_valid_region(
        #         processed_tile, None, out_bbox, is_decoder
        #     ).squeeze(0)
        #     
        #     # Write to local result tensor
        #     local_result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = processed_tile
        
        # Use NCCL reduce to send result to rank 0 (main process)
        dist.reduce(local_result, dst=0, op=dist.ReduceOp.SUM)
                
        torch.cuda.empty_cache()
        print(f'--------Worker {rank}: Completed successfully-----------')
        
        # Rank 0 returns the final result
        if actual_rank == 0:
            return local_result
        
    except Exception as e:
        print(f"--------Worker {rank} ERROR: {e}-----------")
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            dist.destroy_process_group()
        except:
            pass


def execute_distributed_task_queue_static(tile_batch, task_queue_template, net, group_norm_sync, fast_mode):
    """Static version of execute_distributed_task_queue for global function"""
    device = tile_batch.device
    # Clone task queue and replace network references with local replica
    task_queue = build_local_task_queue_static(task_queue_template, net)
    
    while len(task_queue) > 0:
        task = task_queue.pop(0)
        
        if task[0] == 'pre_norm':
            # Distributed Group Norm Synchronization
            tile_batch = group_norm_sync.collect_and_sync_group_norm_batch(
                tile_batch, task[1]
            )
            
        elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
            task_id = 0
            res = task[1](tile_batch)
            if not fast_mode or task[0] == 'store_res_cpu':
                res = res.cpu()
            while task_id < len(task_queue) and task_queue[task_id][0] != 'add_res':
                task_id += 1
            if task_id < len(task_queue):
                task_queue[task_id][1] = res
                
        elif task[0] == 'add_res':
            if task[1] is not None:
                tile_batch += task[1].to(device)
                task[1] = None
                
        elif task[0] == 'apply_norm':
            tile_batch = task[1](tile_batch)
            
        else:
            tile_batch = task[1](tile_batch)
    
    return tile_batch


def build_local_task_queue_static(task_queue_template, local_net):
    """Static version of build_local_task_queue for global function"""
    # For now, just return the template - we'll need to fix module mapping
    return task_queue_template


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
        dtype = z.dtype
        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        
        print(f'[Distributed VAE]: input_size: {z.shape}, tile_size: {self.tile_size}')
        
        # Split into tiles
        in_bboxes, out_bboxes = self.split_tiles(height, width)
        num_tiles = len(in_bboxes)
        
        print(f"[Distributed VAE]: Processing {num_tiles} tiles across {self.num_gpus} GPUs")
        
        # Build task queue
        single_task_queue = build_task_queue(self.net, self.is_decoder)
        
        # Fast mode estimation (if enabled)
        if self.fast_mode:
            scale_factor = self.tile_size / max(height, width)
            z_device = z.to(f'cuda:{self.device_ids[0]}')
            downsampled_z = torch.nn.functional.interpolate(z_device, scale_factor=scale_factor, mode='nearest-exact')
            
            std_old, mean_old = torch.std_mean(z_device, dim=[0, 2, 3], keepdim=True)
            std_new, mean_new = torch.std_mean(downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean_new) / std_new * std_old + mean_old
            downsampled_z = torch.clamp_(downsampled_z, min=z_device.min(), max=z_device.max())
            
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self.estimate_group_norm(downsampled_z, estimate_task_queue, color_fix=self.color_fix):
                single_task_queue = estimate_task_queue
            
            del downsampled_z, z_device
        
        # Prepare data for multiprocessing - only small serializable data
        input_data = {
            'z_shape': z.shape,
            'z_dtype': z.dtype,
            'in_bboxes': in_bboxes,
            'out_bboxes': out_bboxes,
            'is_decoder': self.is_decoder
            # No task_queue or z_cpu - build locally in each process
        }
        
        # Setup distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Create result tensor shape
        result_shape = (z.shape[0], z.shape[1], 
                       height * 8 if self.is_decoder else height // 8, 
                       width * 8 if self.is_decoder else width // 8)
        
        if self.num_gpus == 1:
            # Single GPU case - no distributed processing needed
            return super().vae_tile_forward(z)
        
        # Alternative approach: Use all processes as workers, no separate main process logic
        print("========1")
        if self.num_gpus > 1:
            try:
                # Setup distributed environment for ALL processes including main
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                
                # Try different start method to avoid conflicts
                mp.set_start_method('spawn', force=True)
                print("========1.1: Starting mp.spawn")
                
                # Test serialization of each argument
                try:
                    import pickle
                    print("========1.1.1: Testing argument serialization")
                    pickle.dumps(self.num_gpus)
                    print("========1.1.2: num_gpus serializable")
                    pickle.dumps(input_data)
                    print("========1.1.3: input_data serializable")
                    pickle.dumps(self.is_decoder)
                    print("========1.1.4: is_decoder serializable")
                    pickle.dumps(self.fast_mode)
                    print("========1.1.5: fast_mode serializable")
                    pickle.dumps(z)
                    print("========1.1.6: z serializable")
                    # Don't test self.net yet - it's likely the problem
                    print("========1.1.7: About to test network serialization")
                    pickle.dumps(self.net)
                    print("========1.1.8: network serializable")
                    
                    # Test function serialization
                    print("========1.1.9: Testing function serialization")
                    print(f"Function module: {distributed_worker_function.__module__}")
                    print(f"Function name: {distributed_worker_function.__name__}")
                    
                    # Fix the module name to be properly importable
                    # Use the current module name without the full path
                    import __main__
                    current_module_name = __name__
                    print(f"Current module __name__: {current_module_name}")
                    
                    # Try to use the proper module name that child processes can import
                    if current_module_name and not current_module_name.startswith('/'):
                        # If we have a proper module name, use it
                        distributed_worker_function.__module__ = current_module_name
                        print(f"Using current module name: {current_module_name}")
                    else:
                        # Fallback to a simple name
                        distributed_worker_function.__module__ = 'multi_gpu_vae_hook'
                        print(f"Using fallback module name: multi_gpu_vae_hook")
                    
                    # Test if this can be pickled
                    pickle.dumps(distributed_worker_function)
                    print("========1.1.10: function serializable")
                    
                    # Add current directory to environment for child processes
                    import sys
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    print(f"Added to sys.path: {current_dir}")
                except Exception as e:
                    print(f"========1.1.ERROR: Serialization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                
                # Spawn ALL workers (including what would be rank 0)
                # Remove self.net to test if that fixes the module import issue
                results = mp.spawn(
                    distributed_worker_function,  # Use global function instead of method
                    args=(None, self.num_gpus, input_data, self.is_decoder, self.fast_mode, z),
                    nprocs=self.num_gpus,  # ALL GPUs as workers
                    join=True  # Wait for completion and get results
                )
                print("========1.2: mp.spawn completed")
                
                # Extract result from rank 0 (first process)
                if results and len(results) > 0:
                    result = results[0]  # Result from rank 0
                    if result is not None:
                        print("========1.3: Got result from rank 0")
                        return result.to(z.device)
                
                print("========1.3: No result from workers, using fallback")
                
            except Exception as e:
                print(f"========1.ERROR: mp.spawn failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to single GPU
                return super().vae_tile_forward(z)
        else:
            return super().vae_tile_forward(z)

    def distributed_worker_nccl(self, rank, world_size, input_data):
        """Distributed worker for ranks 1, 2, 3... (rank 0 = main process)"""
        # Adjust rank since main process is rank 0, spawned processes are rank 1,2,3...
        actual_rank = rank + 1
        
        # Initialize distributed environment
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=actual_rank,
            world_size=world_size
        )
        
        device = f'cuda:{actual_rank}'
        torch.cuda.set_device(actual_rank)
        print(f'--------{rank}-----------')
        try:
            # Extract serializable data (no CUDA tensors here)
            z_shape = input_data['z_shape']
            z_dtype = input_data['z_dtype']
            in_bboxes = input_data['in_bboxes']
            out_bboxes = input_data['out_bboxes']
            is_decoder = input_data['is_decoder']
            
            # Create local tensor to receive broadcast from rank 0
            z_local = torch.zeros(z_shape, dtype=z_dtype, device=device)
            
            # Broadcast input tensor from rank 0 to all ranks
            dist.broadcast(z_local, src=0)
            
            # Load model replica on this GPU
            import copy
            net_replica = copy.deepcopy(self.net).to(device).eval()
            
            # Build task queue locally in this process
            task_queue_template = build_task_queue(net_replica, is_decoder)
            
            # Distribute tiles to this process (using actual_rank for distribution)
            tiles_for_rank = []
            bboxes_for_rank = []
            
            for i, (in_bbox, out_bbox) in enumerate(zip(in_bboxes, out_bboxes)):
                if i % world_size == actual_rank:  # Round-robin distribution
                    tile = z_local[:, :, in_bbox[2]:in_bbox[3], in_bbox[0]:in_bbox[1]]
                    tiles_for_rank.append(tile)
                    bboxes_for_rank.append((in_bbox, out_bbox))
            
            if len(tiles_for_rank) == 0:
                return
            
            # Create distributed sync for group norm
            group_norm_sync = DistributedGroupNormSync(actual_rank, world_size)
            
            # Initialize result tensor on this GPU
            result_shape = (z_shape[0], z_shape[1], 
                           z_shape[2] * 8 if self.is_decoder else z_shape[2] // 8, 
                           z_shape[3] * 8 if self.is_decoder else z_shape[3] // 8)
            local_result = torch.zeros(result_shape, dtype=z_dtype, device=device)
            
            # Process assigned tiles and write to local result
            for tile, (in_bbox, out_bbox) in zip(tiles_for_rank, bboxes_for_rank):
                processed_tile = self.execute_distributed_task_queue(
                    tile.unsqueeze(0), task_queue_template, net_replica, group_norm_sync
                )
                
                processed_tile = crop_valid_region(
                    processed_tile, None, out_bbox, self.is_decoder
                ).squeeze(0)
                
                # Write to local result tensor
                local_result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = processed_tile
            
            # Use NCCL reduce to send result to rank 0 (main process)
            dist.reduce(local_result, dst=0, op=dist.ReduceOp.SUM)
                    
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Process {rank}] Error: {e}")
            raise
        finally:
            dist.destroy_process_group()
    
    def execute_rank_0_processing(self, z, input_data, result_shape):
        """Main process executes as rank 0 in distributed group"""
        print("========2.1: Starting rank 0 processing")
        
        # Initialize distributed environment as rank 0
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=0,
                world_size=self.num_gpus
            )
            print("========2.2: Distributed group initialized")
        except Exception as e:
            print(f"========2.2 ERROR: Failed to init distributed group: {e}")
            raise
        
        torch.cuda.set_device(0)
        print("========2.3: CUDA device set")
        
        try:
            # Extract data
            z_shape = input_data['z_shape']
            z_dtype = input_data['z_dtype']
            in_bboxes = input_data['in_bboxes']
            out_bboxes = input_data['out_bboxes']
            is_decoder = input_data['is_decoder']
            print("========2.4: Data extracted")
            
            # Load model replica on GPU 0 FIRST
            import copy
            net_replica = copy.deepcopy(self.net).to('cuda:0').eval()
            print("========2.5: Model replica created")
            
            # Move input to GPU 0 and broadcast to all ranks
            z_gpu0 = z.to('cuda:0')  # Use original z instead of z_cpu
            dist.broadcast(z_gpu0, src=0)
            print("========2.6: Tensor broadcast completed")
            
            # Build task queue locally in rank 0 AFTER net_replica is defined
            task_queue_template = build_task_queue(net_replica, is_decoder)
            print("========2.7: Task queue built")
            
            # Distribute tiles to rank 0
            tiles_for_rank0 = []
            bboxes_for_rank0 = []
            
            for i, (in_bbox, out_bbox) in enumerate(zip(in_bboxes, out_bboxes)):
                if i % self.num_gpus == 0:  # Rank 0 gets tiles 0, num_gpus, 2*num_gpus, ...
                    tile = z_gpu0[:, :, in_bbox[2]:in_bbox[3], in_bbox[0]:in_bbox[1]]
                    tiles_for_rank0.append(tile)
                    bboxes_for_rank0.append((in_bbox, out_bbox))
            
            # Create distributed sync for group norm
            group_norm_sync = DistributedGroupNormSync(0, self.num_gpus)
            
            # Initialize result tensor on GPU 0
            result = torch.zeros(result_shape, dtype=z_dtype, device='cuda:0')
            
            # Process rank 0's assigned tiles
            for tile, (in_bbox, out_bbox) in zip(tiles_for_rank0, bboxes_for_rank0):
                processed_tile = self.execute_distributed_task_queue(
                    tile.unsqueeze(0), task_queue_template, net_replica, group_norm_sync
                )
                
                processed_tile = crop_valid_region(
                    processed_tile, None, out_bbox, self.is_decoder
                ).squeeze(0)
                
                # Write to result tensor
                result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = processed_tile
            
            # Receive results from other ranks via NCCL reduce
            # The reduce operation will sum all local_result tensors to rank 0
            # Since each rank writes to different regions, SUM effectively combines them
            dist.reduce(result, dst=0, op=dist.ReduceOp.SUM)
            
            torch.cuda.empty_cache()
            return result
            
        finally:
            dist.destroy_process_group()

    def execute_distributed_task_queue(self, tile_batch, task_queue_template, net, group_norm_sync):
        """Execute task queue with distributed group norm synchronization"""
        device = tile_batch.device
        # Clone task queue and replace network references with local replica
        task_queue = self.build_local_task_queue(task_queue_template, net)
        
        while len(task_queue) > 0:
            task = task_queue.pop(0)
            
            if task[0] == 'pre_norm':
                # Distributed Group Norm Synchronization
                tile_batch = group_norm_sync.collect_and_sync_group_norm_batch(
                    tile_batch, task[1]
                )
                
            elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
                task_id = 0
                res = task[1](tile_batch)
                if not self.fast_mode or task[0] == 'store_res_cpu':
                    res = res.cpu()
                while task_id < len(task_queue) and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id < len(task_queue):
                    task_queue[task_id][1] = res
                    
            elif task[0] == 'add_res':
                if task[1] is not None:
                    tile_batch += task[1].to(device)
                    task[1] = None
                    
            elif task[0] == 'apply_norm':
                tile_batch = task[1](tile_batch)
                
            else:
                tile_batch = task[1](tile_batch)
        
        return tile_batch
    
    def build_local_task_queue(self, task_queue_template, local_net):
        """Build task queue with local network references instead of original ones"""
        # Create mapping from original network modules to local network modules
        original_net = self.net
        module_mapping = self.create_module_mapping(original_net, local_net)
        
        # Clone task queue and replace module references
        local_task_queue = []
        for task in task_queue_template:
            if len(task) >= 2 and hasattr(task[1], '__call__'):
                # This is a network operation task
                original_module = task[1]
                if original_module in module_mapping:
                    # Replace with local network module
                    local_task = [task[0], module_mapping[original_module]] + task[2:]
                else:
                    # Keep as is (might be a function or other callable)
                    local_task = list(task)
            else:
                # Non-network task, clone as-is
                local_task = list(task)
            local_task_queue.append(local_task)
        
        return local_task_queue
    
    def create_module_mapping(self, original_net, local_net):
        """Create mapping from original network modules to local network modules"""
        mapping = {}
        
        # Map the network itself
        mapping[original_net] = local_net
        
        # Recursively map all submodules
        for (orig_name, orig_module), (local_name, local_module) in zip(
            original_net.named_modules(), local_net.named_modules()
        ):
            if orig_name == local_name:  # Should match if networks are identical
                mapping[orig_module] = local_module
        
        return mapping
    
    def allocate_tiles_to_processes(self, tiles, in_bboxes, out_bboxes):
        """Allocate tiles to processes in round-robin fashion"""
        tile_batches = [[] for _ in range(self.num_gpus)]
        bbox_batches = [[] for _ in range(self.num_gpus)]
        
        for i, (tile, in_bbox, out_bbox) in enumerate(zip(tiles, in_bboxes, out_bboxes)):
            process_id = i % self.num_gpus
            tile_batches[process_id].append(tile)
            bbox_batches[process_id].append((in_bbox, out_bbox))
        
        return tile_batches, bbox_batches



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