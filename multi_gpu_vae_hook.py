import torch
import threading
import queue
import math
from typing import List, Tuple, Dict
import comfy.model_management as mm
from .SUPIR.utils.tilevae import VAEHook, GroupNormParam, clone_task_queue, build_task_queue, crop_valid_region, get_var_mean, custom_group_norm
import gc
from tqdm import tqdm
import comfy.utils
from einops import rearrange


class SDXLGroupNormSync:
    """
    Lightweight synchronization for SDXL VAE Group Normalization
    Handles exactly 17 sync points for encoder + 17 for decoder = 34 total
    OPTIMIZED: Each GPU processes tile batches, sync across GPU threads only
    """
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.sync_counter = 0
        
        # Shared memory for statistics collection (sync GPU threads)
        self.sync_barrier = threading.Barrier(num_gpus)
        self.stats_lock = threading.Lock()
        self.batch_stats = []  # Stats from each GPU's tile batch
        self.current_sync_id = 0
        
    def reset_for_sync_point(self):
        """Reset for a new synchronization point"""
        with self.stats_lock:
            self.batch_stats.clear()
            self.current_sync_id += 1
    
    def collect_and_sync_group_norm_batch(self, tile_batch: torch.Tensor, norm_layer, gpu_id: int):
        """
        Collect group norm statistics from tile batch and synchronize across GPUs
        tile_batch shape: [batch_size, C, H, W] where batch_size = tiles per GPU
        Returns normalized tile batch using global statistics
        """
        # 1. Compute batch statistics (aggregate across all tiles in this batch)
        batch_size = tile_batch.shape[0]
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
        
        # 2. Aggregate statistics for this GPU's batch
        total_pixels = sum(pixel_counts)
        weights = [count / total_pixels for count in pixel_counts]
        
        # Weighted average across tiles in this GPU's batch
        batch_mean = sum(w * m for w, m in zip(weights, mean_list))
        batch_var = sum(w * v for w, v in zip(weights, var_list))
        
        # 3. Collect batch statistics from this GPU
        with self.stats_lock:
            self.batch_stats.append({
                'mean': batch_mean.cpu(),
                'var': batch_var.cpu(), 
                'pixel_count': total_pixels,
                'gpu_id': gpu_id
            })
        
        # 4. Wait for ALL GPUs to reach this point
        self.sync_barrier.wait()
        
        # 5. Compute global statistics (only one thread does this)
        device = tile_batch.device
        with self.stats_lock:
            if len(self.batch_stats) == self.num_gpus:
                # Weighted average across all GPU batches
                total_pixels_global = sum(stat['pixel_count'] for stat in self.batch_stats)
                global_weights = [stat['pixel_count'] / total_pixels_global for stat in self.batch_stats]
                
                # Move all stats to current device before computation
                global_mean = torch.zeros_like(self.batch_stats[0]['mean']).to(device)
                global_var = torch.zeros_like(self.batch_stats[0]['var']).to(device)
                
                for w, stat in zip(global_weights, self.batch_stats):
                    global_mean += w * stat['mean'].to(device)
                    global_var += w * stat['var'].to(device)
                
                # Store for other threads
                self._global_mean = global_mean
                self._global_var = global_var
        
        # 6. All threads use the computed global statistics
        weight = norm_layer.weight.to(device) if hasattr(norm_layer, 'weight') and norm_layer.weight is not None else None
        bias = norm_layer.bias.to(device) if hasattr(norm_layer, 'bias') and norm_layer.bias is not None else None
        
        # Apply group norm to entire batch
        normalized_batch = []
        for i in range(batch_size):
            tile = tile_batch[i:i+1]  # [1, C, H, W]
            normalized_tile = custom_group_norm(
                tile, 32, self._global_mean, self._global_var, weight, bias
            )
            normalized_batch.append(normalized_tile)
        
        normalized_tile_batch = torch.cat(normalized_batch, dim=0)  # [batch_size, C, H, W]
        
        # 7. Reset for next sync point (first GPU to finish resets)
        try:
            self.sync_barrier.wait()
            # Thread-safe reset - only first GPU to reach here resets
            with self.stats_lock:
                if len(self.batch_stats) == self.num_gpus:
                    self.reset_for_sync_point()
        except threading.BrokenBarrierError:
            pass
        
        return normalized_tile_batch


class MultiGPUVAEHook(VAEHook):
    """
    Multi-GPU version of VAEHook optimized for SDXL VAE
    Handles exactly 34 group norm synchronization points
    """
    
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, num_gpus=None, to_gpu=True):
        super().__init__(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu)
        
        # Auto-detect available GPUs if not specified
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
            
        self.device_ids = list(range(self.num_gpus))
        
        # Create synchronizer for GPU threads
        self.group_norm_sync = SDXLGroupNormSync(self.num_gpus)
        
        print(f"[Multi-GPU VAE]: Using {self.num_gpus} GPUs for SDXL VAE: {self.device_ids}")
        print(f"[Multi-GPU VAE]: Expected sync points: {17 if is_decoder else 17}")
    
    def __call__(self, x):
        B, C, H, W = x.shape
        original_device = next(self.net.parameters()).device
        
        try:
            if max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Multi-GPU VAE]: Input size is small, using single GPU")
                return self.net.original_forward(x)
            elif self.num_gpus <= 1:
                print("[Multi-GPU VAE]: Only 1 GPU available, falling back to single GPU")
                return super().vae_tile_forward(x)
            else:
                return self.multi_gpu_vae_tile_forward(x)
        finally:
            self.net.to(original_device)
    
    def gpu_worker(self, gpu_id, tile_batches, result_queue, task_queue_template, gpu_network):
        """Worker function that processes tile batches on a specific GPU"""
        device = f'cuda:{gpu_id}'
        
        try:
            net_replica = self.net
            
            # Get tiles assigned to this GPU
            gpu_tiles = tile_batches[gpu_id]['tiles']
            gpu_in_bboxes = tile_batches[gpu_id]['in_bboxes']
            gpu_out_bboxes = tile_batches[gpu_id]['out_bboxes']
            gpu_tile_indices = tile_batches[gpu_id]['tile_indices']
            
            if len(gpu_tiles) == 0:
                return  # No tiles for this GPU
            
            # Concatenate tiles into batch: [batch_size, C, H, W]
            # Each tile: [1, C, H, W] -> concat -> [num_tiles, C, H, W]
            tile_batch = torch.cat([tile.to(device) for tile in gpu_tiles], dim=0)
            
            # Use the pre-loaded network for this GPU
            net_replica = gpu_network
            
            # Process the tile batch with multi-GPU sync (no gradients needed)
            with torch.no_grad():
                processed_batch = self.execute_task_queue_on_batch_threaded(
                    tile_batch, task_queue_template, net_replica, gpu_id)
            
            # Process results for each tile in the batch
            for i, (processed_tile, in_bbox, out_bbox, tile_idx) in enumerate(
                zip(processed_batch, gpu_in_bboxes, gpu_out_bboxes, gpu_tile_indices)):
                
                # Crop to valid region
                cropped_tile = crop_valid_region(
                    processed_tile.unsqueeze(0), in_bbox, out_bbox, self.is_decoder).squeeze(0)
                
                # Return result
                result_queue.put({
                    'tile_idx': tile_idx,
                    'tile': cropped_tile.cpu(),
                    'out_bbox': out_bbox,
                    'success': True
                })
            
            del tile_batch, processed_batch
            torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"GPU worker {gpu_id} failed: {e}")
            result_queue.put({
                'tile_idx': -1,
                'error': str(e),
                'success': False
            })
        
    def execute_task_queue_on_batch_threaded(self, tile_batch, task_queue_template, net, gpu_id):
        """Execute task queue on a tile batch with multi-GPU synchronization"""
        device = tile_batch.device
        task_queue = clone_task_queue(task_queue_template)
        
        while len(task_queue) > 0:
            task = task_queue.pop(0)
            
            if task[0] == 'pre_norm':
                # CRITICAL: Multi-GPU Group Norm Synchronization on batch
                tile_batch = self.group_norm_sync.collect_and_sync_group_norm_batch(
                    tile_batch, task[1], gpu_id)
                
            elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
                # Store residual connection (apply to batch)
                task_id = 0
                res = task[1](tile_batch)
                if not self.fast_mode or task[0] == 'store_res_cpu':
                    res = res.cpu()
                # Find corresponding add_res task
                while task_id < len(task_queue) and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id < len(task_queue):
                    task_queue[task_id][1] = res
                    
            elif task[0] == 'add_res':
                # Add residual connection (apply to batch)
                if task[1] is not None:
                    tile_batch += task[1].to(device)
                    task[1] = None
                    
            elif task[0] == 'apply_norm':
                # Pre-computed norm from fast mode (apply to batch)
                tile_batch = task[1](tile_batch)
                
            else:
                # Regular operations: conv, silu, etc. (apply to batch)
                tile_batch = task[1](tile_batch)
        
        return tile_batch
    
    def multi_gpu_vae_tile_forward(self, z):
        """Multi-GPU tiled VAE forward pass with optimized batching"""
        dtype = z.dtype
        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        
        print(f'[Multi-GPU VAE]: input_size: {z.shape}, tile_size: {self.tile_size}, padding: {self.pad}')
        
        # Split into tiles
        in_bboxes, out_bboxes = self.split_tiles(height, width)
        num_tiles = len(in_bboxes)
        
        # OPTIMIZATION: Adjust tile_size if too many tiles
        if num_tiles > self.num_gpus * 2:  # Allow max 2 batches
            # Increase tile size to reduce tile count
            scale_factor = math.ceil(math.sqrt(num_tiles / self.num_gpus))
            new_tile_size = min(self.tile_size * scale_factor, max(height, width))
            print(f"[Multi-GPU VAE]: Too many tiles ({num_tiles}), increasing tile_size from {self.tile_size} to {new_tile_size}")
            
            # Temporarily increase tile size
            original_tile_size = self.tile_size
            self.tile_size = new_tile_size
            in_bboxes, out_bboxes = self.split_tiles(height, width)
            num_tiles = len(in_bboxes)
            self.tile_size = original_tile_size  # Restore original
        
        print(f"[Multi-GPU VAE]: Processing {num_tiles} tiles across {self.num_gpus} GPUs")
        
        # Prepare tiles
        tiles = []
        for input_bbox in in_bboxes:
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]].cpu()
            tiles.append(tile)
        
        # Build task queues
        single_task_queue = build_task_queue(self.net, self.is_decoder)
        
        # Fast mode estimation (if enabled)
        if self.fast_mode:
            scale_factor = self.tile_size / max(height, width)
            z_device = z.to(self.device_ids[0])  # Use first GPU for estimation
            downsampled_z = torch.nn.functional.interpolate(z_device, scale_factor=scale_factor, mode='nearest-exact')
            
            # Recover statistics
            std_old, mean_old = torch.std_mean(z_device, dim=[0, 2, 3], keepdim=True)
            std_new, mean_new = torch.std_mean(downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean_new) / std_new * std_old + mean_old
            downsampled_z = torch.clamp_(downsampled_z, min=z_device.min(), max=z_device.max())
            
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self.estimate_group_norm(downsampled_z, estimate_task_queue, color_fix=self.color_fix):
                single_task_queue = estimate_task_queue
            
            del downsampled_z, z_device
        
        del z  # Free input memory
        
        # Initialize result tensor
        result = torch.zeros(
            (N, tiles[0].shape[1], 
             height * 8 if self.is_decoder else height // 8, 
             width * 8 if self.is_decoder else width // 8), 
            device=f'cuda:{self.device_ids[0]}', 
            requires_grad=False,
            dtype=dtype
        )
        
        # BATCH ALLOCATION: Distribute tiles across GPUs
        tile_batches = self.allocate_tiles_to_gpus(tiles, in_bboxes, out_bboxes, num_tiles)
        
        # PRE-LOAD MODELS: Create clean network replicas without threading locks
        print("[Multi-GPU VAE]: Pre-loading models on all GPUs...")
        gpu_networks = {}
        
        # Save original forward method and temporarily restore it for copying
        if hasattr(self.net, 'original_forward'):
            temp_forward = self.net.forward
            self.net.forward = self.net.original_forward
        
        for gpu_id in self.device_ids:
            device = f'cuda:{gpu_id}'
            # Now we can safely deepcopy the network without threading locks
            import copy
            gpu_networks[gpu_id] = copy.deepcopy(self.net).to(device).eval()
            print(f"[Multi-GPU VAE]: Model loaded on GPU {gpu_id}")
        
        # Restore our hooked forward method
        if hasattr(self.net, 'original_forward'):
            self.net.forward = temp_forward
        
        # Create result queue
        result_queue = queue.Queue()
                
        # Start worker threads - one per GPU
        workers = []
        for gpu_id in self.device_ids:
            worker = threading.Thread(
                target=self.gpu_worker,
                args=(gpu_id, tile_batches, result_queue, single_task_queue, gpu_networks[gpu_id])
            )
            worker.start()
            workers.append(worker)
        
        # Collect results
        completed_tiles = 0
        pbar = tqdm(total=num_tiles, desc="Processing tiles")
        
        while completed_tiles < num_tiles:
            try:
                tile_result = result_queue.get(timeout=10.0)
                
                if tile_result['success']:
                    out_bbox = tile_result['out_bbox']
                    processed_tile = tile_result['tile'].to(result.device)
                    result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = processed_tile
                    del processed_tile
                    completed_tiles += 1
                    pbar.update(1)
                else:
                    print(f"Error processing tile: {tile_result.get('error', 'Unknown error')}")
                    
            except queue.Empty:
                print("Timeout waiting for tile results")
                break
        
        pbar.close()
        
        # Wait for workers to finish
        for worker in workers:
            worker.join()
        
        return result.to(dtype)
    
    def allocate_tiles_to_gpus(self, tiles, in_bboxes, out_bboxes, num_tiles):
        """Allocate tiles to GPUs in round-robin fashion for load balancing"""
        tile_batches = {}
        
        # Initialize empty batches for each GPU
        for gpu_id in self.device_ids:
            tile_batches[gpu_id] = {
                'tiles': [],
                'in_bboxes': [],
                'out_bboxes': [],
                'tile_indices': []
            }
        
        # Distribute tiles round-robin across GPUs
        for i, (tile, in_bbox, out_bbox) in enumerate(zip(tiles, in_bboxes, out_bboxes)):
            gpu_id = i % self.num_gpus
            tile_batches[gpu_id]['tiles'].append(tile)
            tile_batches[gpu_id]['in_bboxes'].append(in_bbox)
            tile_batches[gpu_id]['out_bboxes'].append(out_bbox)
            tile_batches[gpu_id]['tile_indices'].append(i)
        
        # Print allocation summary
        for gpu_id in self.device_ids:
            batch_size = len(tile_batches[gpu_id]['tiles'])
            print(f"[Multi-GPU VAE]: GPU {gpu_id} allocated {batch_size} tiles")
        
        return tile_batches


def create_multi_gpu_vae_hook(net, tile_size, is_decoder, fast_decoder=False, fast_encoder=False, color_fix=False, num_gpus=None):
    """Factory function to create MultiGPUVAEHook"""
    return MultiGPUVAEHook(
        net=net,
        tile_size=tile_size, 
        is_decoder=is_decoder,
        fast_decoder=fast_decoder,
        fast_encoder=fast_encoder,
        color_fix=color_fix,
        num_gpus=num_gpus,
        to_gpu=True
    )