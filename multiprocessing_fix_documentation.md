# Technical Document: Solving PyTorch Multiprocessing Module Import Issues in ComfyUI Custom Nodes

## Problem Overview

When implementing multi-GPU processing using `torch.multiprocessing.spawn()` in ComfyUI custom nodes, we encountered a critical module import error that prevented spawned processes from starting:

```
ModuleNotFoundError: No module named '/workspace/MyDeptEDS/yangli/projects/ComfyUI/custom_nodes/ComfyUI-SUPIR-Parallel'
```

## Root Cause Analysis

### 1. **Invalid Module Names in ComfyUI**
ComfyUI loads custom nodes as file paths rather than proper Python modules, resulting in invalid `__module__` attributes:

```python
# Problematic module name assigned by ComfyUI:
function.__module__ = '/workspace/.../ComfyUI-SUPIR-Parallel.multi_gpu_vae_hook'
# Issues:
# - Contains absolute file paths (invalid in Python module names)
# - Contains hyphens (invalid in Python identifiers)
# - Not importable by child processes
```

### 2. **Pickle Serialization Process**
When `mp.spawn()` serializes functions, pickle stores a reference to the module rather than the function code:

```python
# Pickle stores:
{
  'type': 'function',
  'module': '/invalid/path/ComfyUI-SUPIR-Parallel.multi_gpu_vae_hook',
  'name': 'distributed_worker_function'
}

# Child process tries to recreate:
from '/invalid/path/ComfyUI-SUPIR-Parallel.multi_gpu_vae_hook' import distributed_worker_function
# FAILS: Invalid module path
```

### 3. **Missing Module Path in Child Processes**
Child processes start with clean Python environments and don't inherit the parent's module loading context from ComfyUI.

## Solution Implementation

### Step 1: Module Name Transformation
Convert invalid file paths to proper Python module names:

```python
# Input: '/workspace/.../custom_nodes/ComfyUI-SUPIR-Parallel.multi_gpu_vae_hook'
# Output: 'ComfyUI-SUPIR-Parallel.multi_gpu_vae_hook'

def fix_module_name(current_module_name):
    if 'ComfyUI-SUPIR-Parallel' in current_module_name:
        parts = current_module_name.split('ComfyUI-SUPIR-Parallel')
        module_suffix = parts[1].lstrip('.')
        return f"ComfyUI-SUPIR-Parallel.{module_suffix}"
```

### Step 2: Sys.Path Configuration
Add the parent directory to `sys.path` so child processes can find the module:

```python
# Extract parent directory: '/workspace/.../custom_nodes'
parent_dir = parts[0].rstrip('/')
if not parent_dir.endswith('custom_nodes'):
    parent_dir = parent_dir + '/custom_nodes'

# Add to sys.path for child process imports
sys.path.insert(0, parent_dir)
```

### Step 3: Function Re-import for Object Identity
Resolve pickle's object identity verification by importing the function from its proper module:

```python
# Import the function from the corrected module path
import importlib
module = importlib.import_module(proper_module_name)
imported_function = getattr(module, 'distributed_worker_function')

# Use imported function instead of original
mp.spawn(imported_function, args=(...))
```

## Complete Solution Code

```python
def fix_multiprocessing_function(function, current_module_name):
    """Fix function module name for multiprocessing compatibility"""
    
    if current_module_name.startswith('/'):
        if 'ComfyUI-SUPIR-Parallel' in current_module_name:
            # Transform path to module name
            parts = current_module_name.split('ComfyUI-SUPIR-Parallel')
            module_suffix = parts[1].lstrip('.')
            proper_module_name = f"ComfyUI-SUPIR-Parallel.{module_suffix}"
            
            # Fix sys.path for child processes
            parent_dir = parts[0].rstrip('/')
            if not parent_dir.endswith('custom_nodes'):
                parent_dir = parent_dir + '/custom_nodes'
            
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Re-import function for proper object identity
            try:
                import importlib
                module = importlib.import_module(proper_module_name)
                return getattr(module, function.__name__)
            except Exception:
                # Fallback to original function
                function.__module__ = proper_module_name
                return function
    
    return function
```

## Key Technical Insights

### 1. **Module Name vs File Path**
- **File paths** (with `/` and `-`) cannot be Python module names
- **Module names** must follow Python identifier rules (`package.module`)

### 2. **Pickle Object Identity**
- Pickle verifies that the function being serialized exists in its claimed module
- The imported function must be the **same object** as the one being pickled

### 3. **Child Process Environment**
- Child processes have clean Python environments
- They need explicit `sys.path` configuration to find custom modules
- ComfyUI's module loading doesn't automatically propagate to child processes

## Verification Process

The solution includes comprehensive debugging to verify each step:

```python
# 1. Test argument serialization
pickle.dumps(each_argument)

# 2. Verify module name transformation
print(f"Original: {original_module}")
print(f"Fixed: {fixed_module}")

# 3. Confirm function import success
print(f"Successfully imported function from module")

# 4. Validate pickle compatibility
pickle.dumps(imported_function)

# 5. Test child process startup
# Worker processes print startup messages
```

## Results

**Before Fix:**
```
ModuleNotFoundError: No module named '/workspace/.../ComfyUI-SUPIR-Parallel'
```

**After Fix:**
```
Successfully imported function from module
========1.1.10: function serializable
--------Worker 0 (actual_rank 0) starting-----------
--------Worker 1 (actual_rank 1) starting-----------
--------Worker 0: Completed successfully-----------
--------Worker 1: Completed successfully-----------
========1.2: mp.spawn completed
```

## Applicability

This solution applies to:
- **ComfyUI custom nodes** using multiprocessing
- **PyTorch distributed processing** in custom node environments  
- **Any Python multiprocessing** with dynamically loaded modules
- **Custom nodes with hyphens** in directory names

The approach transforms invalid module paths into valid Python module names while ensuring proper import paths for child processes.