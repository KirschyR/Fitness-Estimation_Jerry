"""
Numba Switchable Decorator
==========================

Provides configurable Numba JIT compilation with three-level control:
1. Function-level (highest priority)
2. Module-level
3. Global (lowest priority)

Configuration is read from configs.numba_config.
"""

import os
from pathlib import Path
from typing import Callable, Optional
from functools import wraps

from configs import numba_config as config


# ============================================================================
# Cache Directory Setup
# ============================================================================

def _setup_cache_dir():
    """Configure Numba cache directory based on config."""
    cache_dir = config.CACHE_DIR
    
    if cache_dir is None:
        return
    
    cache_path = Path(cache_dir)
    
    # Create directory if not exists
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable (must be done before importing numba)
    os.environ['NUMBA_CACHE_DIR'] = str(cache_path.resolve())


# Initialize cache directory on module load
_setup_cache_dir()


# ============================================================================
# JIT Decision Logic
# ============================================================================

def _should_jit(func: Callable) -> bool:
    """
    Determine whether a function should be JIT compiled.
    
    Priority: Function > Module > Global
    """
    module_name = func.__module__
    full_name = f"{module_name}.{func.__qualname__}"
    
    # 1. Function-level override (highest priority)
    if full_name in config.FUNCTION_OVERRIDES:
        return config.FUNCTION_OVERRIDES[full_name]
    
    # 2. Module-level override
    if module_name in config.MODULE_OVERRIDES:
        return config.MODULE_OVERRIDES[module_name]
    
    # 3. Global setting (lowest priority)
    return config.ENABLED_GLOBAL


# ============================================================================
# Decorator
# ============================================================================

def numba_switchable(
    func: Optional[Callable] = None,
    *,
    cache: bool = True,
    parallel: bool = False,
    fastmath: bool = False,
    **njit_kwargs
) -> Callable:
    """
    Configurable Numba JIT decorator.
    
    Args:
        func: Function to decorate
        cache: Cache compiled functions (default: True)
        parallel: Enable automatic parallelization (default: False)
        fastmath: Enable fast math optimizations (default: False)
        **njit_kwargs: Additional arguments for numba.njit
    
    Usage:
        ```python
        @numba_switchable
        def my_func(x):
            ...
        
        @numba_switchable(parallel=True, fastmath=True)
        def my_parallel_func(x):
            ...
        ```
    
    Attributes on decorated function:
        - python: Original Python function
        - numba: JIT-compiled function (None if disabled)
        - is_jit_enabled: Whether JIT is enabled
        - full_name: Fully qualified function name
    """
    
    def decorator(fn: Callable) -> Callable:
        module_name = fn.__module__
        full_name = f"{module_name}.{fn.__qualname__}"
        
        # Determine if JIT should be applied
        should_jit = _should_jit(fn)
        numba_func = None
        
        if should_jit:
            try:
                from numba import njit
                numba_func = njit(
                    fn,
                    cache=cache,
                    parallel=parallel,
                    fastmath=fastmath,
                    **njit_kwargs
                )
            except ImportError:
                import warnings
                warnings.warn(
                    f"Numba not installed. '{full_name}' uses pure Python.",
                    RuntimeWarning
                )
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Numba compilation failed for '{full_name}': {e}",
                    RuntimeWarning
                )
        
        # Choose which version to use
        if numba_func is not None:
            wrapper = numba_func
        else:
            wrapper = fn
        
        # Attach metadata
        wrapper.python = fn
        wrapper.numba = numba_func
        wrapper.is_jit_enabled = numba_func is not None
        wrapper.full_name = full_name
        
        return wrapper
    
    # Support both @numba_switchable and @numba_switchable(...)
    if func is not None:
        return decorator(func)
    return decorator


# ============================================================================
# Utility Functions
# ============================================================================

def disable_globally():
    """Disable Numba globally."""
    config.ENABLED_GLOBAL = False


def enable_globally():
    """Enable Numba globally."""
    config.ENABLED_GLOBAL = True


def disable_module(module_name: str):
    """Disable Numba for a specific module."""
    config.MODULE_OVERRIDES[module_name] = False


def enable_module(module_name: str):
    """Enable Numba for a specific module."""
    config.MODULE_OVERRIDES[module_name] = True


def disable_function(full_name: str):
    """Disable Numba for a specific function."""
    config.FUNCTION_OVERRIDES[full_name] = False


def enable_function(full_name: str):
    """Enable Numba for a specific function."""
    config.FUNCTION_OVERRIDES[full_name] = True


def reset_config():
    """Reset all overrides."""
    config.ENABLED_GLOBAL = True
    config.MODULE_OVERRIDES.clear()
    config.FUNCTION_OVERRIDES.clear()


def get_cache_dir() -> Path | None:
    """Get the current cache directory."""
    cache_env = os.environ.get('NUMBA_CACHE_DIR')
    if cache_env:
        return Path(cache_env)
    return None


def clear_cache():
    """Clear all cached compiled functions."""
    import shutil
    
    cache_dir = get_cache_dir()
    if cache_dir and cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleared cache: {cache_dir}")
    else:
        print("No custom cache directory configured.")


def print_status():
    """Print current configuration status."""
    cache_dir = get_cache_dir()
    
    print("=" * 50)
    print("Numba Configuration Status")
    print("=" * 50)
    print(f"Cache Directory: {cache_dir or '(default)'}")
    print(f"Global Enabled: {config.ENABLED_GLOBAL}")
    print(f"Module Overrides: {config.MODULE_OVERRIDES or '(none)'}")
    print(f"Function Overrides: {config.FUNCTION_OVERRIDES or '(none)'}")
    print("=" * 50)
