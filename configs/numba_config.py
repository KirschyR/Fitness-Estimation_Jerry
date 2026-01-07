"""
Numba Configuration
===================

Controls Numba JIT compilation behavior at three levels:
1. FUNCTION_OVERRIDES - Highest priority, control specific functions
2. MODULE_OVERRIDES   - Medium priority, control entire modules  
3. ENABLED_GLOBAL     - Lowest priority, global default

Function/Module names use fully qualified format:
- Function: 'module.submodule.function' or 'module.submodule.Class.method'
- Module: 'module.submodule'
"""

from pathlib import Path

# ============================================================================
# Cache Configuration
# ============================================================================
# Directory for Numba cache files
# - None: Use default (each module's __pycache__ folder)
# - Path or str: Use specified directory (will be created if not exists)
CACHE_DIR: Path | str | None = None

# Examples:
# CACHE_DIR = None                          # Default behavior
# CACHE_DIR = ".numba_cache"                # Relative to working directory
# CACHE_DIR = "/tmp/numba_cache"            # Absolute path
# CACHE_DIR = Path.home() / ".cache/numba"  # User cache directory

# ============================================================================
# Global Switch (lowest priority)
# ============================================================================
ENABLED_GLOBAL: bool = True

# ============================================================================
# Module-level Overrides (medium priority)
# ============================================================================
# Format: 'module_name': bool
MODULE_OVERRIDES: dict[str, bool] = {
    'utils.algorithms': False,   # Disable for entire module
}

# ============================================================================
# Function-level Overrides (highest priority)
# ============================================================================
# Format: 'module.function' or 'module.Class.method': bool
FUNCTION_OVERRIDES: dict[str, bool] = {
    # 'utils.algorithms.generate_offspring_distribution': True,
    'utils.simulation_kernels.run_tick': False,  # 包含非 Numba 兼容代码
}
