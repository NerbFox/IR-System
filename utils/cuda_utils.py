"""CUDA utility module for array computations.

This module provides unified array computation interface using either CuPy (GPU)
or NumPy (CPU) based on hardware availability. It exposes:
- xp: The array computation module (either cupy or numpy)
- GPU_AVAILABLE: Boolean indicating if GPU computation is available
"""

try:
    import cupy as cp
    if cp.cuda.is_available():
        xp = cp
        GPU_AVAILABLE = True
    else:
        raise ImportError("CuPy is not available or GPU is not available.")
except ImportError:
    import numpy as np
    xp = np
    GPU_AVAILABLE = False
