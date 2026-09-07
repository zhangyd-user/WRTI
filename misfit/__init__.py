"""Fixed-mask VFSA reflection traveltime objective for WRTI Step 6."""

from .mask import FixedMaskResult, MisfitError, build_fixed_mask
from .objective import (
    MisfitConfig,
    MisfitResult,
    build_fixed_mask_from_config,
    compute_misfit,
    compute_vfsa_misfit,
)

__all__ = [
    "FixedMaskResult",
    "MisfitError",
    "MisfitConfig",
    "MisfitResult",
    "build_fixed_mask",
    "build_fixed_mask_from_config",
    "compute_misfit",
    "compute_vfsa_misfit",
]
