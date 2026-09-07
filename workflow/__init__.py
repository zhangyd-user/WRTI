"""Lifecycle-safe WRTI outer-stage builder and VFSA objective evaluator."""

from .adapters import AdapterError, adfwi_to_wrti, from_adfwi
from .builder import OuterStageWindowBuilder, WorkflowBuildError
from .config import GridConfig, WRTIConfig, WorkflowConfigError
from .evaluator import EvaluatorError, WRTIObjectiveEvaluator
from .observed_center import (
    DIRECT_TRACKING,
    QUADRATIC_ONLY,
    QUADRATIC_PLUS_ENVELOPE,
    ObservedCenterRecoveryResult,
    recover_edge_observed_centers,
    select_observed_center_anchors,
    snap_observed_center,
    weighted_quadratic_fit,
)
from .state import ReferenceStateError, WRTIReferenceState

__all__ = [
    "AdapterError",
    "EvaluatorError",
    "GridConfig",
    "OuterStageWindowBuilder",
    "ReferenceStateError",
    "WRTIConfig",
    "WRTIObjectiveEvaluator",
    "WRTIReferenceState",
    "WorkflowBuildError",
    "WorkflowConfigError",
    "DIRECT_TRACKING",
    "QUADRATIC_ONLY",
    "QUADRATIC_PLUS_ENVELOPE",
    "ObservedCenterRecoveryResult",
    "recover_edge_observed_centers",
    "select_observed_center_anchors",
    "snap_observed_center",
    "weighted_quadratic_fit",
    "adfwi_to_wrti",
    "from_adfwi",
]
