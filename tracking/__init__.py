"""Seed-centered bidirectional global ZNCC tracking for WRTI Step 5."""

from .tracker import (
    TrackingError,
    TrackingResult,
    track_correlation_result,
    track_zncc,
)
from .fallback import ShiftCompletionResult, complete_tracked_shift
from .flat_event import (
    FlatEventCandidate,
    SparseFlatTrackingResult,
    track_flattened_event_sparse,
)
from .quality import boundary_qc_mask, low_correlation_qc_mask, path_failure_mask
from .quality_control import (
    QualityAuditConfig,
    RkShotAudit,
    SideAudit,
    TrackingStatus,
    audit_rkshot,
    audit_tracking_side,
)
from .seed_search import SeedSearchResult, discover_tracking_seed

__all__ = [
    "TrackingError",
    "TrackingResult",
    "track_correlation_result",
    "track_zncc",
    "path_failure_mask",
    "low_correlation_qc_mask",
    "boundary_qc_mask",
    "QualityAuditConfig",
    "RkShotAudit",
    "SideAudit",
    "TrackingStatus",
    "audit_rkshot",
    "audit_tracking_side",
    "ShiftCompletionResult",
    "complete_tracked_shift",
    "FlatEventCandidate",
    "SparseFlatTrackingResult",
    "track_flattened_event_sparse",
    "SeedSearchResult",
    "discover_tracking_seed",
]
