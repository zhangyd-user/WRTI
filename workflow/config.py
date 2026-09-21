"""Unified YAML-backed configuration for the WRTI workflow layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..misfit import MisfitConfig, MisfitError
from ..tracking.quality_control import QualityAuditConfig


class WorkflowConfigError(ValueError):
    """Raised when the WRTI YAML configuration is incomplete or invalid."""


@dataclass(frozen=True)
class GridConfig:
    """Regular model-grid definition used by Step 1/2."""

    x0: float
    z0: float
    dx: float
    dz: float
    nx: int
    nz: int
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        values = np.asarray([self.x0, self.z0, self.dx, self.dz, self.tolerance])
        if not np.isfinite(values).all() or self.dx <= 0 or self.dz <= 0:
            raise WorkflowConfigError("grid origins, spacings, and tolerance must be finite; dx/dz > 0.")
        if self.tolerance < 0:
            raise WorkflowConfigError("grid.tolerance must be non-negative.")
        if (
            isinstance(self.nx, bool)
            or isinstance(self.nz, bool)
            or int(self.nx) != self.nx
            or int(self.nz) != self.nz
            or self.nx <= 0
            or self.nz <= 0
        ):
            raise WorkflowConfigError("grid.nx and grid.nz must be positive integers.")
        object.__setattr__(self, "x0", float(self.x0))
        object.__setattr__(self, "z0", float(self.z0))
        object.__setattr__(self, "dx", float(self.dx))
        object.__setattr__(self, "dz", float(self.dz))
        object.__setattr__(self, "nx", int(self.nx))
        object.__setattr__(self, "nz", int(self.nz))
        object.__setattr__(self, "tolerance", float(self.tolerance))

    @property
    def x_axis(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.nx, dtype=float)

    @property
    def z_axis(self) -> np.ndarray:
        return self.z0 + self.dz * np.arange(self.nz, dtype=float)


@dataclass(frozen=True)
class WRTIConfig:
    """Frozen snapshot of all workflow settings used by one outer stage."""

    grid: GridConfig
    window_type: str
    half_window_time: float | tuple[float, ...]
    center_time_shift: float
    tukey_alpha: float
    max_lag_time: float
    seed_lag_range_time: float
    epsilon_time: float
    tracking_min_correlation: float | None
    boundary_margin_samples: int
    misfit: MisfitConfig
    eikonal_workers: int = 16
    wrti_workers: int = 16
    save_correlation_for: tuple[tuple[int, int], ...] = ()
    diagnostics_output_dir: str = ""
    log_level: str = "INFO"
    verbose: bool = False
    use_envelope_coarse: bool = True
    # Waveform ZNCC half-width around the selected coarse lag.
    envelope_fine_half_width_time: float = 0.04
    # Receiver-to-receiver envelope continuity radius; distinct from the
    # waveform ZNCC basin above.
    envelope_tracking_epsilon_time: float = 0.040
    tracking_enhancement_enabled: bool = True
    tracking_agc_fraction: float = 0.25
    tracking_agc_floor_ratio: float = 0.20
    tracking_receiver_stack: bool = True
    tracking_raw_refine_radius_samples: int = 1
    flat_tracking_slope_penalty: float = 0.10
    flat_tracking_refine_radius_samples: int = 3
    flat_tracking_method: str = "dense_envelope_dp"
    flat_candidate_min_distance_time: float = 0.015
    flat_candidate_min_prominence: float = 0.05
    flat_max_candidates_per_trace: int = 64
    flat_coherence_half_window_time: float = 0.040
    flat_min_neighbor_zncc: float = 0.50
    flat_hard_neighbor_zncc_gate: bool = False
    flat_max_residual_slope_ms_per_100m: float = 150.0
    flat_prediction_soft_scale_time: float = 0.010
    flat_prediction_penalty_weight: float = 0.20
    flat_max_prediction_error_time: float = 0.050
    flat_max_active_states: int = 2000
    flat_ownership_enabled: bool = True
    flat_ownership_overlap_fraction: float = 0.10
    flat_ownership_edge_fraction: float = 0.60
    flat_ownership_escape_enabled: bool = False
    flat_ownership_escape_min_neighbor_similarity: float = 0.85
    flat_ownership_escape_max_prediction_error_time: float = 0.015
    flat_ownership_escape_extra_gap_fraction: float = 0.20
    flat_ownership_escape_max_extra_time: float = 0.120
    flat_seed_snap_enabled: bool = True
    flat_seed_snap_half_width_time: float = 0.030
    flat_compute_legacy_dense_diagnostic: bool = False
    flat_legacy_dense_diagnostic_shots: tuple[int, ...] = ()
    quality_audit: QualityAuditConfig = field(default_factory=QualityAuditConfig)
    dual_center_enabled: bool = False
    dual_center_use_candidate_eikonal: bool = True
    dual_center_local_max_shift_time: float = 0.100
    dual_center_archive_each_evaluation: bool = True

    def __post_init__(self) -> None:
        if self.window_type not in {"rectangular", "tukey"}:
            raise WorkflowConfigError(
                "window.type must be 'rectangular' or 'tukey'."
            )
        half = self.half_window_time
        if isinstance(half, tuple):
            half_values = np.asarray(half, dtype=float)
        else:
            half_values = np.asarray([half], dtype=float)
        if half_values.ndim != 1 or not np.isfinite(half_values).all() or np.any(half_values < 0):
            raise WorkflowConfigError(
                "window.half_window_time must be a finite scalar or non-negative list."
            )
        if not np.isfinite(self.center_time_shift):
            raise WorkflowConfigError("window.center_time_shift must be finite.")
        if not np.isfinite(self.tukey_alpha) or not 0 <= self.tukey_alpha <= 1:
            raise WorkflowConfigError("window.tukey_alpha must be in [0, 1].")
        for value, name in (
            (self.max_lag_time, "correlation.max_lag_time"),
            (self.seed_lag_range_time, "correlation.seed_lag_range_time"),
            (self.epsilon_time, "tracking.epsilon_time"),
        ):
            if not np.isfinite(value) or value < 0:
                raise WorkflowConfigError(f"{name} must be finite and non-negative.")
        if self.tracking_min_correlation is not None and (
            not np.isfinite(self.tracking_min_correlation)
            or not -1.0 <= self.tracking_min_correlation <= 1.0
        ):
            raise WorkflowConfigError(
                "tracking.min_correlation must be in [-1, 1] or null."
            )
        if (
            isinstance(self.boundary_margin_samples, bool)
            or int(self.boundary_margin_samples) != self.boundary_margin_samples
            or self.boundary_margin_samples < 0
        ):
            raise WorkflowConfigError("qc.boundary_margin must be a non-negative integer.")
        if (
            isinstance(self.eikonal_workers, bool)
            or int(self.eikonal_workers) != self.eikonal_workers
            or self.eikonal_workers < 1
        ):
            raise WorkflowConfigError(
                "parallel.eikonal_workers must be a positive integer."
            )
        if (
            isinstance(self.wrti_workers, bool)
            or int(self.wrti_workers) != self.wrti_workers
            or self.wrti_workers < 1
        ):
            raise WorkflowConfigError(
                "parallel.wrti_workers must be a positive integer."
            )
        if not isinstance(self.use_envelope_coarse, (bool, np.bool_)):
            raise WorkflowConfigError("correlation.use_envelope_coarse must be boolean.")
        if not isinstance(self.dual_center_enabled, (bool, np.bool_)):
            raise WorkflowConfigError("dual_center_windows.enabled must be boolean.")
        if not isinstance(
            self.dual_center_use_candidate_eikonal, (bool, np.bool_)
        ):
            raise WorkflowConfigError(
                "dual_center_windows.use_candidate_eikonal must be boolean."
            )
        if not isinstance(
            self.dual_center_archive_each_evaluation, (bool, np.bool_)
        ):
            raise WorkflowConfigError(
                "dual_center_windows.archive_each_evaluation must be boolean."
            )
        if (
            not np.isfinite(self.dual_center_local_max_shift_time)
            or self.dual_center_local_max_shift_time < 0
        ):
            raise WorkflowConfigError(
                "dual_center_windows.local_max_shift_time must be finite and non-negative."
            )
        if (
            not np.isfinite(self.envelope_fine_half_width_time)
            or self.envelope_fine_half_width_time < 0
        ):
            raise WorkflowConfigError(
                "correlation.envelope_fine_half_width_time must be finite and non-negative."
            )
        if (
            not np.isfinite(self.envelope_tracking_epsilon_time)
            or self.envelope_tracking_epsilon_time <= 0
        ):
            raise WorkflowConfigError(
                "correlation.envelope_tracking_epsilon_time must be finite and positive."
            )
        if not isinstance(self.tracking_enhancement_enabled, (bool, np.bool_)):
            raise WorkflowConfigError("tracking.enhancement_enabled must be boolean.")
        if not np.isfinite(self.flat_tracking_slope_penalty) or self.flat_tracking_slope_penalty < 0:
            raise WorkflowConfigError("bootstrap_tracking.slope_penalty must be finite and non-negative.")
        if (
            isinstance(self.flat_tracking_refine_radius_samples, bool)
            or int(self.flat_tracking_refine_radius_samples) != self.flat_tracking_refine_radius_samples
            or self.flat_tracking_refine_radius_samples < 0
        ):
            raise WorkflowConfigError("bootstrap_tracking.refine_radius_samples must be a non-negative integer.")
        if self.flat_tracking_method not in {"dense_envelope_dp", "sparse_event_dp"}:
            raise WorkflowConfigError("bootstrap_tracking.method is invalid.")
        for value, name in (
            (self.flat_candidate_min_distance_time, "candidate_min_distance_time"),
            (self.flat_candidate_min_prominence, "candidate_min_prominence"),
            (self.flat_coherence_half_window_time, "coherence_half_window_time"),
            (self.flat_max_residual_slope_ms_per_100m, "max_residual_slope_ms_per_100m"),
            (self.flat_max_prediction_error_time, "max_prediction_error_time"),
            (self.flat_prediction_soft_scale_time, "prediction_soft_scale_time"),
            (self.flat_prediction_penalty_weight, "prediction_penalty_weight"),
            (self.flat_ownership_overlap_fraction, "ownership_overlap_fraction"),
            (self.flat_ownership_edge_fraction, "ownership_edge_fraction"),
            (self.flat_ownership_escape_min_neighbor_similarity, "ownership_escape.min_neighbor_similarity"),
            (self.flat_ownership_escape_max_prediction_error_time, "ownership_escape.max_prediction_error_time"),
            (self.flat_ownership_escape_extra_gap_fraction, "ownership_escape.extra_gap_fraction"),
            (self.flat_ownership_escape_max_extra_time, "ownership_escape.max_extra_time"),
            (self.flat_seed_snap_half_width_time, "seed_snap_half_width_time"),
        ):
            if not np.isfinite(value) or value < 0:
                raise WorkflowConfigError(f"bootstrap_tracking.{name} must be finite and non-negative.")
        if not np.isfinite(self.flat_min_neighbor_zncc) or not -1 <= self.flat_min_neighbor_zncc <= 1:
            raise WorkflowConfigError("bootstrap_tracking.min_neighbor_zncc must be in [-1, 1].")
        if self.flat_prediction_soft_scale_time <= 0:
            raise WorkflowConfigError("bootstrap_tracking.prediction_soft_scale_time must be positive.")
        for value, name in (
            (self.flat_max_candidates_per_trace, "max_candidates_per_trace"),
            (self.flat_max_active_states, "max_active_states"),
        ):
            if isinstance(value, bool) or int(value) != value or value < (0 if name == "max_candidates_per_trace" else 1):
                raise WorkflowConfigError(f"bootstrap_tracking.{name} must be a non-negative integer.")
        if (not isinstance(self.flat_hard_neighbor_zncc_gate, (bool, np.bool_))
                or not isinstance(self.flat_ownership_enabled, (bool, np.bool_))
                or not isinstance(self.flat_seed_snap_enabled, (bool, np.bool_))
                or not isinstance(self.flat_ownership_escape_enabled, (bool, np.bool_))
                or not isinstance(self.flat_compute_legacy_dense_diagnostic, (bool, np.bool_))):
            raise WorkflowConfigError("bootstrap_tracking ownership/ZNCC switches must be boolean.")
        if any(isinstance(shot, bool) or int(shot) != shot or shot < 1 for shot in self.flat_legacy_dense_diagnostic_shots):
            raise WorkflowConfigError("bootstrap_tracking.legacy_dense_diagnostic_shots uses positive 1-based shot numbers.")
        if (
            not np.isfinite(self.tracking_agc_fraction)
            or not 0 < self.tracking_agc_fraction <= 1
        ):
            raise WorkflowConfigError("tracking.agc_fraction must be in (0, 1].")
        if not np.isfinite(self.tracking_agc_floor_ratio) or self.tracking_agc_floor_ratio <= 0:
            raise WorkflowConfigError("tracking.agc_floor_ratio must be positive.")
        if not isinstance(self.tracking_receiver_stack, (bool, np.bool_)):
            raise WorkflowConfigError("tracking.receiver_stack must be boolean.")
        if (
            isinstance(self.tracking_raw_refine_radius_samples, bool)
            or int(self.tracking_raw_refine_radius_samples) != self.tracking_raw_refine_radius_samples
            or self.tracking_raw_refine_radius_samples < 0
        ):
            raise WorkflowConfigError(
                "tracking.raw_refine_radius_samples must be a non-negative integer."
            )
        for pair in self.save_correlation_for:
            if len(pair) != 2 or any(int(value) != value or int(value) < 0 for value in pair):
                raise WorkflowConfigError(
                    "diagnostics.save_correlation_for must contain [reflector, shot] pairs."
                )
        object.__setattr__(self, "window_type", str(self.window_type))
        if isinstance(half, tuple):
            object.__setattr__(self, "half_window_time", tuple(float(value) for value in half))
        else:
            object.__setattr__(self, "half_window_time", float(half))
        object.__setattr__(self, "center_time_shift", float(self.center_time_shift))
        object.__setattr__(self, "tukey_alpha", float(self.tukey_alpha))
        object.__setattr__(self, "max_lag_time", float(self.max_lag_time))
        object.__setattr__(self, "seed_lag_range_time", float(self.seed_lag_range_time))
        object.__setattr__(self, "epsilon_time", float(self.epsilon_time))
        if self.tracking_min_correlation is not None:
            object.__setattr__(
                self,
                "tracking_min_correlation",
                float(self.tracking_min_correlation),
            )
        object.__setattr__(self, "boundary_margin_samples", int(self.boundary_margin_samples))
        object.__setattr__(self, "eikonal_workers", int(self.eikonal_workers))
        object.__setattr__(self, "wrti_workers", int(self.wrti_workers))
        object.__setattr__(self, "use_envelope_coarse", bool(self.use_envelope_coarse))
        object.__setattr__(self, "dual_center_enabled", bool(self.dual_center_enabled))
        object.__setattr__(
            self,
            "dual_center_use_candidate_eikonal",
            bool(self.dual_center_use_candidate_eikonal),
        )
        object.__setattr__(
            self,
            "dual_center_local_max_shift_time",
            float(self.dual_center_local_max_shift_time),
        )
        object.__setattr__(
            self,
            "dual_center_archive_each_evaluation",
            bool(self.dual_center_archive_each_evaluation),
        )
        object.__setattr__(
            self,
            "envelope_fine_half_width_time",
            float(self.envelope_fine_half_width_time),
        )
        object.__setattr__(
            self,
            "envelope_tracking_epsilon_time",
            float(self.envelope_tracking_epsilon_time),
        )
        object.__setattr__(
            self, "tracking_enhancement_enabled", bool(self.tracking_enhancement_enabled)
        )
        object.__setattr__(self, "tracking_agc_fraction", float(self.tracking_agc_fraction))
        object.__setattr__(
            self, "tracking_agc_floor_ratio", float(self.tracking_agc_floor_ratio)
        )
        object.__setattr__(self, "flat_tracking_slope_penalty", float(self.flat_tracking_slope_penalty))
        object.__setattr__(self, "flat_tracking_refine_radius_samples", int(self.flat_tracking_refine_radius_samples))
        object.__setattr__(self, "flat_tracking_method", str(self.flat_tracking_method))
        for name in (
            "flat_candidate_min_distance_time", "flat_candidate_min_prominence",
            "flat_coherence_half_window_time", "flat_min_neighbor_zncc",
            "flat_max_residual_slope_ms_per_100m", "flat_max_prediction_error_time",
            "flat_prediction_soft_scale_time", "flat_prediction_penalty_weight",
            "flat_ownership_overlap_fraction", "flat_ownership_edge_fraction",
            "flat_ownership_escape_min_neighbor_similarity",
            "flat_ownership_escape_max_prediction_error_time",
            "flat_ownership_escape_extra_gap_fraction", "flat_ownership_escape_max_extra_time",
            "flat_seed_snap_half_width_time",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "flat_max_candidates_per_trace", int(self.flat_max_candidates_per_trace))
        object.__setattr__(self, "flat_max_active_states", int(self.flat_max_active_states))
        object.__setattr__(self, "flat_hard_neighbor_zncc_gate", bool(self.flat_hard_neighbor_zncc_gate))
        object.__setattr__(self, "flat_ownership_enabled", bool(self.flat_ownership_enabled))
        object.__setattr__(self, "flat_ownership_escape_enabled", bool(self.flat_ownership_escape_enabled))
        object.__setattr__(self, "flat_seed_snap_enabled", bool(self.flat_seed_snap_enabled))
        object.__setattr__(self, "flat_compute_legacy_dense_diagnostic", bool(self.flat_compute_legacy_dense_diagnostic))
        object.__setattr__(self, "flat_legacy_dense_diagnostic_shots", tuple(int(shot) for shot in self.flat_legacy_dense_diagnostic_shots))
        object.__setattr__(self, "tracking_receiver_stack", bool(self.tracking_receiver_stack))
        object.__setattr__(
            self,
            "tracking_raw_refine_radius_samples",
            int(self.tracking_raw_refine_radius_samples),
        )
        object.__setattr__(
            self,
            "save_correlation_for",
            tuple((int(pair[0]), int(pair[1])) for pair in self.save_correlation_for),
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "WRTIConfig":
        root = dict(mapping)
        for section in ("grid", "window", "correlation", "tracking", "qc"):
            if not isinstance(root.get(section), Mapping):
                raise WorkflowConfigError(f"wrti.yaml requires a '{section}' mapping.")
        grid = root["grid"]
        window = root["window"]
        correlation = root["correlation"]
        tracking = root["tracking"]
        bootstrap_tracking = root.get("bootstrap_tracking", {})
        if not isinstance(bootstrap_tracking, Mapping):
            raise WorkflowConfigError("bootstrap_tracking must be a mapping.")
        ownership_escape = bootstrap_tracking.get("ownership_escape", {})
        if not isinstance(ownership_escape, Mapping):
            raise WorkflowConfigError("bootstrap_tracking.ownership_escape must be a mapping.")
        quality_audit = root.get("quality_audit", {})
        if not isinstance(quality_audit, Mapping):
            raise WorkflowConfigError("quality_audit must be a mapping.")
        dual_center = root.get("dual_center_windows", {})
        if not isinstance(dual_center, Mapping):
            raise WorkflowConfigError("dual_center_windows must be a mapping.")
        qc = root["qc"]
        eikonal = root.get("eikonal", {})
        parallel = root.get("parallel", {})
        diagnostics = root.get("diagnostics", {})
        logging = root.get("logging", {})
        if (
            not isinstance(eikonal, Mapping)
            or not isinstance(parallel, Mapping)
            or not isinstance(diagnostics, Mapping)
            or not isinstance(logging, Mapping)
        ):
            raise WorkflowConfigError(
                "parallel, eikonal, diagnostics, and logging must be mappings."
            )
        if correlation.get("method", "zncc") != "zncc":
            raise WorkflowConfigError("Only correlation.method='zncc' is supported.")
        half = window.get("half_window_time")
        if half is None:
            raise WorkflowConfigError("window.half_window_time is required.")
        if isinstance(half, list):
            half = tuple(float(value) for value in half)

        try:
            misfit = MisfitConfig.from_mapping(root)
        except MisfitError as exc:
            raise WorkflowConfigError(str(exc)) from exc
        try:
            audit_config = QualityAuditConfig(**quality_audit)
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f"invalid quality_audit configuration: {exc}") from exc

        pairs = diagnostics.get("save_correlation_for", ())
        if pairs is None:
            pairs = ()
        try:
            pairs = tuple((int(pair[0]), int(pair[1])) for pair in pairs)
        except (TypeError, IndexError, ValueError) as exc:
            raise WorkflowConfigError(
                "diagnostics.save_correlation_for must contain [reflector, shot] pairs."
            ) from exc

        return cls(
            grid=GridConfig(
                x0=grid["x0"],
                z0=grid["z0"],
                dx=grid["dx"],
                dz=grid["dz"],
                nx=grid["nx"],
                nz=grid["nz"],
                tolerance=grid.get("tolerance", 1e-6),
            ),
            window_type=window["type"],
            half_window_time=half,
            center_time_shift=window.get("center_time_shift", 0.0),
            tukey_alpha=window.get("tukey_alpha", 0.5),
            max_lag_time=correlation["max_lag_time"],
            seed_lag_range_time=correlation["seed_lag_range_time"],
            use_envelope_coarse=correlation.get("use_envelope_coarse", True),
            envelope_fine_half_width_time=correlation.get(
                "envelope_fine_half_width_time", 0.04
            ),
            envelope_tracking_epsilon_time=correlation.get(
                "envelope_tracking_epsilon_time", 0.040
            ),
            epsilon_time=tracking["epsilon_time"],
            tracking_min_correlation=tracking.get("min_correlation"),
            tracking_enhancement_enabled=tracking.get("enhancement_enabled", True),
            tracking_agc_fraction=tracking.get("agc_fraction", 0.25),
            tracking_agc_floor_ratio=tracking.get("agc_floor_ratio", 0.20),
            tracking_receiver_stack=tracking.get("receiver_stack", True),
            tracking_raw_refine_radius_samples=tracking.get(
                "raw_refine_radius_samples", 1
            ),
            flat_tracking_slope_penalty=bootstrap_tracking.get("slope_penalty", 0.10),
            flat_tracking_refine_radius_samples=bootstrap_tracking.get(
                "refine_radius_samples", 3
            ),
            flat_tracking_method=bootstrap_tracking.get("method", "dense_envelope_dp"),
            flat_candidate_min_distance_time=bootstrap_tracking.get("candidate_min_distance_time", 0.015),
            flat_candidate_min_prominence=bootstrap_tracking.get("candidate_min_prominence", 0.05),
            flat_max_candidates_per_trace=bootstrap_tracking.get("max_candidates_per_trace", 64),
            flat_coherence_half_window_time=bootstrap_tracking.get("coherence_half_window_time", 0.040),
            flat_min_neighbor_zncc=bootstrap_tracking.get("min_neighbor_zncc", 0.50),
            flat_hard_neighbor_zncc_gate=bootstrap_tracking.get("hard_neighbor_zncc_gate", False),
            flat_max_residual_slope_ms_per_100m=bootstrap_tracking.get("max_residual_slope_ms_per_100m", 150.0),
            flat_prediction_soft_scale_time=bootstrap_tracking.get("prediction_soft_scale_time", 0.010),
            flat_prediction_penalty_weight=bootstrap_tracking.get("prediction_penalty_weight", 0.20),
            flat_max_prediction_error_time=bootstrap_tracking.get("max_prediction_error_time", 0.050),
            flat_max_active_states=bootstrap_tracking.get("max_active_states", 2000),
            flat_ownership_enabled=bootstrap_tracking.get("ownership_enabled", True),
            flat_ownership_overlap_fraction=bootstrap_tracking.get("ownership_overlap_fraction", 0.10),
            flat_ownership_edge_fraction=bootstrap_tracking.get("ownership_edge_fraction", 0.60),
            flat_ownership_escape_enabled=ownership_escape.get("enabled", False),
            flat_ownership_escape_min_neighbor_similarity=ownership_escape.get("min_neighbor_similarity", 0.85),
            flat_ownership_escape_max_prediction_error_time=ownership_escape.get("max_prediction_error_time", 0.015),
            flat_ownership_escape_extra_gap_fraction=ownership_escape.get("extra_gap_fraction", 0.20),
            flat_ownership_escape_max_extra_time=ownership_escape.get("max_extra_time", 0.120),
            flat_seed_snap_enabled=bootstrap_tracking.get("seed_snap_enabled", True),
            flat_seed_snap_half_width_time=bootstrap_tracking.get("seed_snap_half_width_time", 0.030),
            flat_compute_legacy_dense_diagnostic=bootstrap_tracking.get("compute_legacy_dense_diagnostic", False),
            flat_legacy_dense_diagnostic_shots=tuple(bootstrap_tracking.get("legacy_dense_diagnostic_shots", ())),
            quality_audit=audit_config,
            boundary_margin_samples=qc.get("boundary_margin", 0),
            # ``parallel`` is the unified schema.  The old eikonal mapping is
            # accepted only as a compatibility fallback for existing callers.
            eikonal_workers=parallel.get(
                "eikonal_workers", eikonal.get("workers", 16)
            ),
            wrti_workers=parallel.get("wrti_workers", 16),
            misfit=misfit,
            save_correlation_for=pairs,
            diagnostics_output_dir=str(diagnostics.get("output_dir", "")),
            log_level=str(logging.get("log_level", "INFO")),
            verbose=bool(logging.get("verbose", False)),
            dual_center_enabled=dual_center.get("enabled", False),
            dual_center_use_candidate_eikonal=dual_center.get(
                "use_candidate_eikonal", True
            ),
            dual_center_local_max_shift_time=dual_center.get(
                "local_max_shift_time", 0.100
            ),
            dual_center_archive_each_evaluation=dual_center.get(
                "archive_each_evaluation", True
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WRTIConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise WorkflowConfigError(
                "PyYAML is required to load WRTIConfig from YAML."
            ) from exc
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, Mapping):
            raise WorkflowConfigError("The WRTI YAML root must be a mapping.")
        return cls.from_mapping(loaded)
