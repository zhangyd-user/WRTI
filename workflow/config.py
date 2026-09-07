"""Unified YAML-backed configuration for the WRTI workflow layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..misfit import MisfitConfig, MisfitError


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
    # Compatibility name: this threshold is QC-only and never deletes a path.
    tracking_min_correlation: float | None
    boundary_margin_samples: int
    misfit: MisfitConfig
    eikonal_workers: int = 16
    wrti_workers: int = 16
    save_correlation_for: tuple[tuple[int, int], ...] = ()
    diagnostics_output_dir: str = ""
    debug_tracking_shots: tuple[int, ...] = ()
    debug_tracking_reflectors: tuple[int, ...] = ()
    save_tracking_snapshot: bool = False
    log_level: str = "INFO"
    verbose: bool = False
    use_envelope_coarse: bool = True
    # Compatibility key retained as the default coarse soft-prior width.
    envelope_fine_half_width_time: float = 0.04
    # Receiver-to-receiver envelope continuity radius; distinct from the
    # waveform ZNCC basin above.
    envelope_tracking_epsilon_time: float = 0.040
    tracking_enhancement_enabled: bool = True
    tracking_agc_fraction: float = 0.25
    tracking_agc_floor_ratio: float = 0.20
    tracking_receiver_stack: bool = True
    tracking_raw_refine_radius_samples: int = 1
    ownership_guard_time: float = 0.02
    tracking_top_k_peaks: int = 4
    tracking_peak_min_separation_samples: int = 2
    tracking_coarse_soft_width_time: float = 0.05
    tracking_coarse_soft_weight: float = 0.05
    tracking_coarse_soft_penalty_cap: float = 0.25
    tracking_smooth_weight: float = 0.02
    tracking_max_residual_jump_time: float | None = None
    tracking_max_skip_rows: int = 3
    tracking_gap_penalty: float = 0.05
    tracking_correlation_mode: str = "auto_polarity"
    tracking_tracker_mode: str = "sparse_global"
    tracking_restart_enabled: bool = True
    tracking_restart_confirm_rows: int = 3
    tracking_restart_min_mean_correlation: float = 0.55
    tracking_restart_max_coarse_deviation_time: float = 0.10
    local_search_half_width_time: float = 0.040
    max_search_half_width_time: float = 0.080
    gap_expand_time: float = 0.010
    residual_history: int = 4
    min_peak_margin: float = 0.03
    prediction_weight: float = 0.10
    max_gap_rows: int = 4
    relock_confirm_rows: int = 3
    restart_min_correlation: float = 0.60
    bridge_enabled: bool = True
    bridge_half_width_time: float = 0.05
    bridge_max_width_time: float = 0.08
    bridge_max_receivers: int = 8
    bridge_max_distance: float | None = None

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
        for value, name in (
            (self.ownership_guard_time, "correlation.ownership_guard_time"),
            (self.tracking_coarse_soft_width_time, "tracking.coarse_soft_width_time"),
            (self.tracking_coarse_soft_weight, "tracking.coarse_soft_weight"),
            (self.tracking_coarse_soft_penalty_cap, "tracking.coarse_soft_penalty_cap"),
            (self.tracking_smooth_weight, "tracking.smooth_weight"),
            (self.tracking_gap_penalty, "tracking.gap_penalty"),
            (self.tracking_restart_min_mean_correlation, "tracking.restart_min_mean_correlation"),
            (self.tracking_restart_max_coarse_deviation_time, "tracking.restart_max_coarse_deviation_time"),
            (self.bridge_half_width_time, "tracking.bridge_half_width_time"),
            (self.bridge_max_width_time, "tracking.bridge_max_width_time"),
        ):
            if not np.isfinite(value) or value < 0:
                raise WorkflowConfigError(f"{name} must be finite and non-negative.")
        if self.tracking_coarse_soft_width_time == 0:
            raise WorkflowConfigError("tracking.coarse_soft_width_time must be positive.")
        if self.bridge_max_width_time < self.bridge_half_width_time:
            raise WorkflowConfigError(
                "tracking.bridge_max_width_time must be at least bridge_half_width_time."
            )
        if self.tracking_max_residual_jump_time is not None and (
            not np.isfinite(self.tracking_max_residual_jump_time)
            or self.tracking_max_residual_jump_time < 0
        ):
            raise WorkflowConfigError("tracking.max_residual_jump_time must be non-negative or null.")
        if self.tracking_correlation_mode not in {"positive", "negative", "absolute", "auto_polarity"}:
            raise WorkflowConfigError("tracking.correlation_mode must be positive, negative, absolute, or auto_polarity.")
        if self.tracking_tracker_mode not in {"sparse_global", "local"}:
            raise WorkflowConfigError("tracking.tracker_mode must be sparse_global or local.")
        if not isinstance(self.tracking_restart_enabled, (bool, np.bool_)):
            raise WorkflowConfigError("tracking.restart_enabled must be boolean.")
        for value, name, positive in (
            (self.tracking_top_k_peaks, "tracking.top_k_peaks", True),
            (
                self.tracking_peak_min_separation_samples,
                "tracking.peak_min_separation_samples",
                False,
            ),
            (self.bridge_max_receivers, "tracking.bridge_max_receivers", False),
            (self.tracking_max_skip_rows, "tracking.max_skip_rows", False),
            (self.tracking_restart_confirm_rows, "tracking.restart_confirm_rows", True),
        ):
            if (
                isinstance(value, bool)
                or int(value) != value
                or int(value) < (1 if positive else 0)
            ):
                qualifier = "positive" if positive else "non-negative"
                raise WorkflowConfigError(f"{name} must be a {qualifier} integer.")
        if not isinstance(self.bridge_enabled, (bool, np.bool_)):
            raise WorkflowConfigError("tracking.bridge_enabled must be boolean.")
        if self.bridge_max_distance is not None and (
            not np.isfinite(self.bridge_max_distance) or self.bridge_max_distance < 0
        ):
            raise WorkflowConfigError(
                "tracking.bridge_max_distance must be finite and non-negative or null."
            )
        for pair in self.save_correlation_for:
            if len(pair) != 2 or any(int(value) != value or int(value) < 0 for value in pair):
                raise WorkflowConfigError(
                    "diagnostics.save_correlation_for must contain [reflector, shot] pairs."
                )
        for name, values in (
            ("diagnostics.debug_tracking_shots", self.debug_tracking_shots),
            ("diagnostics.debug_tracking_reflectors", self.debug_tracking_reflectors),
        ):
            if any(isinstance(value, bool) or int(value) != value or int(value) < 0 for value in values):
                raise WorkflowConfigError(f"{name} must contain non-negative integers.")
        if not isinstance(self.save_tracking_snapshot, (bool, np.bool_)):
            raise WorkflowConfigError("diagnostics.save_tracking_snapshot must be boolean.")
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
        object.__setattr__(self, "tracking_receiver_stack", bool(self.tracking_receiver_stack))
        object.__setattr__(
            self,
            "tracking_raw_refine_radius_samples",
            int(self.tracking_raw_refine_radius_samples),
        )
        for name in (
            "ownership_guard_time",
            "tracking_coarse_soft_width_time",
            "tracking_coarse_soft_weight",
            "tracking_coarse_soft_penalty_cap",
            "tracking_smooth_weight",
            "tracking_gap_penalty",
            "tracking_restart_min_mean_correlation",
            "tracking_restart_max_coarse_deviation_time",
            "bridge_half_width_time",
            "bridge_max_width_time",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "tracking_top_k_peaks", int(self.tracking_top_k_peaks))
        object.__setattr__(
            self,
            "tracking_peak_min_separation_samples",
            int(self.tracking_peak_min_separation_samples),
        )
        object.__setattr__(self, "bridge_enabled", bool(self.bridge_enabled))
        object.__setattr__(self, "tracking_restart_enabled", bool(self.tracking_restart_enabled))
        object.__setattr__(self, "tracking_max_skip_rows", int(self.tracking_max_skip_rows))
        object.__setattr__(self, "tracking_restart_confirm_rows", int(self.tracking_restart_confirm_rows))
        object.__setattr__(self, "tracking_correlation_mode", str(self.tracking_correlation_mode))
        object.__setattr__(self, "tracking_tracker_mode", str(self.tracking_tracker_mode))
        if self.tracking_max_residual_jump_time is not None:
            object.__setattr__(self, "tracking_max_residual_jump_time", float(self.tracking_max_residual_jump_time))
        object.__setattr__(self, "bridge_max_receivers", int(self.bridge_max_receivers))
        if self.bridge_max_distance is not None:
            object.__setattr__(self, "bridge_max_distance", float(self.bridge_max_distance))
        object.__setattr__(
            self,
            "save_correlation_for",
            tuple((int(pair[0]), int(pair[1])) for pair in self.save_correlation_for),
        )
        object.__setattr__(self, "debug_tracking_shots", tuple(int(value) for value in self.debug_tracking_shots))
        object.__setattr__(self, "debug_tracking_reflectors", tuple(int(value) for value in self.debug_tracking_reflectors))
        object.__setattr__(self, "save_tracking_snapshot", bool(self.save_tracking_snapshot))

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

        pairs = diagnostics.get("save_correlation_for", ())
        if pairs is None:
            pairs = ()
        try:
            pairs = tuple((int(pair[0]), int(pair[1])) for pair in pairs)
        except (TypeError, IndexError, ValueError) as exc:
            raise WorkflowConfigError(
                "diagnostics.save_correlation_for must contain [reflector, shot] pairs."
            ) from exc
        def reflector_index(value: Any) -> int:
            if isinstance(value, str) and value.upper().startswith("R"):
                value = value[1:]
            index = int(value)
            return index - 1 if isinstance(value, str) else index
        try:
            debug_shots = tuple(
                int(value) - 1 for value in diagnostics.get("debug_tracking_shots", ())
            )
            debug_reflectors = tuple(
                reflector_index(value)
                for value in diagnostics.get("debug_tracking_reflectors", ())
            )
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError("diagnostics debug shot/reflector values are invalid.") from exc

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
            ownership_guard_time=correlation.get("ownership_guard_time", 0.02),
            epsilon_time=tracking["epsilon_time"],
            tracking_min_correlation=tracking.get("min_correlation"),
            tracking_enhancement_enabled=tracking.get("enhancement_enabled", True),
            tracking_agc_fraction=tracking.get("agc_fraction", 0.25),
            tracking_agc_floor_ratio=tracking.get("agc_floor_ratio", 0.20),
            tracking_receiver_stack=tracking.get("receiver_stack", True),
            tracking_raw_refine_radius_samples=tracking.get(
                "raw_refine_radius_samples", 1
            ),
            tracking_top_k_peaks=tracking.get("top_k_peaks", 4),
            tracking_peak_min_separation_samples=tracking.get(
                "peak_min_separation_samples", 2
            ),
            tracking_coarse_soft_width_time=tracking.get(
                "coarse_soft_width_time",
                correlation.get("envelope_fine_half_width_time", 0.05),
            ),
            tracking_coarse_soft_weight=tracking.get("coarse_soft_weight", 0.05),
            tracking_coarse_soft_penalty_cap=tracking.get(
                "coarse_soft_penalty_cap", 0.25
            ),
            tracking_smooth_weight=tracking.get("smooth_weight", 0.02),
            tracking_max_residual_jump_time=tracking.get("max_residual_jump_time"),
            tracking_max_skip_rows=tracking.get("max_skip_rows", 3),
            tracking_gap_penalty=tracking.get("gap_penalty", 0.05),
            tracking_correlation_mode=tracking.get("correlation_mode", "auto_polarity"),
            tracking_tracker_mode=tracking.get("tracker_mode", "sparse_global"),
            tracking_restart_enabled=tracking.get("restart_enabled", True),
            tracking_restart_confirm_rows=tracking.get("restart_confirm_rows", 3),
            tracking_restart_min_mean_correlation=tracking.get("restart_min_mean_correlation", 0.55),
            tracking_restart_max_coarse_deviation_time=tracking.get("restart_max_coarse_deviation_time", 0.10),
            local_search_half_width_time=tracking.get("local_search_half_width_time", 0.040),
            max_search_half_width_time=tracking.get("max_search_half_width_time", 0.080),
            gap_expand_time=tracking.get("gap_expand_time", 0.010),
            residual_history=tracking.get("residual_history", 4),
            min_peak_margin=tracking.get("min_peak_margin", 0.03),
            prediction_weight=tracking.get("prediction_weight", 0.10),
            max_gap_rows=tracking.get("max_gap_rows", 4),
            relock_confirm_rows=tracking.get("relock_confirm_rows", 3),
            restart_min_correlation=tracking.get("restart_min_correlation", 0.60),
            bridge_enabled=tracking.get("bridge_enabled", True),
            bridge_half_width_time=tracking.get("bridge_half_width_time", 0.05),
            bridge_max_width_time=tracking.get("bridge_max_width_time", 0.08),
            bridge_max_receivers=tracking.get("bridge_max_receivers", 8),
            bridge_max_distance=tracking.get("bridge_max_distance"),
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
            debug_tracking_shots=debug_shots,
            debug_tracking_reflectors=debug_reflectors,
            save_tracking_snapshot=diagnostics.get("save_tracking_snapshot", False),
            log_level=str(logging.get("log_level", "INFO")),
            verbose=bool(logging.get("verbose", False)),
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
