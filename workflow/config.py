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
