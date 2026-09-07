"""Frozen-mask VFSA reflection traveltime objective for WRTI Step 6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .mask import FixedMaskResult, MisfitError, _readonly, build_fixed_mask


@dataclass(frozen=True)
class MisfitConfig:
    """Explicit Step 6 configuration.

    ``failure_penalty_time`` has no code default and must be supplied by the
    YAML configuration or by the caller.  Zero explicitly disables the
    objective contribution of failed paths while preserving their count and
    mean-diagnostic membership.  ``candidate_boundary_policy`` is retained for
    configuration compatibility, but a valid boundary state is QC only and is
    never converted into a candidate path failure.
    """

    failure_penalty_time: float
    min_correlation: float | None = None
    exclude_search_boundary: bool = False
    use_energy_threshold: bool = False
    min_window_energy: float | None = None
    candidate_boundary_policy: str = "penalty"
    reflector_weights: Sequence[float] | None = None
    fallback_use_in_misfit: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.failure_penalty_time) or self.failure_penalty_time < 0:
            raise MisfitError(
                "failure_penalty_time must be finite and non-negative."
            )
        if self.min_correlation is not None and (
            not np.isfinite(self.min_correlation)
            or not -1.0 <= self.min_correlation <= 1.0
        ):
            raise MisfitError("min_correlation must be in [-1, 1].")
        if not isinstance(self.exclude_search_boundary, (bool, np.bool_)):
            raise MisfitError("exclude_search_boundary must be boolean.")
        if self.use_energy_threshold and self.min_window_energy is None:
            raise MisfitError(
                "min_window_energy is required when use_energy_threshold=True."
            )
        if self.min_window_energy is not None and (
            not np.isfinite(self.min_window_energy) or self.min_window_energy < 0
        ):
            raise MisfitError("min_window_energy must be finite and non-negative.")
        if self.candidate_boundary_policy not in {"penalty", "use_shift"}:
            raise MisfitError(
                "candidate_boundary_policy must be 'penalty' or 'use_shift'."
            )
        if not isinstance(self.fallback_use_in_misfit, (bool, np.bool_)):
            raise MisfitError("fallback_use_in_misfit must be boolean.")
        weights = self.reflector_weights
        if weights is not None:
            values = tuple(float(value) for value in weights)
            if not values or not np.isfinite(values).all() or any(value < 0 for value in values):
                raise MisfitError(
                    "reflector_weights must be a finite non-negative sequence."
                )
            object.__setattr__(self, "reflector_weights", values)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MisfitConfig":
        """Build from the WRTI YAML-shaped mapping.

        The expected layout is ``qc.failure_penalty_time`` and
        ``misfit.reflector_weights``.  Flat mappings are also accepted for
        small scripts and tests.
        """

        root = dict(mapping)
        qc = root.get("qc", {})
        misfit = root.get("misfit", {})
        tracking = root.get("tracking", {})
        if not all(isinstance(value, Mapping) for value in (qc, misfit, tracking)):
            raise MisfitError(
                "qc, tracking, and misfit configuration sections must be mappings."
            )
        penalty = qc.get("failure_penalty_time", root.get("failure_penalty_time"))
        if penalty is None:
            raise MisfitError(
                "failure_penalty_time must be explicitly provided by YAML/config."
            )
        boundary_policy = qc.get(
            "candidate_boundary_policy",
            qc.get("boundary_failure_policy", root.get("candidate_boundary_policy", "penalty")),
        )
        weights = misfit.get(
            "reflector_weights",
            root.get("reflector_weights"),
        )
        return cls(
            failure_penalty_time=float(penalty),
            min_correlation=qc.get(
                "min_correlation",
                tracking.get("min_correlation", root.get("min_correlation")),
            ),
            exclude_search_boundary=qc.get(
                "exclude_search_boundary",
                root.get("exclude_search_boundary", False),
            ),
            use_energy_threshold=qc.get(
                "use_energy_threshold",
                root.get("use_energy_threshold", False),
            ),
            min_window_energy=qc.get(
                "min_window_energy",
                root.get("min_window_energy"),
            ),
            candidate_boundary_policy=boundary_policy,
            reflector_weights=weights,
            fallback_use_in_misfit=misfit.get(
                "fallback_use_in_misfit",
                root.get("fallback_use_in_misfit", False),
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MisfitConfig":
        """Load Step 6 values through ``yaml.safe_load``."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MisfitError(
                "PyYAML is required to load MisfitConfig from YAML."
            ) from exc
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, Mapping):
            raise MisfitError("The YAML root must be a mapping.")
        return cls.from_mapping(loaded)

    def weights_for(self, n_reflector: int) -> np.ndarray:
        if self.reflector_weights is None:
            return np.ones(n_reflector, dtype=float)
        if len(self.reflector_weights) != n_reflector:
            raise MisfitError(
                "reflector_weights length must equal the number of reflectors."
            )
        return np.asarray(self.reflector_weights, dtype=float)


@dataclass(frozen=True)
class MisfitResult:
    """Formal fixed-mask sum objective and tracking/fallback diagnostics."""

    misfit_sum: float
    shift_time: np.ndarray
    tracked_correlation: np.ndarray
    window_energy_obs: np.ndarray
    window_energy_syn: np.ndarray
    boundary_flag: np.ndarray
    tracking_success: np.ndarray
    fixed_mask: np.ndarray
    path_failure_mask: np.ndarray
    data_invalid_mask: np.ndarray
    low_correlation_qc_mask: np.ndarray
    boundary_qc_mask: np.ndarray
    failure_count: int
    data_invalid_count: int = 0
    fallback_local_count: int = 0
    fallback_global_count: int = 0
    fallback_argmax_count: int = 0
    low_correlation_qc_count: int = 0
    boundary_qc_count: int = 0

    def __post_init__(self) -> None:
        fixed_mask = np.asarray(self.fixed_mask, dtype=bool)
        if fixed_mask.ndim != 3:
            raise MisfitError("fixed_mask must have shape [nref, ns, nr].")
        shape = fixed_mask.shape
        arrays = {
            "shift_time": np.asarray(self.shift_time, dtype=float),
            "tracked_correlation": np.asarray(
                self.tracked_correlation, dtype=float
            ),
            "window_energy_obs": np.asarray(self.window_energy_obs, dtype=float),
            "window_energy_syn": np.asarray(self.window_energy_syn, dtype=float),
            "boundary_flag": np.asarray(self.boundary_flag, dtype=bool),
            "tracking_success": np.asarray(self.tracking_success, dtype=bool),
            "path_failure_mask": np.asarray(
                self.path_failure_mask, dtype=bool
            ),
            "data_invalid_mask": np.asarray(self.data_invalid_mask, dtype=bool),
            "low_correlation_qc_mask": np.asarray(
                self.low_correlation_qc_mask, dtype=bool
            ),
            "boundary_qc_mask": np.asarray(self.boundary_qc_mask, dtype=bool),
        }
        for name, value in arrays.items():
            if value.shape != shape:
                raise MisfitError(f"{name} must have shape {shape}.")
        if not np.isfinite(self.misfit_sum) or self.misfit_sum < 0:
            raise MisfitError("misfit_sum must be finite and non-negative.")
        if (
            isinstance(self.failure_count, bool)
            or int(self.failure_count) != self.failure_count
            or self.failure_count < 0
        ):
            raise MisfitError("failure_count must be a non-negative integer.")
        for name, value in (
            ("data_invalid_count", self.data_invalid_count),
            ("fallback_local_count", self.fallback_local_count),
            ("fallback_global_count", self.fallback_global_count),
            ("fallback_argmax_count", self.fallback_argmax_count),
            ("low_correlation_qc_count", self.low_correlation_qc_count),
            ("boundary_qc_count", self.boundary_qc_count),
        ):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise MisfitError(f"{name} must be a non-negative integer.")

        expected_counts = {
            "failure_count": int(np.count_nonzero(arrays["path_failure_mask"])),
            "data_invalid_count": int(
                np.count_nonzero(arrays["data_invalid_mask"])
            ),
            "low_correlation_qc_count": int(
                np.count_nonzero(arrays["low_correlation_qc_mask"])
            ),
            "boundary_qc_count": int(
                np.count_nonzero(arrays["boundary_qc_mask"])
            ),
        }
        for name, expected in expected_counts.items():
            if int(getattr(self, name)) != expected:
                raise MisfitError(
                    f"{name}={getattr(self, name)} does not match its mask "
                    f"count {expected}."
                )
        if np.any(arrays["path_failure_mask"] & ~fixed_mask):
            raise MisfitError("path_failure_mask must be contained in fixed_mask.")
        if np.any(arrays["data_invalid_mask"] & ~fixed_mask):
            raise MisfitError("data_invalid_mask must be contained in fixed_mask.")
        if np.any(arrays["low_correlation_qc_mask"] & ~fixed_mask):
            raise MisfitError(
                "low_correlation_qc_mask must be contained in fixed_mask."
            )
        if np.any(arrays["boundary_qc_mask"] & ~fixed_mask):
            raise MisfitError("boundary_qc_mask must be contained in fixed_mask.")

        object.__setattr__(self, "misfit_sum", float(self.misfit_sum))
        for name, value in arrays.items():
            object.__setattr__(self, name, _readonly(value, value.dtype))
        object.__setattr__(self, "fixed_mask", _readonly(fixed_mask, bool))
        object.__setattr__(self, "failure_count", int(self.failure_count))
        for name in (
            "data_invalid_count",
            "fallback_local_count",
            "fallback_global_count",
            "fallback_argmax_count",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(
            self,
            "low_correlation_qc_count",
            int(self.low_correlation_qc_count),
        )
        object.__setattr__(self, "boundary_qc_count", int(self.boundary_qc_count))

    @property
    def path_failure_count(self) -> int:
        """Number of DP path failures inside the frozen mask."""

        return self.failure_count

    @property
    def n_fixed(self) -> int:
        """Number of frozen objective slots used by this result."""

        return int(np.count_nonzero(self.fixed_mask))

    @property
    def misfit_mean(self) -> float:
        """Mean diagnostic; never the optimization objective."""

        return self.misfit_sum / self.n_fixed

    @property
    def mean_misfit(self) -> float:
        """Explicitly named alias for the mean diagnostic."""

        return self.misfit_mean


def _quality_arrays(
    fixed_mask: FixedMaskResult | np.ndarray,
    shift_time: np.ndarray,
    tracked_correlation: np.ndarray,
    window_energy_obs: np.ndarray,
    window_energy_syn: np.ndarray,
    boundary_flag: np.ndarray,
    tracking_success: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    reference_mask = (
        fixed_mask.fixed_mask
        if isinstance(fixed_mask, FixedMaskResult)
        else np.asarray(fixed_mask, dtype=bool)
    )
    if reference_mask.ndim != 3:
        raise MisfitError("fixed_mask must have shape [nref, ns, nr].")
    shape = reference_mask.shape
    arrays = tuple(
        np.asarray(value, dtype=dtype)
        for value, dtype in (
            (shift_time, float),
            (tracked_correlation, float),
            (window_energy_obs, float),
            (window_energy_syn, float),
            (boundary_flag, bool),
            (tracking_success, bool),
        )
    )
    for name, value in zip(
        (
            "shift_time",
            "tracked_correlation",
            "window_energy_obs",
            "window_energy_syn",
            "boundary_flag",
            "tracking_success",
        ),
        arrays,
    ):
        if value.shape != shape:
            raise MisfitError(f"{name} must have shape {shape}.")
    return np.asarray(reference_mask, dtype=bool), arrays


def compute_vfsa_misfit(
    fixed_mask: FixedMaskResult | np.ndarray,
    *,
    shift_time: np.ndarray,
    tracked_correlation: np.ndarray,
    window_energy_obs: np.ndarray,
    window_energy_syn: np.ndarray,
    boundary_flag: np.ndarray,
    tracking_success: np.ndarray,
    config: MisfitConfig,
    path_failure_mask: np.ndarray | None = None,
    fallback_local_mask: np.ndarray | None = None,
    fallback_global_mask: np.ndarray | None = None,
    fallback_argmax_mask: np.ndarray | None = None,
) -> MisfitResult:
    """Evaluate the fixed-mask sum traveltime-square objective.

    The mask is read once from ``fixed_mask`` and is never altered using
    candidate quality.  ``tracking_success`` explicitly controls whether a
    DP/bridge measurement (or an opted-in fallback) is eligible.  A finite
    fallback with ``tracking_success=False`` receives the failure penalty.
    The returned optimization objective is ``0.5 * sum(fixed_residual**2)``;
    ``N_fixed`` is used only for the optional mean diagnostic.
    """

    if not isinstance(config, MisfitConfig):
        raise MisfitError("config must be a MisfitConfig instance.")
    reference_mask, arrays = _quality_arrays(
        fixed_mask,
        shift_time,
        tracked_correlation,
        window_energy_obs,
        window_energy_syn,
        boundary_flag,
        tracking_success,
    )
    candidate_shift, candidate_corr, energy_obs, energy_syn, boundary, success = arrays
    n_fixed = int(np.count_nonzero(reference_mask))
    if n_fixed == 0:
        raise MisfitError(
            "fixed_mask contains zero data items; cannot compute VFSA misfit."
        )

    weights = config.weights_for(reference_mask.shape[0])
    # ``success`` is explicit measurement eligibility.  The workflow defaults
    # it to DP/bridge success and opts fallback in only by configuration.
    data_invalid = (~success) | ~np.isfinite(candidate_shift)
    data_invalid_on_fixed = reference_mask & data_invalid
    if path_failure_mask is None:
        dp_failure = data_invalid
    else:
        dp_failure = np.asarray(path_failure_mask, dtype=bool)
        if dp_failure.shape != reference_mask.shape:
            raise MisfitError("path_failure_mask must have shape fixed_mask.shape.")
    path_failure_on_fixed = reference_mask & dp_failure

    def _optional_mask(value, name):
        if value is None:
            return np.zeros(reference_mask.shape, dtype=bool)
        result = np.asarray(value, dtype=bool)
        if result.shape != reference_mask.shape:
            raise MisfitError(f"{name} must have shape fixed_mask.shape.")
        return result & reference_mask

    fallback_local = _optional_mask(fallback_local_mask, "fallback_local_mask")
    fallback_global = _optional_mask(fallback_global_mask, "fallback_global_mask")
    fallback_argmax = _optional_mask(fallback_argmax_mask, "fallback_argmax_mask")

    # QC values are reported separately.  A finite DP-selected shift remains
    # part of the objective even when its ZNCC is below threshold or its lag
    # touches the search boundary.
    low_correlation_qc = np.zeros(reference_mask.shape, dtype=bool)
    if config.min_correlation is not None:
        low_correlation_qc = (
            reference_mask
            & ~data_invalid
            & np.isfinite(candidate_corr)
            & (candidate_corr < config.min_correlation)
        )
    boundary_qc = reference_mask & ~data_invalid & boundary

    effective_shift = np.array(candidate_shift, dtype=float, copy=True)
    effective_shift[data_invalid_on_fixed] = config.failure_penalty_time
    weighted_squared = np.where(
        reference_mask,
        weights[:, None, None] * effective_shift * effective_shift,
        0.0,
    )
    misfit_sum = float(np.sum(weighted_squared) / 2.0)

    return MisfitResult(
        misfit_sum=misfit_sum,
        shift_time=effective_shift,
        tracked_correlation=candidate_corr,
        window_energy_obs=energy_obs,
        window_energy_syn=energy_syn,
        boundary_flag=boundary,
        # Expose final data validity to downstream code.  DP path quality is
        # carried separately by path_failure_mask and remains diagnostic-only
        # when a finite fallback shift is available.
        tracking_success=~data_invalid,
        fixed_mask=reference_mask,
        path_failure_mask=path_failure_on_fixed,
        data_invalid_mask=data_invalid_on_fixed,
        low_correlation_qc_mask=low_correlation_qc,
        boundary_qc_mask=boundary_qc,
        failure_count=int(np.count_nonzero(path_failure_on_fixed)),
        data_invalid_count=int(np.count_nonzero(data_invalid_on_fixed)),
        fallback_local_count=int(np.count_nonzero(fallback_local)),
        fallback_global_count=int(np.count_nonzero(fallback_global)),
        fallback_argmax_count=int(np.count_nonzero(fallback_argmax)),
        low_correlation_qc_count=int(np.count_nonzero(low_correlation_qc)),
        boundary_qc_count=int(np.count_nonzero(boundary_qc)),
    )


def build_fixed_mask_from_config(
    trace_valid: np.ndarray,
    window_valid: np.ndarray,
    tracking_success: np.ndarray,
    tracked_correlation: np.ndarray,
    window_energy_obs: np.ndarray,
    window_energy_syn: np.ndarray,
    boundary_flag: np.ndarray,
    *,
    config: MisfitConfig,
) -> FixedMaskResult:
    """Build the immutable reference mask using one config snapshot."""

    if not isinstance(config, MisfitConfig):
        raise MisfitError("config must be a MisfitConfig instance.")
    return build_fixed_mask(
        trace_valid,
        window_valid,
        tracking_success,
        tracked_correlation,
        window_energy_obs,
        window_energy_syn,
        boundary_flag,
        # min_correlation is a QC threshold only; it must not remove a valid
        # reference path from the frozen mask.
        min_correlation=None,
        exclude_search_boundary=config.exclude_search_boundary,
        use_energy_threshold=config.use_energy_threshold,
        min_window_energy=config.min_window_energy,
    )


def compute_misfit(*args, **kwargs) -> MisfitResult:
    """Alias for :func:`compute_vfsa_misfit`."""

    return compute_vfsa_misfit(*args, **kwargs)
