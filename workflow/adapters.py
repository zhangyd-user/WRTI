"""Small adapters for external forward-model data containers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class AdapterError(ValueError):
    """Raised when external seismic data cannot be normalized."""


def _to_numpy(value) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result, dtype=float)


def _extract_adfwi_field(output, field: str):
    if isinstance(output, Mapping):
        if field not in output:
            raise AdapterError(f"ADFWI mapping does not contain field '{field}'.")
        return output[field]
    data = getattr(output, "data", None)
    if isinstance(data, Mapping):
        if field not in data:
            raise AdapterError(f"ADFWI output.data does not contain field '{field}'.")
        return data[field]
    parser = getattr(output, "parse_acoustic_data", None)
    if callable(parser):
        parsed = parser(normalize=False)
        fields = {"p": 0, "u": 1, "w": 2}
        if field not in fields:
            raise AdapterError(
                "parse_acoustic_data output supports fields 'p', 'u', and 'w'."
            )
        return parsed[fields[field]]
    return output


def adfwi_to_wrti(
    output,
    *,
    field: str = "p",
    layout: str = "shot_time_receiver",
) -> np.ndarray:
    """Convert ADFWI waveform output to WRTI ``[shot, receiver, time]``.

    ADFWI ``SeismicData`` acoustic fields are conventionally
    ``[shot, time, receiver]``.  The adapter does not import ADFWI or invoke
    its forward solver; it only extracts the selected field and transposes it.
    Set ``layout='shot_receiver_time'`` when an external caller already uses
    WRTI order.
    """

    values = _to_numpy(_extract_adfwi_field(output, field))
    if values.ndim != 3:
        raise AdapterError(
            "ADFWI waveform output must be three-dimensional: [shot,time,receiver] "
            "or [shot,receiver,time]."
        )
    if layout == "shot_time_receiver":
        values = values.transpose(0, 2, 1)
    elif layout != "shot_receiver_time":
        raise AdapterError(
            "layout must be 'shot_time_receiver' or 'shot_receiver_time'."
        )
    if not np.isfinite(values).all():
        raise AdapterError("converted waveform data must contain only finite values.")
    return np.array(values, dtype=float, copy=True)


def from_adfwi(output, **kwargs) -> np.ndarray:
    """Alias for :func:`adfwi_to_wrti`."""

    return adfwi_to_wrti(output, **kwargs)
