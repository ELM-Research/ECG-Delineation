"""Write a JS global assignment so the demo works on file:// (no fetch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def round_array(arr: np.ndarray, ndigits: int = 4) -> list:
    return np.asarray(arr, dtype=np.float64).round(ndigits).tolist()


def write_js(path: Path, var_name: str, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"window.{var_name} = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n"
    )
    path.write_text(text, encoding="utf-8")
