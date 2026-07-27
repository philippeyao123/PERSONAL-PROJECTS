"""Experiment identity, seeds and metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-serializable content."""
    payload = asdict(value) if is_dataclass(value) else value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def set_global_seed(seed: int) -> np.random.Generator:
    """Control Python and NumPy random sources and return the preferred generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def experiment_metadata(config: Any, data_hash: str) -> Dict[str, Any]:
    """Capture enough metadata to reproduce a result."""
    return {
        "config_hash": stable_hash(config),
        "data_hash": data_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
