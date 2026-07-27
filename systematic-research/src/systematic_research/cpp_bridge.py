"""Optional bridge to the separate qf-rates pybind11 module."""

from __future__ import annotations

from typing import Any

from systematic_research.exceptions import ResearchError


def load_qf_rates() -> Any:
    try:
        import qf_rates_python
    except ImportError as error:
        raise ResearchError(
            "qf-rates bindings are optional; build qf-rates with -DQF_BUILD_PYTHON=ON"
        ) from error
    return qf_rates_python
