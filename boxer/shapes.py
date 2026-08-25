"""Minimal replacement for EMAN2's EMShape: a small mutable geometric shape
used to annotate the 2D slice views (particle box markers, crosshair lines,
eraser cursor preview).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Shape:
    kind: str                      # "circle", "rect", "line", or "hidden"
    color: Tuple[float, float, float]
    geom: List[float] = field(default_factory=list)
    linewidth: int = 2

    # kind == "circle": geom = [x, y, r]
    # kind == "rect":   geom = [x0, y0, x1, y1]
    # kind == "line":   geom = [x0, y0, x1, y1]
