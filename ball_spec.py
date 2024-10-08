"""
FIXME - Need Module docstring
"""
from dataclasses import dataclass
from typing import Final


class BallSpecDefaults:
    """
    FIXME
    """
    MASS: Final[float] = 1.0  # kg
    RADIUS: Final[float] = 1.0  # meters
    SPHERE_DRAG_COEFFICIENT: Final[float] = 0.47

@dataclass
class BallSpec:
    """
    FIXME
    """
    mass: float = BallSpecDefaults.MASS
    radius: float = BallSpecDefaults.RADIUS
    drag_coefficient: float = BallSpecDefaults.SPHERE_DRAG_COEFFICIENT
