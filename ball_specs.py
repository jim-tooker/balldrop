"""
FIXME - Need Module docstring
"""
from dataclasses import dataclass
from typing import Final


class BallSpecsDefaults:
    """
    FIXME
    """
    MASS: Final[float] = 1.0  # kg
    RADIUS: Final[float] = 1.0  # meters
    SPHERE_DRAG_COEFFICIENT: Final[float] = 0.47

@dataclass
class BallSpecs:
    """
    FIXME
    """
    mass: float = BallSpecsDefaults.MASS
    radius: float = BallSpecsDefaults.RADIUS
    drag_coefficient: float = BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT
