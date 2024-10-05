"""
FIXME - Need Module docstring
"""
from dataclasses import dataclass
from typing import Final


class EnvironmentDefaults:
    """
    FIXME
    """
    EARTH_GRAVITY: Final[float] = 9.80665  # m/s²
    EARTH_AIR_DENSITY: Final[float] = 1.225  # kg/m³
    DEFAULT_COR: Final[float] = 0.8  # Coefficient of Restitution

@dataclass
class Environment:
    """
    FIXME
    """
    gravity: float = EnvironmentDefaults.EARTH_GRAVITY
    air_density: float = EnvironmentDefaults.EARTH_AIR_DENSITY
    cor: float = EnvironmentDefaults.DEFAULT_COR
