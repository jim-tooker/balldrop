"""
This module defines the environment in which the ball drop simulation occurs.

It includes the `EnvironmentDefaults` class, which holds default values for 
environmental properties such as gravity, air density, and the coefficient of restitution (COR),
as well as the `Environment` data class that can be used to specify these properties 
for individual simulations.
"""

from dataclasses import dataclass
from typing import Final

__author__ = 'Jim Tooker'


class EnvironmentDefaults:
    """
    Default values for environmental properties used in the ball drop simulation.

    Attributes:
        EARTH_GRAVITY (float): The gravitational acceleration on Earth in m/s².
        EARTH_AIR_DENSITY (float): The air density at sea level on Earth in kg/m³.
        DEFAULT_COR (float): The default coefficient of restitution (COR) for collisions.
    """
    EARTH_GRAVITY: Final[float] = 9.80665  # m/s², standard gravity on Earth
    EARTH_AIR_DENSITY: Final[float] = 1.225  # kg/m³, standard air density at sea level
    DEFAULT_COR: Final[float] = 0.8  # Coefficient of restitution for bounces


@dataclass
class Environment:
    """
    Environmental conditions for the ball drop simulation.

    Attributes:
        gravity (float): The gravitational acceleration to be applied. Defaults to `EnvironmentDefaults.EARTH_GRAVITY`.
        air_density (float): The air density for calculating drag forces.
            Defaults to `EnvironmentDefaults.EARTH_AIR_DENSITY`.
        cor (float): The coefficient of restitution for collisions, used to simulate energy loss during bounces. 
                     Defaults to `EnvironmentDefaults.DEFAULT_COR`.
    """
    gravity: float = EnvironmentDefaults.EARTH_GRAVITY
    air_density: float = EnvironmentDefaults.EARTH_AIR_DENSITY
    cor: float = EnvironmentDefaults.DEFAULT_COR
