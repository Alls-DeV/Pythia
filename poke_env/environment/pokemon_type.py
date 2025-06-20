"""This module defines the PokemonType class, which represents a Pokemon type.
PokemonTypes are mainly associated with Pokemons and moves.
"""

from __future__ import annotations

from enum import Enum, auto, unique
from typing import Dict, Optional


@unique
class PokemonType(Enum):
    """A Pokemon type

    This enumeration represents pokemon types. Each type is an instance of this class,
    whose name corresponds to the upper case spelling of its english name (ie. FIRE).
    """

    BUG = auto()
    DARK = auto()
    DRAGON = auto()
    ELECTRIC = auto()
    FAIRY = auto()
    FIGHTING = auto()
    FIRE = auto()
    FLYING = auto()
    GHOST = auto()
    GRASS = auto()
    GROUND = auto()
    ICE = auto()
    NORMAL = auto()
    POISON = auto()
    PSYCHIC = auto()
    ROCK = auto()
    STEEL = auto()
    STELLAR = auto()
    WATER = auto()
    THREE_QUESTION_MARKS = auto()

    def __str__(self) -> str:
        return f"{self.name} (pokemon type) object"

    def damage_multiplier(
        self,
        type_1: PokemonType,
        type_2: Optional[PokemonType] = None,
        *,
        type_chart: Dict[str, Dict[str, float]],
    ) -> float:
        """Computes the damage multiplier from this type on a pokemon with types `type_1`
        and, optionally, `type_2`.

        :param type_1: The first type of the target.
        :type type_1: PokemonType
        :param type_2: The second type of the target. Defaults to None.
        :type type_2: PokemonType, optional
        :return: The damage multiplier from this type on a pokemon with types `type_1`
            and, optionally, `type_2`.
        :rtype: float
        """
        if (
            self == PokemonType.THREE_QUESTION_MARKS
            or type_1 == PokemonType.THREE_QUESTION_MARKS
            or type_1 == "STELLAR"
            or type_2 == "STELLAR"
            or self.name == "STELLAR"
        ):
            return 1

        damage_multiplier = type_chart[type_1.name][self.name]
        if type_2 is not None:
            return damage_multiplier * type_chart[type_2.name][self.name]
        return damage_multiplier

    @staticmethod
    def from_name(name: str) -> PokemonType:
        """Returns a pokemon type based on its name.

        :param name: The name of the pokemon type.
        :type name: str
        :return: The corresponding type object.
        :rtype: PokemonType
        """
        if name == "???":
            return PokemonType.THREE_QUESTION_MARKS
        return PokemonType[name.upper()]
