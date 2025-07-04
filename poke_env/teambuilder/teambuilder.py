"""This module defines the Teambuilder abstract class, which represents objects yielding
Pokemon Showdown teams in the context of communicating with Pokemon Showdown.
"""

from abc import ABC, abstractmethod
from typing import List

from poke_env.stats import STATS_TO_IDX
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon


class Teambuilder(ABC):
    """Teambuilder objects allow the generation of teams by Player instances.

    They must implement the yield_team method, which must return a valid
    packed-formatted showdown team every time it is called.

    This format is a custom format described in Pokemon's showdown protocol
    documentation:
    https://github.com/smogon/pokemon-showdown/blob/master/PROTOCOL.md#team-format

    This class also implements a helper function to convert teams from the classical
    showdown team text format into the packed-format.
    """

    @abstractmethod
    def yield_team(self) -> str:
        """Returns a packed-format team."""

    @staticmethod
    def parse_showdown_team(team: str) -> List[TeambuilderPokemon]:
        """Converts a showdown-formatted team string into a list of TeambuilderPokemon
        objects.

        This method can be used when using teams built in the showdown teambuilder.

        :param team: The showdown-format team to convert.
        :type team: str
        :return: The formatted team.
        :rtype: list of TeambuilderPokemon
        """
        current_mon = TeambuilderPokemon()
        current_mon_has_been_added = True
        mons: List[TeambuilderPokemon] = []

        for line in team.split("\n"):
            while line and line.startswith(" "):
                line = line[1:]

            if line == "":
                if not current_mon_has_been_added:
                    mons.append(current_mon)
                current_mon_has_been_added = True
            elif line.startswith("Ability"):
                ability = line.replace("Ability: ", "")
                current_mon.ability = ability.strip()
            elif line.startswith("Level: "):
                level = line.replace("Level: ", "")
                current_mon.level = int(level.strip())
            elif line.startswith("Happiness: "):
                happiness = line.replace("Happiness: ", "")
                current_mon.happiness = int(happiness.strip())
            elif line.startswith("EVs: "):
                evs = line.replace("EVs: ", "")
                evs = evs.split(" / ")
                for ev in evs:
                    ev = ev.split(" ")
                    n = ev[0]
                    stat = ev[1]
                    idx = STATS_TO_IDX[stat.lower()]
                    current_mon.evs[idx] = int(n)
            elif line.startswith("IVs: "):
                ivs = line.replace("IVs: ", "")
                ivs = ivs.split(" / ")
                for iv in ivs:
                    iv = iv.split(" ")
                    n = iv[0]
                    stat = iv[1]
                    idx = STATS_TO_IDX[stat.lower()]
                    current_mon.ivs[idx] = int(n)
            elif line.startswith("- "):
                line = line.replace("- ", "").strip()
                current_mon.moves.append(line)
            elif line.startswith("Shiny"):
                current_mon.shiny = line.strip().endswith("Yes")
            elif line.startswith("Gigantamax"):
                current_mon.gmax = line.strip().endswith("Yes")
            elif line.strip().endswith(" Nature"):
                nature = line.strip().replace(" Nature", "")
                current_mon.nature = nature
            elif line.startswith("Hidden Power: "):
                hp_type = line.replace("Hidden Power: ", "").strip()
                current_mon.hiddenpowertype = hp_type
            elif line.startswith("Tera Type: "):
                tera_type = line.replace("Tera Type: ", "").strip()
                current_mon.tera_type = tera_type
            else:
                current_mon = TeambuilderPokemon()
                current_mon_has_been_added = False
                if "@" in line:
                    mon_info, item = line.split(" @ ")
                    current_mon.item = item.strip()
                else:
                    mon_info = line
                split_mon_info = mon_info.split(" ")

                if split_mon_info[-1] == "(M)":
                    current_mon.gender = "M"
                    split_mon_info.pop()
                if split_mon_info[-1] == "(F)":
                    current_mon.gender = "F"
                    split_mon_info.pop()
                if split_mon_info[-1].endswith(")"):
                    for i, info in enumerate(split_mon_info):
                        if info[0] == "(":
                            current_mon.species = " ".join(split_mon_info[i:])[1:-1]
                            split_mon_info = split_mon_info[:i]
                            break
                    current_mon.nickname = " ".join(split_mon_info)
                current_mon.nickname = " ".join(split_mon_info)
        if not current_mon_has_been_added:
            mons.append(current_mon)
        return mons

    @staticmethod
    def join_team(team: List[TeambuilderPokemon]) -> str:
        """Converts a list of TeambuilderPokemon objects into the corresponding packed
        showdown team format.

        :param team: The list of TeambuilderPokemon objects that form the team.
        :type team: list of TeambuilderPokemon
        :return: The formatted team string.
        :rtype: str"""
        return "]".join([mon.formatted for mon in team])
