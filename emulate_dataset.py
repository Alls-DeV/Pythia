import asyncio
import string
from typing import Any, Dict, List, Optional

import numpy as np
import orjson
from datasets import Dataset  # Assuming Dataset is imported if type hinted
from datasets import load_from_disk
from tqdm import tqdm

from common import PNUMBER1
from poke_env.data import GenData  # For gen in Move object
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status
from poke_env.player.llm_player import LLMPlayer
from poke_env.player.local_simulation import LocalSim
from poke_env.player.player import BattleOrder  # Added for type hinting
from poke_env.player.pokechamp import Pokechamp
from poke_env.player.pythia import Pythia
from poke_env.player.pythia_prompt import prompt_translate
from poke_env.player.translate import (
    recursive_nick_removal,
)  # Not strictly needed for this modification; extract_elo_from_file,; get_player_team,
from poke_env.ps_client.account_configuration import AccountConfiguration


def get_player_team(
    battle_turns_text: List[List[str]], player_id: str
) -> List[Dict[str, Any]]:
    # Simplified placeholder based on common log structure
    species_list = []

    # Look for initial |poke| lines for the specified player
    if battle_turns_text and battle_turns_text[0]:
        for line_str in battle_turns_text[0]:
            line_parts = line_str.split("|")
            if (
                len(line_parts) > 3
                and line_parts[1] == "poke"
                and line_parts[2] == player_id
            ):
                # Extract species name, e.g., "Pikachu, L80, M" -> "Pikachu"
                species_name = line_parts[3].split(",")[0].strip()
                if species_name not in species_list:  # Avoid duplicates if log is messy
                    species_list.append(species_name)

    # Initialize move tracking
    moves = {mon.split(",")[0]: [] for mon in species_list}

    for turn in battle_turns_text:
        for msg in turn:
            m = msg.replace("\n", "").split("|")
            if len(m) >= 4 and m[1] == "move" and f"{player_id}a" in m[2]:
                mon = m[2].split(":")[1].strip()
                move = m[3]
                if move not in moves[mon]:
                    moves[mon].append(move)

    # Normalize species and move names, and structure output
    team = []
    for mon, mon_moves in moves.items():
        normalized_species = (
            mon.replace(" ", "")
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
        )
        normalized_moves = [
            move.replace(" ", "")
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
            for move in mon_moves
        ]
        team.append(
            {
                "species": normalized_species,
                "moves": normalized_moves,
                # "species": mon,
                # "moves": mon_moves,
                "item": None,
                "ability": None,
                "nature": None,
                "tera": None,
                "evs": None,
            }
        )

    return team


# --- End of helper functions from original file ---


def get_player_action_details_from_log(
    turn_messages: List[List[str]], player_id_str: str, battle_gen: int
) -> Optional[Dict[str, Any]]:
    action_type = None
    target_name_raw = None  # Store raw name for move resolution
    normalized_target_name = None
    is_dynamax = False
    is_terastallize = False
    pokemon_that_acted = None

    for m_parts in turn_messages:
        if len(m_parts) < 3:
            continue

        # Identify the Pokémon that acted, important for context like Tera type
        if m_parts[1] == "switch" and m_parts[2].startswith(f"{player_id_str}a:"):
            pokemon_that_acted_name_raw = m_parts[2].split(":")[1].strip()
            # We need the actual species of the pokemon being switched TO.
            # The log format is |switch|p1a: OldPokemon|NewPokemon, Lvl|hp
            # Here target_name_raw is the NewPokemon species.
            if len(m_parts) > 3:  # New pokemon species is in m_parts[3]
                target_name_raw = m_parts[3].split(",")[0].strip()
            else:  # Fallback if format is slightly different
                target_name_raw = pokemon_that_acted_name_raw

            action_type = "switch"
            normalized_target_name = (
                target_name_raw.lower().replace(" ", "").replace("-", "")
            )
            # Switches don't inherently involve dyna/tera in the same action command
            break  # Found switch action for the player

        elif m_parts[1] == "move" and m_parts[2].startswith(f"{player_id_str}a:"):
            pokemon_that_acted_name_raw = m_parts[2].split(":")[1].strip()
            action_type = "move"
            target_name_raw = m_parts[
                3
            ]  # Raw move name, e.g., "Max Airstream" or "Thunderbolt"
            normalized_target_name = (
                target_name_raw.lower().replace(" ", "").replace("-", "")
            )

            # Check for dynamax/terastallize in the same line or subsequent related lines
            # Showdown logs might append [from] Dynamax or [from] Terastallization
            if len(m_parts) > 4 and isinstance(
                m_parts[-1], str
            ):  # Check last part for details
                if "[from] Dynamax" in m_parts[-1]:
                    is_dynamax = True
                if "[from] Terastallization" in m_parts[-1]:
                    is_terastallize = True

            # Dynamax can also be inferred from move names
            if normalized_target_name.startswith(
                "max"
            ) or normalized_target_name.startswith("gmax"):
                is_dynamax = True

            # Check for |tera| message for the acting Pokémon
            for (
                sub_m_parts
            ) in turn_messages:  # Re-scan for tera, as it's a separate message
                if (
                    len(sub_m_parts) > 3
                    and sub_m_parts[1] == "tera"
                    and sub_m_parts[2].startswith(f"{player_id_str}a:")
                    and pokemon_that_acted_name_raw in sub_m_parts[2]
                ):  # Ensure it's the same Pokemon
                    is_terastallize = True
                    break
            break  # Found move action for the player

    if action_type:
        return {
            "type": action_type,
            "target_raw": target_name_raw,  # Keep raw for Move object creation
            "target_normalized": normalized_target_name,
            "dynamax": is_dynamax,
            "terastallize": is_terastallize,
        }
    return None


def compare_llm_and_human_actions(
    llm_order: BattleOrder, human_action_details: Dict[str, Any], battle_gen: int
) -> bool:
    if not human_action_details:
        return False

    llm_action_type = "move" if isinstance(llm_order.order, Move) else "switch"
    if llm_action_type != human_action_details["type"]:
        return False

    llm_is_dynamax = llm_order.dynamax
    llm_is_terastallize = llm_order.terastallize

    if llm_is_dynamax != human_action_details["dynamax"]:
        return False
    if llm_is_terastallize != human_action_details["terastallize"]:
        return False

    if llm_action_type == "move":
        llm_move_id = llm_order.order.id
        human_move_id_normalized = human_action_details["target_normalized"]

        if llm_is_dynamax:  # Both are dynamaxing (flags matched already)
            # LLM provides base move. Human log provides Max Move name.
            # We need to see if LLM's base move, when dynamaxed, becomes human's Max Move.
            try:
                llm_base_move_obj = Move(llm_move_id, gen=battle_gen)
                # Dynamaxed version of the LLM's chosen base move
                llm_expected_max_move_id = llm_base_move_obj.dynamaxed.id

                # The human_move_id_normalized is already the ID of the max move from the log
                if llm_expected_max_move_id == human_move_id_normalized:
                    return True  # Max moves match
                else:
                    # Edge case: Max Guard. If LLM chose a status move and dynamaxed, it's Max Guard.
                    if (
                        llm_base_move_obj.category == MoveCategory.STATUS
                        and human_move_id_normalized == "maxguard"
                    ):
                        return True
                    return False
            except Exception:  # Failed to resolve dynamaxed move, treat as mismatch
                return False
        else:  # Not dynamaxing, direct move ID comparison
            if llm_move_id == human_move_id_normalized:
                return True
            return False
    elif llm_action_type == "switch":
        llm_switch_species_id = (
            llm_order.order.species.lower().replace(" ", "").replace("-", "")
        )
        human_switch_species_id = human_action_details["target_normalized"]
        if llm_switch_species_id == human_switch_species_id:
            return True
        return False

    return False


def enrich_team(team: List[Dict[str, Any]]) -> None:
    file = f"poke_env/data/static/gen9/ou/sets_1000.json"
    with open(file, "r") as f:
        pokedex = orjson.loads(f.read())

    for mon in team:
        species = mon["species"].lower().replace(" ", "").replace("-", "")

        if species not in pokedex:
            continue

        data = pokedex[species]

        # Fill missing moves (up to 4 total)
        try:
            possible_moves = [
                move["name"].lower().replace(" ", "").replace("-", "")
                for move in data.get("moves", [])
            ]
        except:
            possible_moves = []

        while len(mon["moves"]) < 4 and possible_moves:
            move_unseen = possible_moves.pop(0)
            if move_unseen not in mon["moves"]:
                mon["moves"].append(move_unseen)

        # Fill missing fields from pokedex
        if mon.get("item") is None and data.get("items"):
            mon["item"] = (
                data["items"][0]["name"].lower().replace(" ", "").replace("-", "")
            )

        if mon.get("ability") is None and data.get("abilities"):
            mon["ability"] = (
                data["abilities"][0]["name"].lower().replace(" ", "").replace("-", "")
            )

        if mon.get("nature") is None and data.get("spreads"):
            mon["nature"] = data["spreads"][0]["nature"]

        if mon.get("evs") is None and data.get("spreads"):
            mon["evs"] = data["spreads"][0].get("stats", {})

        if mon.get("tera") is None and data.get("tera"):
            mon["tera"] = data["tera"][0]["name"]


def convert_team_to_showdown_format(team_data: List[Dict[str, Any]]) -> str:
    """
    Converts a list of Pokémon data dictionaries into a Showdown text-formatted team string.
    """
    showdown_team_str = ""
    for mon_data in team_data:
        # Species and Item
        species_line = mon_data.get("species", "UnknownSpecies")
        if mon_data.get("nickname") and mon_data["nickname"] != mon_data.get("species"):
            species_line = f"{mon_data['nickname']} ({mon_data.get('species')})"

        item = mon_data.get("item")
        if item:
            species_line += f" @ {item}"
        showdown_team_str += species_line + "\n"

        # Ability
        ability = mon_data.get("ability")
        if ability:
            showdown_team_str += f"Ability: {ability}\n"

        # Level (default to 100 if not specified, common in OU)
        level = mon_data.get("level", 100)
        if level != 100:  # Only add if not default for competitive
            showdown_team_str += f"Level: {level}\n"

        # EVs
        evs = mon_data.get("evs")
        if evs:
            ev_parts = []
            # Order is important for Showdown: HP, Atk, Def, SpA, SpD, Spe
            stat_map_to_showdown = {
                "hp": "HP",
                "atk": "Atk",
                "def": "Def",
                "spa": "SpA",
                "spd": "SpD",
                "spe": "Spe",
            }
            # Ensure standard order
            for stat_key_internal, stat_name_showdown in stat_map_to_showdown.items():
                if stat_key_internal in evs and evs[stat_key_internal] > 0:
                    ev_parts.append(f"{evs[stat_key_internal]} {stat_name_showdown}")
            if ev_parts:
                showdown_team_str += f"EVs: {' / '.join(ev_parts)}\n"

        # Tera Type
        tera_type = mon_data.get("tera")
        if tera_type:
            showdown_team_str += f"Tera Type: {tera_type}\n"

        # Nature
        nature = mon_data.get("nature")
        if nature:
            showdown_team_str += f"{nature} Nature\n"

        # IVs (assuming default 31 if not specified, common in OU)
        # If you have IV data, you'd add it similarly to EVs:
        # ivs = mon_data.get("ivs")
        # if ivs:
        #     iv_parts = []
        #     for stat_key_internal, stat_name_showdown in stat_map_to_showdown.items():
        #          # Check if IV is not default 31
        #         if stat_key_internal in ivs and ivs[stat_key_internal] != 31:
        #             iv_parts.append(f"{ivs[stat_key_internal]} {stat_name_showdown}")
        #     if iv_parts:
        #         showdown_team_str += f"IVs: {' / '.join(iv_parts)}\n"

        # Moves
        moves = mon_data.get("moves", [])
        for move in moves:
            # If moves in enrich_team are dicts like {"name": "Surf"}, extract name
            if isinstance(move, dict) and "name" in move:
                showdown_team_str += f"- {move['name']}\n"
            elif isinstance(move, str):
                showdown_team_str += f"- {move}\n"

        showdown_team_str += "\n"  # Blank line between Pokémon

    return showdown_team_str.strip()


miaomioa = set()


async def emulate_battle(
    battle_text: str,
    format_str: str,  # renamed to avoid conflict with built-in format
    battle_id: str,
    # json_text: List, # Unused in this modification
    gen_num: int,  # Renamed for clarity
    p1_side: bool,
    prompt_translate=prompt_translate,
) -> Optional[tuple[int, int]]:

    llm_matching_actions = 0
    total_player_actions = 0

    battle_lines = battle_text.splitlines()
    battle_lines = [t.replace("type: null", "typenull") for t in battle_lines]
    battle_lines = recursive_nick_removal(battle_lines)

    battle_turns_text, p1_username, p2_username, winner_username = [], "", "", ""
    current_turn_messages = []
    for line in battle_lines:
        if any(
            skip in line.lower() for skip in ["zoroark", "|c|", "|raw|"]
        ):  # Skip noisy lines
            continue
        if "player|p1" in line and not p1_username:
            p1_username = line.split("|")[3]
        elif "player|p2" in line and not p2_username:
            p2_username = line.split("|")[3]
        elif "|win|" in line:
            winner_username = line.split("|")[2].rstrip()

        current_turn_messages.append(line)
        # Split turns more reliably: on new |turn| or end of game signals like |win|
        if (
            line.startswith("|turn|")
            or line.startswith("|win|")
            or line.startswith("|tie|")
        ):
            if len(current_turn_messages) > 1 and line.startswith(
                "|turn|"
            ):  # if it's a new turn, the last line is part of it
                battle_turns_text.append(current_turn_messages)
                current_turn_messages = [line]  # Start new turn with this line
            elif line.startswith("|win|") or line.startswith("|tie|"):  # End of game
                battle_turns_text.append(current_turn_messages)
                current_turn_messages = []

    if (
        not battle_turns_text or len(battle_turns_text) < 1
    ):  # Need at least one turn block
        print(f"Battle {battle_id} has too few turns or parsing error.")
        return None
    if (
        winner_username not in [p1_username, p2_username] and winner_username != ""
    ):  # Allow empty if no win/tie yet
        print(f"Battle {battle_id} winner mismatch or not found.")
        return None

    player_id_str, opponent_id_str = ("p1", "p2") if p1_side else ("p2", "p1")
    player_username = p1_username if p1_side else p2_username

    # Initial team setup (simplified for this task, LLMPlayer usually gets team via its init)
    # Create player and battle simulation
    # The LLMPlayer needs a team string if the format requires it.
    # For random battles, team is not provided by user.
    # We are emulating, so we don't give it a pre-set team string.
    # It will discover its team through game messages.
    llm_player = Pythia(
        battle_format=format_str,
        account_configuration=AccountConfiguration(player_username, ""),
        prompt_translate=prompt_translate,
        save_replays=False,  # Ensure this is False for emulation,
    )

    llm_player._dynamax_disable = gen_num != 8  # Disable dyna if not gen 8

    player_team = get_player_team(battle_turns_text, player_id_str)
    enrich_team(player_team)
    initial_battle_for_sim = await llm_player._create_battle(
        f">battle-{format_str}-{battle_id}".split("-")
    )

    sim = LocalSim(
        initial_battle_for_sim,  # Pass the battle object
        llm_player.move_effect,
        llm_player.pokemon_move_dict,
        llm_player.ability_effect,
        llm_player.pokemon_ability_dict,
        llm_player.item_effect,
        llm_player.pokemon_item_dict,
        llm_player.gen,
        llm_player._dynamax_disable,
        format=llm_player.format,
        prompt_translate=llm_player.prompt_translate,
    )

    flag = True
    # Process each turn
    for turn_idx, turn_message_list in enumerate(battle_turns_text):
        # print(f"\n--- Emulating Turn {turn_idx + 1} for battle {battle_id} ---")
        split_messages_for_turn = [
            m.replace("\n", "").split("|") for m in turn_message_list
        ]

        human_action_details = None
        if turn_idx > 0:  # Skip teampreview/initial setup for action comparison
            human_action_details = get_player_action_details_from_log(
                split_messages_for_turn, player_id_str, sim.gen.gen
            )

        # Before applying this turn's messages to sim, if it's a decision point for player:
        # A decision point is when sim.battle.request is active for the player.
        # This typically happens after a |request| message for the player has been processed.
        # The human_action_details is for *this* turn. LLM decision should be based on *previous* turn's end state.

        # The LLM choose_move should be called *before* any message of the current turn is processed by sim,
        # if the previous turn ended with a request for the player.
        # Let's refine the logic:
        # 1. At start of turn `N` (after turn `N-1` fully processed):
        #    Check `sim.battle.request`. If it's active for `player_id_str`:
        #    a. Get `human_action_details` for turn `N` from `split_messages_for_turn_N`.
        #    b. Get `llm_action` using `llm_player.choose_move(sim.battle)`. `sim.battle` is end of `N-1`.
        #    c. Compare.
        # 2. Process all messages of turn `N` using `sim._handle_battle_message`.

        made_decision_this_turn = False
        if (
            sim.battle.rqid is not None and sim.battle.active_pokemon
        ):  # A request is pending from previous turn
            # This check needs to be more robust; rqid alone isn't enough.
            # Check if the request is for the player and if a choice is possible.
            # A simpler trigger: if human_action_details is found for *this* turn, it implies a decision was made.

            if flag:
                for real_pokemon in sim.battle.available_switches:
                    for inferred_pokemon in player_team:
                        if inferred_pokemon["species"] == real_pokemon.species:
                            real_pokemon.ability = inferred_pokemon["ability"]
                            real_pokemon.item = inferred_pokemon["item"]
                            real_pokemon._terastallized_type = PokemonType.from_name(
                                inferred_pokemon["tera"]
                            )
                            for move in inferred_pokemon["moves"]:
                                real_pokemon._add_move(move)

                all_switches_backup = sim.battle.available_switches
                flag = False
            sim.battle._available_moves = [
                Move(move, gen=llm_player.gen.gen)
                for move in sim.battle.active_pokemon.moves
            ]
            real_available_switches = []
            for switch in all_switches_backup:
                if switch.active == False and switch.status != Status.FNT:
                    real_available_switches.append(switch)
            sim.battle._available_switches = real_available_switches
            # ----------------------------------------

            if human_action_details:
                # print(f"Player {player_id_str} (human) action for turn {turn_idx+1}: {human_action_details}")

                # Ensure the battle state is ready for a choice (active Pokemon exists, etc.)
                if sim.battle.active_pokemon and (
                    not sim.battle.active_pokemon.fainted or sim.battle.force_switch
                ):
                    total_player_actions += 1
                    made_decision_this_turn = True
                    # print(f"Requesting LLM choice for turn {turn_idx+1}. Active: {sim.battle.active_pokemon.species if sim.battle.active_pokemon else 'None'}")

                    try:
                        llm_action_order = llm_player.choose_move(sim.battle)
                        # print(f"LLM action for turn {turn_idx+1}: {llm_action_order.message if llm_action_order else 'None'}")

                        # Open file in append mode to accumulate all messages
                        with open(f"llm_log/{PNUMBER1}/emulate_stats.log", "a") as f:
                            if llm_action_order and compare_llm_and_human_actions(
                                llm_action_order, human_action_details, sim.gen.gen
                            ):
                                llm_matching_actions += 1
                                f.write(
                                    f"MATCH! Human: {human_action_details['type']} {human_action_details['target_raw']}, LLM: {llm_action_order.message}\n"
                                )
                            else:
                                f.write(
                                    f"NO MATCH. Human: {human_action_details['type']} {human_action_details['target_raw']}, LLM: {llm_action_order.message if llm_action_order else 'None'}\n"
                                )
                    except Exception as e:
                        with open(f"llm_log/{PNUMBER1}/emulate_stats.log", "a") as f:
                            f.write(
                                f"Error during LLM choose_move or comparison for battle {battle_id} turn {turn_idx+1}: {e}\n"
                            )
                # else:
                # print(f"Skipping LLM choice: No active/fainted Pokemon or no force switch. Active: {sim.battle.active_pokemon}")

        # Process all messages of the current turn to update sim.battle for the *next* iteration
        for msg_parts in split_messages_for_turn:
            if len(msg_parts) >= 2 and msg_parts[1]:
                try:
                    # print(f"  Processing msg: {'|'.join(msg_parts)}")
                    sim._handle_battle_message(msg_parts)
                except (KeyError, ValueError, NotImplementedError) as e:
                    # print(f"    Skipping message due to parsing error: {e} ({'|'.join(msg_parts)})")
                    continue
                except Exception as e:
                    # print(f"    Unexpected error processing message: {e} ({'|'.join(msg_parts)})")
                    continue

        if sim.battle.finished:
            # print(f"Battle {battle_id} finished in sim at turn {turn_idx+1}.")
            break

    # print(f"Battle {battle_id} emulation finished. Matches: {llm_matching_actions}/{total_player_actions}")
    if (
        total_player_actions == 0 and not made_decision_this_turn and turn_idx > 0
    ):  # if no actions were ever logged by player but battle progressed
        # print(f"Warning: No player actions were logged for comparison in battle {battle_id}, despite {turn_idx+1} turns.")
        pass

    return total_player_actions, llm_matching_actions


async def main():
    dataset: Dataset = load_from_disk("mini_dataset")
    dataset = dataset.select(range(min(7, len(dataset))))

    gen = 9  # This should ideally come from the dataset's format if it varies

    grand_total_player_actions = 0
    grand_total_llm_matches = 0
    battles_processed = 0
    battles_with_errors_or_skipped = 0

    grand_total_player_actions = 0
    grand_total_llm_matches = 0
    battles_processed = 0
    battles_with_errors_or_skipped = 0

    for i, battle_data in tqdm(
        enumerate(dataset), total=len(dataset), desc="Processing battles"
    ):
        battle_id = battle_data["battle_id"]
        battle_text = battle_data["text"]
        # with open("mini_dataset_battles/0", "r") as f:
        #     battle_text = f.read()

        battle_gamemode = battle_data["gamemode"]
        # Determine gen from format string if possible, e.g. "gen9randombattle"
        current_gen = gen  # Default
        if "gen" in battle_gamemode:
            try:
                gen_char_index = battle_gamemode.find("gen") + 3
                current_gen = int(battle_gamemode[gen_char_index])
            except ValueError:
                print(
                    f"Could not parse gen from format: {battle_gamemode}, defaulting to {gen}"
                )

        print(
            f"\nProcessing battle {i+1}: {battle_id} (Format: {battle_gamemode}, Gen: {current_gen})"
        )

        try:
            # Emulate for P1's perspective
            # In a real scenario, you might want to decide which player's perspective to take,
            # or do both if the log contains enough info for both sides' teams.
            # For now, let's assume we are evaluating based on P1 if P1 data is clear.
            # The original script had a p1_side flag.

            # A simple way to decide perspective: if the dataset has an "elo" field for P1/P2,
            # we can pick the higher elo player, or just always P1.
            # For this example, sticking to p1_side=True as in the original thought process.
            result = await emulate_battle(
                battle_text, battle_gamemode, battle_id, current_gen, p1_side=True
            )

            if result:
                total_actions, matching_actions = result
                print(
                    f"Battle {battle_id}: Matched {matching_actions} / {total_actions} actions."
                )
                grand_total_player_actions += total_actions
                grand_total_llm_matches += matching_actions
                battles_processed += 1
            else:
                print(
                    f"Skipped battle {battle_id} due to parsing issues or insufficient data."
                )
                battles_with_errors_or_skipped += 1

        except Exception as e:
            print(f"Critical error processing battle {battle_id}: {e}")
            battles_with_errors_or_skipped += 1
            # import traceback
            # traceback.print_exc()

    print("\n--- Emulation Summary ---")
    print(f"Total battles processed: {battles_processed}")
    print(f"Total battles skipped or with errors: {battles_with_errors_or_skipped}")
    if grand_total_player_actions > 0:
        match_rate = (grand_total_llm_matches / grand_total_player_actions) * 100
        print(
            f"Overall LLM action match rate: {match_rate:.2f}% ({grand_total_llm_matches} / {grand_total_player_actions})"
        )
    else:
        print(
            "No player actions were available for comparison across processed battles."
        )
    for species in miaomioa:
        print(f'"{species}": "",')


if __name__ == "__main__":
    asyncio.run(main())
