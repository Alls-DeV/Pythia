import asyncio
import string
from typing import List

import numpy as np
from datasets import load_from_disk

from poke_env.player.prompts import prompt_translate as pt
from poke_env.player.translate import (
    extract_elo_from_file,
    get_player_team,
    recursive_nick_removal,
)


async def f(
    battle_text: str,
    format: str,
    battle_id: str,
    json_text: List,
    gen: int,
    prompt_translate=pt,
    use_winner=True,
):
    battle_text = battle_text.splitlines()

    # Extract Elo ratings and preprocess battle text
    player_elo, opponent_elo = extract_elo_from_file(battle_text)
    battle_text = [t.replace("type: null", "typenull") for t in battle_text]
    battle_text = recursive_nick_removal(battle_text)

    # Parse battle into turns and extract player information
    battle_turns_text, p1_username, p2_username, winner_username = [], "", "", ""
    turn_text = []
    for line in battle_text:
        if any(skip in line.lower() for skip in ["zoroark", "|c|", "|raw|"]):
            continue
        if "player|p1" in line and not p1_username:
            p1_username = line.split("|")[3]
        elif "player|p2" in line and not p2_username:
            p2_username = line.split("|")[3]
        elif "|win|" in line:
            winner_username = line.split("|")[2].rstrip()

        turn_text.append(line)
        if "|turn|" in line or "|faint|" in line:
            battle_turns_text.append(turn_text)
            turn_text = []

    # Validate battle data
    if (
        len(battle_turns_text) < 2
        or len(battle_turns_text) > 80
        or winner_username not in [p1_username, p2_username]
    ):
        return False

    # Set player IDs and determine winner
    if use_winner and winner_username != p1_username:
        p1_username, p2_username = p2_username, p1_username
        player_id, opponent_id = "p2", "p1"
    else:
        player_id, opponent_id = (
            ("p2", "p1") if np.random.random() < 0.5 else ("p1", "p2")
        )
    winner = "player" if winner_username == p1_username else "opponent"

    # Extract player team and moves
    team_player, team_mons = get_player_team(battle_turns_text, player_id)
    moves = {mon.split(",")[0]: [] for mon in team_mons}
    for turn in battle_turns_text:
        for m in [msg.replace("\n", "").split("|") for msg in turn]:
            if len(m) >= 4 and "move" in m[1] and f"{player_id}a" in m[2]:
                mon = m[2].split(":")[1].strip()
                move = m[3]
                if move not in moves[mon]:
                    moves[mon].append(move)

    # Normalize move names
    moves_parsed = {
        mon.replace(" ", "").translate(str.maketrans("", "", string.punctuation)): [
            move.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
            for move in mon_moves
        ]
        for mon, mon_moves in moves.items()
    }

    # Handle team preview
    if "random" in format or True:
        start_line = next(
            (
                j
                for j, line in enumerate(battle_turns_text[0])
                if "|start" in line or "|teampreview" in line
            ),
            -1,
        )
        if start_line == -1:
            return False
        for species, mon in zip(team_mons, team_player):
            mon_species = species.split(",")[0]
            for move in moves[mon_species]:
                battle_turns_text[0].insert(
                    start_line, f"|premove|{player_id}a: {mon_species}|{move}"
                )
            if "random" in format:
                battle_turns_text[0].insert(start_line, mon)
        if "random" in format:
            battle_turns_text[0].insert(start_line, "|teampreview")

    # Create player and battle simulation
    llm_player = LLMPlayer(
        battle_format=format,
        model="gpt",
        temperature=1.0,
        prompt_algo="io",
        account_configuration=AccountConfiguration(p1_username, ""),
        prompt_translate=prompt_translate,
        save_replays=False,
    )
    llm_player._dynamax_disable = False
    battle = await llm_player._create_battle(f">battle-{format}-{battle_id}".split("-"))
    sim = LocalSim(
        battle,
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

    # Process each turn
    for turn, message in enumerate(battle_turns_text):
        split_messages = [m.replace("\n", "").split("|") for m in message]

        # Handle initial turn
        if turn == 0:
            for msg in split_messages:
                if len(msg) >= 2 and msg[1]:
                    try:
                        sim._handle_battle_message(msg)
                    except (KeyError, ValueError, NotImplementedError):
                        continue
            continue

        # Extract player and opponent actions
        player_action = opponent_action = None
        for m in split_messages:
            if len(m) >= 4 and m[1] in ["move", "switch"]:
                action = (
                    f'{{"move":"{m[3]}"}}'
                    if m[1] == "move"
                    else f'{{"switch":"{m[2].split(":")[1]}"}}'
                )
                if f"{player_id}a" in m[2]:
                    player_action = action
                elif f"{opponent_id}a" in m[2]:
                    opponent_action = action

        # Skip turn if no player action
        if not player_action:
            for msg in split_messages:
                if len(msg) >= 2 and msg[1]:
                    try:
                        sim._handle_battle_message(msg)
                    except (KeyError, ValueError, NotImplementedError):
                        continue
            continue

        # Set default opponent action if none
        opponent_action = opponent_action or '"None"'

        # Add available moves to current pokemon
        if sim.battle.active_pokemon.species in moves_parsed:
            sim.battle._available_moves = [
                Move(move, gen=llm_player.gen.gen)
                for move in moves_parsed[sim.battle.active_pokemon.species]
            ]

        # Create player prompt from battle sim
        try:
            system_prompt, state_prompt, state_action_prompt = sim.state_translate(
                sim.battle
            )
            player_prompt = system_prompt + state_prompt + state_action_prompt

            # Create output data
            output = f'{{"action":{player_action[1:-1].split(":")[1]},"opponent":{opponent_action.split(":")[1][:-1] if ':' in opponent_action else opponent_action}}}'

            # Create final data entry
            player_text = {
                "instruction": player_prompt,
                "output": output,
            }
            json_text.append(player_text)

        except Exception as e:
            print(f"Battle not saved: {e}")
            return False

        # Process battle messages
        for msg in split_messages:
            if len(msg) >= 2 and msg[1]:
                try:
                    sim._handle_battle_message(msg)
                except (KeyError, ValueError, NotImplementedError):
                    continue

    return True


async def main():
    # ['text', 'month_year', 'gamemode', 'elo', 'battle_id']
    dataset = load_from_disk("mini_dataset")
    gen = 9
    json_text = []
    for battle in dataset:
        battle_id = battle["battle_id"]
        battle_text = battle["text"]
        battle_gamemode = battle["gamemode"]
        try:
            success = await f(battle_text, battle_gamemode, battle_id, json_text, gen)
        except Exception as e:
            print(f"Error processing battle {battle_id}: {e}")
        exit()


if __name__ == "__main__":
    asyncio.run(main())
