import asyncio
import string

import numpy as np
from datasets import load_from_disk

from poke_env.player.translate import get_player_team, recursive_nick_removal


async def extract_stats(
    battle_text: str,
    p1_side: bool,
):
    # TODO: broken with metronome
    battle_lines = battle_text.splitlines()

    # Extract Elo ratings and preprocess battle text
    battle_lines = [t.replace("type: null", "typenull") for t in battle_lines]
    battle_lines = recursive_nick_removal(battle_lines)

    # Parse battle into turns and extract player information
    battle_turns_text, p1_username, p2_username, winner_username = [], "", "", ""
    turn_text = []
    for line in battle_lines:
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
        return -1, -1, -1

    if p1_side:
        player_id = "p1"
    else:
        player_id = "p2"

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

    cnt = 0
    for _moves in moves_parsed.values():
        cnt += len(_moves)

    print(moves_parsed)
    for turn in battle_turns_text:
        print(turn)

    return cnt, len(moves_parsed), len(battle_turns_text)


async def main():
    # ['text', 'month_year', 'gamemode', 'elo', 'battle_id']
    # dataset = load_from_disk("mini_dataset")
    from battle_translate import load_filtered_dataset

    elo_range = "1600+"
    gamemode = "gen9ou"
    dataset = load_filtered_dataset(
        min_month="January2022",
        max_month="March2026",
        elo_ranges=[elo_range],
        split="train",
        gamemode=gamemode,
    )
    dataset.save_to_disk("dataset50")
    exit()
    dataset = dataset.filter(lambda x: x["battle_id"] == "2279070580-2025-01-11")

    battle_with_errors = 0
    # dataset_info[moves_number] = [total_battle_with_moves_number, total_pokemon_number, total_turns_number]
    dataset_info = dict()
    for battle in dataset:
        battle_id = battle["battle_id"]
        battle_text = battle["text"]
        try:
            moves_number, pokemon_number, turns_number = await extract_stats(
                battle_text, p1_side=True
            )
            if moves_number <= 0:
                continue
            if moves_number > 24:
                print(f"moves number = {moves_number}")
                print(f"battle id = {battle_id}")
                print()
            if moves_number not in dataset_info:
                dataset_info[moves_number] = [
                    0,
                    0,
                    0,
                ]  # [battle_count, pokemon_count, turn_count]
            dataset_info[moves_number][
                0
            ] += 1  # Increment number of battles with this moves_number
            dataset_info[moves_number][1] += pokemon_number  # Add total pokemon count
            dataset_info[moves_number][2] += turns_number  # Add total turns count

        except Exception as e:
            # print(f"Error processing battle {battle_id}: {e}")
            battle_with_errors += 1

    print(f"Total number of battle: {len(dataset)}")
    print(f"Battle with problem while parsing: {battle_with_errors}")

    import json

    # ans = input("Do you want to save results? 1 yes, 2 no")
    ans = 2
    output_file = f"{elo_range}_{gamemode}_stats.json"
    if ans == "1":
        with open(output_file, "w") as ff:
            json.dump(dataset_info, ff, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
