import asyncio
from tqdm import tqdm
import os
import argparse

from common import *
from poke_env.player.team_util import get_llm_player, load_random_team

Pythia = {
    "name": "pythia",
    "prompt_algo": 'None',
    "backend": "deepseek-chat",
    "device": 0
}

Pokechamp = {
    "name": "pokechamp",
    "prompt_algo": "minimax",
    "backend": "gpt-4o",
    "device": 0
}

Abyssal = {
    "name": "abyssal",
    "prompt_algo": "abyssal",
    "backend": "gpt-4o",
    "device": 0
}

PLAYER = Pythia
OPPONENT = Abyssal

parser = argparse.ArgumentParser()

# Player arguments
parser.add_argument("--player_prompt_algo", default=PLAYER["prompt_algo"], choices=ALGO_CHOICES)
parser.add_argument("--player_backend", type=str, default=PLAYER["backend"], choices=BACKEND_CHOICES)
parser.add_argument("--player_name", type=str, default=PLAYER["name"])
parser.add_argument("--player_device", type=int, default=PLAYER["device"])

# Opponent arguments
parser.add_argument("--opponent_prompt_algo", default=OPPONENT["prompt_algo"], choices=ALGO_CHOICES)
parser.add_argument("--opponent_backend", type=str, default=OPPONENT["backend"], choices=BACKEND_CHOICES)
parser.add_argument("--opponent_name", type=str, default=OPPONENT["name"])
parser.add_argument("--opponent_device", type=int, default=OPPONENT["device"])

# Shared arguments
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--battle_format", default="gen9ou", choices=BATTLE_FORMAT_CHOICES)
parser.add_argument("--log_dir", type=str, default="./battle_log/one_vs_one")

args = parser.parse_args()

async def main():
    player = get_llm_player(args, 
                            args.player_backend, 
                            args.player_prompt_algo, 
                            args.player_name, 
                            device=args.player_device,
                            PNUMBER1=PNUMBER1,  # for name uniqueness locally
                            battle_format=args.battle_format)
    
    opponent = get_llm_player(args, 
                            args.opponent_backend, 
                            args.opponent_prompt_algo, 
                            args.opponent_name, 
                            device=args.opponent_device,
                            PNUMBER1=PNUMBER1,  # for name uniqueness locally
                            battle_format=args.battle_format)

    player_team_id = 18
    opponent_team_id = 19
    if not 'random' in args.battle_format:
        player.update_team(load_random_team(player_team_id))
        opponent.update_team(load_random_team(opponent_team_id))

    N = 1
    pbar = tqdm(total=N)
    for i in range(N):
        await player.battle_against(opponent, n_battles=1)
        if not 'random' in args.battle_format:
            player.update_team(load_random_team())
            opponent.update_team(load_random_team())
        pbar.set_description(f"{player.win_rate*100:.2f}%")
        pbar.update(1)
    print(f'player winrate: {player.win_rate*100}')


if __name__ == "__main__":
    asyncio.run(main())
