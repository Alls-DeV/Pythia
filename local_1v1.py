import argparse
import asyncio
import os

from tqdm import tqdm

from common import *
from poke_env.player.player import Player
from poke_env.player.team_util import get_llm_player, load_random_team

Pythia = {
    "name": "pythia",
    "prompt_algo": "None",
    "model": "deepseek-chat",
    "device": 0,
}

Pokechamp = {
    "name": "pokechamp",
    "prompt_algo": "minimax",
    "model": "gpt-4o",
    "device": 0,
}

Abyssal = {
    "name": "abyssal",
    "prompt_algo": "abyssal",
    "model": "None",
    "device": 0,
}

PLAYER = Pythia
OPPONENT = Pokechamp

parser = argparse.ArgumentParser()

# Player arguments
parser.add_argument(
    "--player_prompt_algo", default=PLAYER["prompt_algo"], choices=ALGO_CHOICES
)
parser.add_argument(
    "--player_model", type=str, default=PLAYER["model"], choices=MODEL_CHOICES
)
parser.add_argument("--player_name", type=str, default=PLAYER["name"])
parser.add_argument("--player_device", type=int, default=PLAYER["device"])

# Opponent arguments
parser.add_argument(
    "--opponent_prompt_algo", default=OPPONENT["prompt_algo"], choices=ALGO_CHOICES
)
parser.add_argument(
    "--opponent_model", type=str, default=OPPONENT["model"], choices=MODEL_CHOICES
)
parser.add_argument("--opponent_name", type=str, default=OPPONENT["name"])
parser.add_argument("--opponent_device", type=int, default=OPPONENT["device"])

# Shared arguments
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--battle_format", default="gen9ou", choices=BATTLE_FORMAT_CHOICES)
parser.add_argument("--log_dir", type=str, default="./battle_log/one_vs_one")

args = parser.parse_args()


async def main():
    player: Player = get_llm_player(
        args,
        args.player_model,
        args.player_prompt_algo,
        args.player_name,
        device=args.player_device,
        PNUMBER1=PNUMBER1,  # for name uniqueness locally
        battle_format=args.battle_format,
    )

    opponent: Player = get_llm_player(
        args,
        args.opponent_model,
        args.opponent_prompt_algo,
        args.opponent_name,
        device=args.opponent_device,
        PNUMBER1=PNUMBER1,  # for name uniqueness locally
        battle_format=args.battle_format,
    )

    player_team_id = 18
    opponent_team_id = 18
    if not "random" in args.battle_format:
        player.update_team(load_random_team(player_team_id))
        opponent.update_team(load_random_team(opponent_team_id))

    N = 1
    pbar = tqdm(total=N)
    for i in range(N):
        await player.battle_against(opponent, n_battles=1)
        if not "random" in args.battle_format:
            player.update_team(load_random_team())
            opponent.update_team(load_random_team())
        pbar.set_description(f"{player.win_rate*100:.2f}%")
        pbar.update(1)

        for trainer in [player, opponent]:
            if "gpt" in trainer.model or "deepseek" in trainer.model:
                with open(f"./battle_prompts/{PNUMBER1}/log_{trainer.model}", "a") as f:
                    f.write(
                        f"total explored nodes in the entire game: {trainer.total_explored_nodes}\n"
                    )
                    f.write(
                        f"total time spent on choosing move: {trainer.choose_move_time}\n"
                    )
                    f.write(
                        f"total time spent on llm thinking: {trainer.llm_thinking_time}\n"
                    )
                    f.write(
                        f"diff choose move - llm thinking: {trainer.choose_move_time - trainer.llm_thinking_time}\n"
                    )
                    f.write(
                        f"total completion tokens: {trainer.llm.completion_tokens}\n"
                    )
                    f.write(f"total prompt tokens: {trainer.llm.prompt_tokens}\n")
                trainer.total_explored_nodes = 0
                trainer.choose_move_time = 0
                trainer.llm_thinking_time = 0
                trainer.llm.completion_tokens = 0
                trainer.llm.prompt_tokens = 0

    print(f"player winrate: {player.win_rate*100}")


if __name__ == "__main__":
    asyncio.run(main())
