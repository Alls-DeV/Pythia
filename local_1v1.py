import argparse
import asyncio

from tqdm import tqdm

from common import *
from poke_env.player.player import Player
from poke_env.player.team_util import get_llm_player, load_random_team

Pythia = {
    "name": "pythia",
    "prompt_algo": "None",
    "model": "deepseek-chat",
    # "model": "gpt-4.1-nano",
    "device": 0,
}

PythiaLlama = {
    "name": "pyllama",
    "prompt_algo": "None",
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "device": 0,  # TODO: ???
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

Random = {
    "name": "random",
    "prompt_algo": "random",
    "model": "None",
    "device": 0,
}

PLAYER = Pythia
OPPONENT = Abyssal

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
        PNUMBER1=PNUMBER1 + 1,  # for name uniqueness locally
        battle_format=args.battle_format,
    )

    player_team_id = 2
    opponent_team_id = 3
    if not "random" in args.battle_format:
        player.update_team(load_random_team(player_team_id))
        opponent.update_team(load_random_team(opponent_team_id))

    await player.battle_against(opponent)

    for trainer in [player, opponent]:
        if any(substring in trainer.username for substring in ["pokechamp", "pythia"]):
            with open(f"./llm_log/{PNUMBER1}/log_{trainer.username}", "a") as f:
                f.write(
                    f"total explored nodes in the entire game: {trainer.total_explored_nodes}\n"
                )
                f.write(
                    f"total time spent on choosing moves: {trainer.total_choose_move_time}\n"
                )
                f.write(f"total llm response time: {trainer.llm.total_response_time}\n")
                f.write(
                    f"diff choose move - llm response time: {trainer.total_choose_move_time -trainer.llm.total_response_time}\n"
                )
                f.write(f"total prompt tokens: {trainer.llm.total_prompt_tokens}\n")
                f.write(
                    f"total completion tokens: {trainer.llm.total_completion_tokens}\n"
                )


if __name__ == "__main__":
    asyncio.run(main())
