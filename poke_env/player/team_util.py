from numpy.random import randint

from poke_env.player.baselines import AbyssalPlayer, MaxBasePowerPlayer, OneStepPlayer
from poke_env.player.llm_player import LLMPlayer
from poke_env.player.player import Player
from poke_env.player.prompts import prompt_translate, state_translate2
from poke_env.player.pythia import Pythia
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration


def load_random_team(id=None):
    if id == None:
        team_id = randint(1, 14)
    else:
        team_id = id
    with open(f"poke_env/data/static/teams/gen9ou{team_id}.txt", "r") as f:
        team = f.read()
    return team


def get_llm_player(
    args,
    model: str,
    prompt_algo: str,
    name: str,
    KEY: str = "",
    battle_format="gen9ou",
    llm_backend=None,
    device=0,
    PNUMBER1: str = "",
    USERNAME: str = "",
    PASSWORD: str = "",
    online: bool = False,
) -> Player:
    server_config = None
    if online:
        server_config = ShowdownServerConfiguration
    if USERNAME == "":
        USERNAME = name
    if prompt_algo == "abyssal":
        return AbyssalPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(
                f"{USERNAME}{PNUMBER1}", PASSWORD
            ),
            server_configuration=server_config,
        )
    elif prompt_algo == "max_power":
        return MaxBasePowerPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(
                f"{USERNAME}{PNUMBER1}", PASSWORD
            ),
            server_configuration=server_config,
        )
    elif prompt_algo == "random":
        return RandomPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(
                f"{USERNAME}{PNUMBER1}", PASSWORD
            ),
            server_configuration=server_config,
        )
    elif prompt_algo == "one_step":
        return OneStepPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(
                f"{USERNAME}{PNUMBER1}", PASSWORD
            ),
            server_configuration=server_config,
        )
    elif "pokellmon" in name:
        return LLMPlayer(
            battle_format=battle_format,
            api_key=KEY,
            model=model,
            temperature=args.temperature,
            prompt_algo=prompt_algo,
            log_dir=args.log_dir,
            account_configuration=AccountConfiguration(
                f"{USERNAME}{PNUMBER1}", PASSWORD
            ),
            server_configuration=server_config,
            save_replays=args.log_dir,
            device=device,
            llm_backend=llm_backend,
        )
    elif "pokechamp" in name:
        return LLMPlayer(
            battle_format=battle_format,
            api_key=KEY,
            model=model,
            temperature=args.temperature,
            prompt_algo="minimax",
            log_dir=args.log_dir,
            account_configuration=AccountConfiguration(
                f"{USERNAME}{PNUMBER1}", PASSWORD
            ),
            server_configuration=server_config,
            save_replays=args.log_dir,
            prompt_translate=prompt_translate,
            #    prompt_translate=state_translate2,
            device=device,
            llm_backend=llm_backend,
        )
    elif "pythia" in name:
        return Pythia(
            battle_format=battle_format,
            api_key=KEY,
            model=model,
            temperature=args.temperature,
            log_dir=args.log_dir,
            account_configuration=AccountConfiguration(
                f"{USERNAME}{PNUMBER1}", PASSWORD
            ),
            server_configuration=server_config,
            save_replays=args.log_dir,
            prompt_translate=prompt_translate,
            device=device,
        )
    else:
        raise ValueError("Bot not found")
