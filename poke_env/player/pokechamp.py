import ast
import datetime
import json
import os
import random
import sys
import time
from copy import copy, deepcopy
from difflib import get_close_matches
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from common import PNUMBER1
from poke_env.data.gen_data import GenData
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.gpt_player import GPTPlayer
from poke_env.player.llama_player import LLAMAPlayer
from poke_env.player.local_simulation import LocalSim, SimNode
from poke_env.player.player import BattleOrder, Player
from poke_env.player.prompts import (
    get_number_turns_faint,
    get_status_num_turns_fnt,
    state_translate,
)


class Pokechamp(Player):
    def __init__(
        self,
        battle_format,
        api_key="",
        model="deepseek-chat",
        temperature=0.3,
        log_dir=None,
        team=None,
        save_replays=None,
        account_configuration=None,
        server_configuration=None,
        K=2,
        _use_strat_prompt=False,
        prompt_translate: Callable = state_translate,
        device=0,
    ):

        super().__init__(
            battle_format=battle_format,
            team=team,
            save_replays=save_replays,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
        )

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action: Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.model = model
        self.temperature = temperature
        self.log_dir = log_dir
        self.api_key = api_key
        self.gen = GenData.from_format(battle_format)
        self.genNum = self.gen.gen
        self.prompt_translate = prompt_translate

        self.strategy_prompt = ""
        self.team_str = team
        self.use_strat_prompt = _use_strat_prompt

        with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
            self.move_effect = json.load(f)
        # only used in old prompting method, replaced by statistcal sets data
        with open(f"./poke_env/data/static/moves/gen8pokemon_move_dict.json", "r") as f:
            self.pokemon_move_dict = json.load(f)
        with open("./poke_env/data/static/abilities/ability_effect.json", "r") as f:
            self.ability_effect = json.load(f)
        # only used is old prompting method
        with open(
            "./poke_env/data/static/abilities/gen8pokemon_ability_dict.json", "r"
        ) as f:
            self.pokemon_ability_dict = json.load(f)
        with open("./poke_env/data/static/items/item_effect.json", "r") as f:
            self.item_effect = json.load(f)
        self.pokemon_item_dict = {}
        with open(
            f"./poke_env/data/static/pokedex/gen{self.gen.gen}pokedex.json", "r"
        ) as f:
            self._pokemon_dict = json.load(f)

        self.last_plan = ""

        if "gpt" in model or "deepseek" in model:
            self.llm = GPTPlayer(self.api_key)
        elif "llama" in model:
            self.llm = LLAMAPlayer(model=model, device=device)
        else:
            raise NotImplementedError("LLM type not implemented:", model)

        self.K = K  # for minimax, SC, ToT

        self.total_choose_move_time = 0
        self.total_explored_nodes = 0

    def check_all_pokemon(self, pokemon_str: str) -> Pokemon:
        valid_pokemon = None
        if pokemon_str in self._pokemon_dict:
            valid_pokemon = pokemon_str
        else:
            closest = get_close_matches(
                pokemon_str, self._pokemon_dict.keys(), n=1, cutoff=0.8
            )
            if len(closest) > 0:
                valid_pokemon = closest[0]
        if valid_pokemon is None:
            return None
        pokemon = Pokemon(species=pokemon_str, gen=self.genNum)
        return pokemon

    def choose_move(self, battle: AbstractBattle):
        sim = LocalSim(
            battle,
            self.move_effect,
            self.pokemon_move_dict,
            self.ability_effect,
            self.pokemon_ability_dict,
            self.item_effect,
            self.pokemon_item_dict,
            self.gen,
            self._dynamax_disable,
            self.strategy_prompt,
            format=self.format,
            prompt_translate=self.prompt_translate,
        )
        if battle.turn <= 1 and self.use_strat_prompt:
            self.strategy_prompt = sim.get_llm_system_prompt(
                self.format, self.llm, team_str=self.team_str, model="gpt-4o-2024-05-13"
            )

        if battle.active_pokemon:
            if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
                next_action = BattleOrder(battle.available_switches[0])
                return next_action
            elif (
                not battle.active_pokemon.fainted
                and len(battle.available_moves) == 1
                and len(battle.available_switches) == 0
            ):
                return self.choose_max_damage_move(battle)
        elif len(battle.available_moves) <= 1 and len(battle.available_switches) == 0:
            return self.choose_max_damage_move(battle)

        retries = 2
        try:
            start_time = time.time()
            action, explored_nodes = self.tree_search(retries, battle)
            end_time = time.time()
            # I'm returning the idx of the last node, since they start at 0 I'm adding 1 to the count
            explored_nodes += 1

            with open(f"./llm_log/{PNUMBER1}/log_{self.username}", "a") as f:
                f.write(f"explored nodes to find the best move: {explored_nodes}\n")
                f.write(f"time to choose the move: {end_time - start_time}\n")
                f.write(f"llm response time: {self.llm.single_move_response_time}\n")
                f.write(
                    f"overhead time: {end_time - start_time - self.llm.single_move_response_time}\n"
                )
                f.write(f"llm prompt tokens: {self.llm.single_move_prompt_tokens}\n")
                f.write(
                    f"llm completion tokens: {self.llm.single_move_completion_tokens}\n"
                )
                f.write("-" * 100 + "\n")

            self.llm.single_move_response_time = 0
            self.llm.single_move_prompt_tokens = 0
            self.llm.single_move_completion_tokens = 0

            self.total_choose_move_time += end_time - start_time
            self.total_explored_nodes += explored_nodes

            return action
        except Exception as e:
            print("minimax step failed. Using dmg calc")
            print(f"Exception: {e}", "passed")
            return self.choose_max_damage_move(battle)

    def io(
        self,
        retries,
        system_prompt,
        state_prompt,
        constraint_prompt_cot,
        constraint_prompt_io,
        state_action_prompt,
        battle: Battle,
        sim,
        dont_verify=False,
        log_dict=dict(),
    ):
        """Implements chain-of-thought reasoning to decide on an action."""
        next_action = None
        cot_prompt = "In fewer than 3 sentences, let's think step by step:"
        state_prompt_io = (
            state_prompt + state_action_prompt + constraint_prompt_io + cot_prompt
        )

        for i in range(retries):
            try:
                llm_output = self.llm.get_LLM_action(
                    system_prompt=system_prompt,
                    user_prompt=state_prompt_io,
                    model=self.model,
                    log_dict=log_dict,
                )

                # load when llm does heavylifting for parsing
                llm_action_json = json.loads(llm_output)
                next_action = None

                dynamax = "dynamax" in llm_action_json.keys()
                tera = "terastallize" in llm_action_json.keys()
                is_a_move = dynamax or tera

                if "move" in llm_action_json.keys() or is_a_move:
                    if dynamax:
                        llm_move_id = llm_action_json["dynamax"].strip()
                    elif tera:
                        llm_move_id = llm_action_json["terastallize"].strip()
                    else:
                        llm_move_id = llm_action_json["move"].strip()
                    move_list = battle.active_pokemon.moves.values()
                    if dont_verify:  # opponent
                        move_list = battle.opponent_active_pokemon.moves.values()
                    for i, move in enumerate(move_list):
                        if move.id.lower().replace(
                            " ", ""
                        ) == llm_move_id.lower().replace(" ", ""):
                            # next_action = self.create_order(move, dynamax=sim._should_dynamax(battle), terastallize=sim._should_terastallize(battle))
                            next_action = self.create_order(
                                move, dynamax=dynamax, terastallize=tera
                            )
                    if next_action is None and dont_verify:
                        # unseen move so just check if it is in the action prompt
                        if llm_move_id.lower().replace(" ", "") in state_action_prompt:
                            next_action = self.create_order(
                                Move(
                                    llm_move_id.lower().replace(" ", ""), self.gen.gen
                                ),
                                dynamax=dynamax,
                                terastallize=tera,
                            )
                elif "switch" in llm_action_json.keys():
                    llm_switch_species = llm_action_json["switch"].strip()
                    switch_list = battle.available_switches
                    if dont_verify:  # opponent prediction
                        observable_switches = []
                        for _, opponent_pokemon in battle.opponent_team.items():
                            if not opponent_pokemon.active:
                                observable_switches.append(opponent_pokemon)
                        switch_list = observable_switches
                    for i, pokemon in enumerate(switch_list):
                        if pokemon.species.lower().replace(
                            " ", ""
                        ) == llm_switch_species.lower().replace(" ", ""):
                            next_action = self.create_order(pokemon)
                else:
                    raise ValueError("No valid action")

                if next_action is not None:
                    break
            except Exception as e:
                print(f"Exception: {e}", "passed")
                continue
        if next_action is None:
            print("No action found")
            try:
                print("No action found", llm_action_json, dont_verify)
            except:
                pass
            print()
            # raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
            next_action = self.choose_max_damage_move(battle)
        return next_action

    def estimate_matchup(
        self,
        sim: LocalSim,
        battle: Battle,
        mon: Pokemon,
        mon_opp: Pokemon,
        is_opp: bool = False,
    ) -> Tuple[Move, int]:
        """Evaluates which move is most effective in the current matchup, with the number of turns to faint the opponent."""
        hp_remaining = []
        moves = list(mon.moves.keys())
        if is_opp:
            moves = sim.get_opponent_current_moves(mon=mon)
        if battle.active_pokemon.species == mon.species and not is_opp:
            moves = [move.id for move in battle.available_moves]
        for move_id in moves:
            move = Move(move_id, gen=sim.gen.gen)
            t = np.inf
            if move.category == MoveCategory.STATUS:
                # apply stat boosting effects to see if it will KO in fewer turns
                t = get_status_num_turns_fnt(
                    mon, move, mon_opp, sim, boosts=mon._boosts.copy()
                )
            else:
                t = get_number_turns_faint(
                    mon,
                    move,
                    mon_opp,
                    sim,
                    boosts1=mon._boosts.copy(),
                    boosts2=mon_opp.boosts.copy(),
                )
            hp_remaining.append(t)
            # _, hp2, _, _ = sim.calculate_remaining_hp(battle.active_pokemon, battle.opponent_active_pokemon, move, None)
            # hp_remaining.append(hp2)
        hp_best_index = np.argmin(hp_remaining)
        best_move = moves[hp_best_index]
        best_move_turns = hp_remaining[hp_best_index]
        best_move = Move(best_move, gen=sim.gen.gen)
        best_move = self.create_order(best_move)
        # check terastallize for gen 9
        if sim.battle._data.gen == 9 and sim.battle.can_tera:
            mon.terastallize()
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen)
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(
                        mon,
                        move,
                        mon_opp,
                        sim,
                        boosts1=mon._boosts.copy(),
                        boosts2=mon_opp.boosts.copy(),
                    )
                    if t < best_move_turns:
                        best_move = self.create_order(move, terastallize=True)
                        best_move_turns = t
            mon.unterastallize()

        return best_move, best_move_turns

    def dmg_calc_move(self, battle: AbstractBattle, return_move: bool = False):
        """Wrapper of estimate_matchup"""
        sim = LocalSim(
            battle,
            self.move_effect,
            self.pokemon_move_dict,
            self.ability_effect,
            self.pokemon_ability_dict,
            self.item_effect,
            self.pokemon_item_dict,
            self.gen,
            self._dynamax_disable,
            format=self.format,
        )
        best_action = None
        best_action_turns = np.inf
        if battle.available_moves and not battle.active_pokemon.fainted:
            # try moves and find hp remaining for opponent
            mon = battle.active_pokemon
            mon_opp = battle.opponent_active_pokemon
            best_action, best_action_turns = self.estimate_matchup(
                sim, battle, mon, mon_opp
            )
        if return_move:
            if best_action is None:
                return None, best_action_turns
            return best_action.order, best_action_turns
        if best_action_turns > 4:
            return None, np.inf
        if best_action is not None:
            return best_action, best_action_turns
        return self.choose_random_move(battle), 1

    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        """Computes a numerical score for a matchup between two Pokémon, a higher score indicates a better matchup for the player's Pokémon."""
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def tree_search(self, retries, battle, sim=None) -> Tuple[BattleOrder, int]:
        # generate local simulation
        node_idx = 0
        root = SimNode(
            battle,
            self.move_effect,
            self.pokemon_move_dict,
            self.ability_effect,
            self.pokemon_ability_dict,
            self.item_effect,
            self.pokemon_item_dict,
            self.gen,
            self._dynamax_disable,
            idx=node_idx,
            depth=1,
            format=self.format,
            prompt_translate=self.prompt_translate,
            sim=sim,
        )
        q = [root]
        # create node and add to q B times
        while len(q) != 0:
            node = q.pop(0)
            # choose node for expansion
            # generate B actions
            player_actions = []
            (
                system_prompt,
                state_prompt,
                constraint_prompt_cot,
                constraint_prompt_io,
                state_action_prompt,
                action_prompt_switch,
                action_prompt_move,
            ) = node.simulation.get_player_prompt(return_actions=True)

            log_dict = {
                "player_name": self.username,
                "turn": node.simulation.battle.turn,
                "node_idx": node.idx,
                "parent_idx": node.parent_node.idx if node.parent_node else -1,
                "depth": node.depth,
            }
            # end if terminal
            if node.simulation.is_terminal() or node.depth == self.K:
                try:
                    # value estimation for leaf nodes
                    value_prompt = (
                        "Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points."
                        + "Add points based on the effectiveness of current available moves."
                        + "Award points for each pokemon remaining on the player's team, weighted by their strength"
                        + "Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. "
                        + "Subtract points for excessive switching."
                        + "Subtract points based on the effectiveness of the opponent's current moves, especially if they have a faster speed."
                        + "Remove points for each pokemon remaining on the opponent's team, weighted by their strength.\n"
                    )

                    cot_prompt = 'Answer with the score in the JSON format: {"score": <total_points>}. '
                    state_prompt_io = state_prompt + value_prompt + cot_prompt
                    llm_output = self.llm.get_LLM_action(
                        system_prompt=system_prompt,
                        user_prompt=state_prompt_io,
                        model=self.model,
                        log_dict=log_dict,
                    )
                    # load when llm does heavylifting for parsing
                    llm_action_json = json.loads(llm_output)
                    node.hp_diff = int(llm_action_json["score"])
                except Exception as e:
                    # the value given by the LLM is between 1 and 100, whereas this one can be negative
                    node.hp_diff = node.simulation.get_hp_diff()
                    print(e)

                continue

            # estimate opp
            try:
                action_opp, opp_turns = self.estimate_matchup(
                    node.simulation,
                    node.simulation.battle,
                    node.simulation.battle.opponent_active_pokemon,
                    node.simulation.battle.active_pokemon,
                    is_opp=True,
                )
            except:
                action_opp = None
                opp_turns = np.inf
            ##############################
            # generate players's action  #
            ##############################
            if (
                not node.simulation.battle.active_pokemon.fainted
                and len(battle.available_moves) > 0
            ):
                # get dmg calc move
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(
                    node.simulation.battle
                )
                if dmg_calc_out is not None:
                    if dmg_calc_turns <= opp_turns:
                        try:
                            # ask LLM to use heuristic tool or minimax search
                            tool_prompt = """Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                                1. Damage calculator advantages:
                                - Quick and efficient for finding optimal damaging moves
                                - Useful when a clear type advantage or high-power move is available
                                - Effective when the opponent's is not switching and current pokemon is likely to KO opponent

                                2. Minimax tree search advantages:
                                - Can model opponent behavior and predict future moves
                                - Useful in complex situations with multiple viable options
                                - Effective when long-term strategy is crucial

                                3. Current battle state:
                                - Remaining Pokémon on each side
                                - Health of active Pokémon
                                - Type matchups
                                - Available moves and their effects
                                - Presence of status conditions or field effects

                                4. Uncertainty level:
                                - How predictable is the opponent's next move?
                                - Are there multiple equally viable options for your next move?

                                Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

                                {"choice":"damage calculator"} or {"choice":"minimax"}"""

                            state_prompt_io = state_prompt + tool_prompt
                            llm_output = self.llm.get_LLM_action(
                                system_prompt=system_prompt,
                                user_prompt=state_prompt_io,
                                model=self.model,
                                log_dict=log_dict,
                            )
                            # load when llm does heavylifting for parsing
                            llm_action_json = json.loads(llm_output)
                            if "choice" in llm_action_json.keys():
                                if llm_action_json["choice"] != "minimax":
                                    return dmg_calc_out, node_idx
                        except:
                            print("defaulting to minimax")
                    player_actions.append(dmg_calc_out)

            # get llm switch
            if len(node.simulation.battle.available_switches) != 0:
                state_action_prompt_switch = (
                    state_action_prompt
                    + action_prompt_switch
                    + "\nYou can only choose to switch this turn.\n"
                )
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}.\n'
                for i in range(2):
                    action_llm_switch = self.io(
                        retries,
                        system_prompt,
                        state_prompt,
                        constraint_prompt_cot,
                        constraint_prompt_io,
                        state_action_prompt_switch,
                        node.simulation.battle,
                        node.simulation,
                        log_dict=log_dict,
                    )
                    if len(player_actions) == 0:
                        player_actions.append(action_llm_switch)
                    elif action_llm_switch.message != player_actions[-1].message:
                        player_actions.append(action_llm_switch)

            if (
                not node.simulation.battle.active_pokemon.fainted
                and len(battle.available_moves) > 0
            ):
                # get llm move
                state_action_prompt_move = (
                    state_action_prompt
                    + action_prompt_move
                    + "\nYou can only choose to move this turn.\n"
                )
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"}.\n'
                action_llm_move = self.io(
                    retries,
                    system_prompt,
                    state_prompt,
                    constraint_prompt_cot,
                    constraint_prompt_io,
                    state_action_prompt_move,
                    node.simulation.battle,
                    node.simulation,
                    log_dict=log_dict,
                )
                if len(player_actions) == 0:
                    player_actions.append(action_llm_move)
                elif action_llm_move.message != player_actions[0].message:
                    player_actions.append(action_llm_move)

            ##############################
            # generate opponent's action #
            ##############################
            opponent_actions = []
            # dmg calc suggestion
            if action_opp is not None:
                opponent_actions.append(self.create_order(action_opp))
            # heuristic matchup switch action
            best_score = np.inf
            best_action = None
            for mon in node.simulation.battle.opponent_team.values():
                if (
                    mon.species
                    == node.simulation.battle.opponent_active_pokemon.species
                ):
                    continue
                score = self._estimate_matchup(
                    mon, node.simulation.battle.active_pokemon
                )
                if score < best_score:
                    best_score = score
                    best_action = mon
            if best_action is not None:
                opponent_actions.append(self.create_order(best_action))

            # create opponent prompt from battle sim
            (
                system_prompt_o,
                state_prompt_o,
                constraint_prompt_cot_o,
                constraint_prompt_io_o,
                state_action_prompt_o,
            ) = node.simulation.get_opponent_prompt(system_prompt)
            action_o = self.io(
                2,
                system_prompt_o,
                state_prompt_o,
                constraint_prompt_cot_o,
                constraint_prompt_io_o,
                state_action_prompt_o,
                node.simulation.battle,
                node.simulation,
                dont_verify=True,
                log_dict=log_dict,
            )
            is_repeat_action_o = np.array(
                [
                    action_o.message == opponent_action.message
                    for opponent_action in opponent_actions
                ]
            ).any()
            if not is_repeat_action_o:
                opponent_actions.append(action_o)

            # simulate outcome
            if node.depth < self.K:
                for action_p in player_actions:
                    for action_o in opponent_actions:
                        node_new = copy(node)
                        node_new.simulation.battle = copy(node.simulation.battle)
                        node_new.children = []
                        node_new.depth = node.depth + 1
                        node_new.action = action_p
                        node_new.action_opp = action_o
                        node_new.parent_node = node
                        node_new.parent_action = node.action
                        node_idx += 1
                        node_new.idx = node_idx
                        node.children.append(node_new)
                        node_new.simulation.step(action_p, action_o)
                        q.append(node_new)

        # choose best action according to max or min rule
        def get_tree_action(root: SimNode):
            if len(root.children) == 0:
                return root.action, root.hp_diff, root.action_opp
            score_dict = {}
            action_dict = {}
            opp_dict = {}
            for child in root.children:
                action = str(child.action.order)
                _, score, _ = get_tree_action(child)
                if action in score_dict.keys():
                    # imitation
                    # score_dict[action] = score + score_dict[action]
                    # minimax
                    score_dict[action] = min(score, score_dict[action])
                else:
                    score_dict[action] = score
                    action_dict[action] = child.action
                    opp_dict[action] = child.action_opp
            scores = list(score_dict.values())
            best_action_str = list(action_dict.keys())[np.argmax(scores)]
            return (
                action_dict[best_action_str],
                score_dict[best_action_str],
                opp_dict[best_action_str],
            )

        action, _, action_opp = get_tree_action(root)
        return action, node_idx

    def choose_max_damage_move(self, battle: Battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)
