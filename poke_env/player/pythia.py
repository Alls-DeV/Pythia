import json
import time
import traceback
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import orjson

from common import PNUMBER1
from poke_env.data.gen_data import GenData
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player.gpt_player import GPTPlayer
from poke_env.player.llama_player import LLAMAPlayer
from poke_env.player.local_simulation import LocalSim
from poke_env.player.player import BattleOrder, Player
from poke_env.player.pythia_prompt import state_translate


class Pythia(Player):
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
        K=1,
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
        with open("poke_env/data/static/gen9/ou/sets_1825.json", "r") as f:
            self.pokedex = orjson.loads(f.read())

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

    def choose_move(self, battle: AbstractBattle):
        try:
            start_time = time.time()
            # TODO: probably explored_nodes will be useless
            action, explored_nodes = self.choose_best_action(battle)
            end_time = time.time()
            print(
                f"Action chosen: {action}, explored nodes: {explored_nodes}, time taken: {end_time - start_time}\n"
            )

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
            print("----------------- STACK TRACE -----------------")
            traceback.print_exc()
            print("---------------------------------------------")

            print("minimax step failed. Using dmg calc")
            print(f"Exception: {e}", "passed")
            if battle.available_moves:
                return self.choose_max_damage_move(battle)
            else:
                return BattleOrder(battle.available_switches[0])

    # TODO: make it more complex
    def _evaluate_state(self, battle_state: AbstractBattle) -> float:
        """
        Evaluates the current battle state from the perspective of the player.
        Score = player's active Pokémon HP - opponent's active Pokémon HP.
        """
        score = battle_state._team_size["p1"] - battle_state._team_size["p2"]
        for pokemon in battle_state.team.values():
            score = score - 1 + pokemon.current_hp_fraction
        for pokemon in battle_state.opponent_team.values():
            score = score + 1 - pokemon.current_hp_fraction
        return score

    def _get_opponent_possible_moves(self, battle_state: AbstractBattle) -> List[Move]:
        # TODO: check how to handle strange status
        if (
            battle_state.opponent_active_pokemon
            and not battle_state.opponent_active_pokemon.fainted
        ):
            return battle_state.opponent_active_pokemon.moves.values()
        return []

    def _simulate_one_turn_local(
        self,
        source_battle_state: AbstractBattle,
        player_order: Optional[BattleOrder],
        opponent_order: Optional[BattleOrder],
    ) -> AbstractBattle:
        """
        Simulates one turn of a battle given player and opponent orders.
        Returns the new battle state after the turn.
        """
        # Create a new LocalSim instance for this simulation step
        local_sim = LocalSim(
            battle=source_battle_state,
            move_effect=self.move_effect,
            pokemon_move_dict=self.pokemon_move_dict,
            ability_effect=self.ability_effect,
            pokemon_ability_dict=self.pokemon_ability_dict,
            item_effect=self.item_effect,
            pokemon_item_dict=self.pokemon_item_dict,  # usually {}
            gen=self.gen,
            _dynamax_disable=self._dynamax_disable,  # class attribute
            format=self.format,  # class attribute
            prompt_translate=self.prompt_translate,  # class attribute
        )

        # TODO: check if we can do it better with smogon calculator
        local_sim.step(player_order, opponent_order)

        return local_sim.battle  # Return the new battle state

    def _enrich_opponent_pokemon(self, battle: AbstractBattle):
        mon = battle.opponent_active_pokemon
        if mon.species not in self.pokedex:
            return
        data = self.pokedex[mon.species]

        if len(mon.moves) < 4:
            moves = []
            for key in mon.moves:
                moves.append(mon.moves[key].id)

            for possible_move in data.get("moves", []):
                possible_move = (
                    possible_move["name"].lower().replace(" ", "").replace("-", "")
                )
                if possible_move not in moves:
                    mon._add_move(possible_move)
                    moves.append(possible_move)
                if len(moves) == 4:
                    break

        if mon.ability is None:
            if data.get("abilities"):
                mon.ability = (
                    data["abilities"][0]["name"]
                    .lower()
                    .replace(" ", "")
                    .replace("-", "")
                )

        if mon.item == "unknown_item":
            if data.get("items"):
                mon.item = (
                    data["items"][0]["name"].lower().replace(" ", "").replace("-", "")
                )

        if mon._terastallized_type is None:
            if data.get("tera"):
                mon._terastallized_type = PokemonType.from_name(data["tera"][0]["name"])

    def choose_best_action(
        self, battle: AbstractBattle
    ) -> Tuple[Optional[BattleOrder], int]:
        """
        Top-level function to choose the best action (move or switch) using Minimax.
        It iterates through all possible actions and uses helper functions to evaluate them.
        """
        # Create a single, enriched copy of the battle state to be used for all simulations this turn.
        sim_battle = deepcopy(battle)
        self._enrich_opponent_pokemon(sim_battle)

        best_action_order = self.choose_random_move(battle)  # Default action
        node_idx_counter = {"count": 0}

        # The alpha value for the root of our decision tree (a MAX node).
        root_alpha = -float("inf")

        # 1. Evaluate staying in and using a move
        if sim_battle.available_moves:
            # Sort moves to check more promising ones first, potentially leading to more pruning.
            # A simple heuristic is to check high-power moves first.
            sorted_moves = sorted(
                sim_battle.available_moves, key=lambda m: m.base_power, reverse=True
            )

            for move in sorted_moves:
                move_order = self.create_order(move)
                cnt = node_idx_counter["count"]
                score = self._get_move_value(
                    sim_battle, move_order, root_alpha, node_idx_counter
                )

                print(
                    f"Evaluating move {move.id} with score {score} in {node_idx_counter['count'] - cnt} nodes"
                )
                if score > root_alpha:
                    root_alpha = score
                    best_action_order = move_order

        # 2. Evaluate switching to a benched Pokémon
        if sim_battle.available_switches:
            for pokemon in sim_battle.available_switches:
                switch_order = self.create_order(pokemon)
                score = self._get_switch_value(
                    sim_battle, switch_order, root_alpha, node_idx_counter
                )
                print(
                    f"Evaluating switch to {pokemon.species} with score {score} in {node_idx_counter['count'] - cnt} nodes"
                )
                if score > root_alpha:
                    root_alpha = score
                    best_action_order = switch_order

        return best_action_order, node_idx_counter["count"]

    def _get_move_value(
        self,
        battle_state: AbstractBattle,
        move_order: BattleOrder,
        root_alpha: float,
        node_idx_counter: dict,
    ) -> float:
        """
        Calculates the Minimax value of performing a specific move.
        This function represents a MINIMIZER node (it finds the best opponent response).
        """
        min_score_for_move = float("inf")
        opponent_moves = self._get_opponent_possible_moves(battle_state)

        if not opponent_moves:  # Handle case where opponent is trapped
            next_state = self._simulate_one_turn_local(battle_state, move_order, None)
            return self._minimax_score(
                next_state,
                1,
                True,
                self.K,
                root_alpha,
                min_score_for_move,
                node_idx_counter,
            )

        for opp_move in opponent_moves:
            opp_order = self.create_order(opp_move)
            next_state = self._simulate_one_turn_local(
                battle_state, move_order, opp_order
            )

            # After both Pokémon move, it's our turn again in the next state (Maximizer)
            score = self._minimax_score(
                next_state,
                1,
                True,
                self.K,
                root_alpha,
                min_score_for_move,
                node_idx_counter,
            )

            min_score_for_move = min(min_score_for_move, score)

            # Pruning: If we find an opponent move that leads to a score worse for us
            # than a score we're already guaranteed from a different top-level action,
            # we don't need to check the opponent's other moves for this branch.
            if min_score_for_move <= root_alpha:
                return -float(
                    "inf"
                )  # Return a value that ensures this branch is not chosen

        return min_score_for_move

    def _get_switch_value(
        self,
        battle_state: AbstractBattle,
        switch_order: BattleOrder,
        root_alpha: float,
        node_idx_counter: dict,
    ) -> float:
        """
        Calculates the Minimax value of performing a switch.
        This involves the opponent getting a free attack on the incoming Pokémon.
        """
        min_score_for_switch = float("inf")
        opponent_moves = self._get_opponent_possible_moves(battle_state)

        if not opponent_moves:  # Handle case where opponent is trapped
            next_state = self._simulate_one_turn_local(battle_state, switch_order, None)
            # After we switch, it's our new Pokémon's turn to act (Maximizer)
            return self._minimax_score(
                next_state,
                1,
                True,
                self.K,
                root_alpha,
                min_score_for_switch,
                node_idx_counter,
            )

        for opp_move in opponent_moves:
            opp_order = self.create_order(opp_move)
            state_after_switch = self._simulate_one_turn_local(
                battle_state, switch_order, opp_order
            )

            # After we switch and they attack, it's our new active's turn (Maximizer)
            score = self._minimax_score(
                state_after_switch,
                1,
                True,
                self.K,
                root_alpha,
                min_score_for_switch,
                node_idx_counter,
            )

            min_score_for_switch = min(min_score_for_switch, score)

            if min_score_for_switch <= root_alpha:
                return -float("inf")  # Prune this branch

        return min_score_for_switch

    def _minimax_score(
        self,
        current_battle_state: AbstractBattle,
        depth: int,
        is_maximizing_player: bool,
        max_depth: int,
        alpha: float,
        beta: float,
        node_idx_ref: Dict,
    ) -> int:
        """
        The core recursive Minimax function. It explores the game tree from a given state.
        This function remains largely the same as the previous optimized version.
        """
        node_idx_ref["count"] += 1

        if depth == max_depth or current_battle_state.finished:
            return self._evaluate_state(current_battle_state)

        if is_maximizing_player:
            max_eval = -float("inf")
            # Here we need to check both moves and switches as options
            # For simplicity in this example, we assume we only evaluate moves from this point on
            # A more complex agent could evaluate switches at any depth.
            player_actions = current_battle_state.available_moves
            if not player_actions:
                return self._evaluate_state(current_battle_state)  # End of this path

            for p_action_obj in player_actions:
                # Recursively find the value of this move
                # This requires a "get_move_value"-like logic inside the recursion
                # This shows the complexity can grow. For now, let's keep the logic simple:
                # For simplicity, this recursive step will only consider moves vs moves.
                eval_for_move = self._get_move_value(
                    current_battle_state,
                    self.create_order(p_action_obj),
                    alpha,
                    node_idx_ref,
                )
                max_eval = max(max_eval, eval_for_move)
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval

        else:  # Minimizing player
            min_eval = float("inf")
            opponent_actions = self._get_opponent_possible_moves(current_battle_state)
            if not opponent_actions:
                return self._evaluate_state(current_battle_state)

            for o_action_obj in opponent_actions:
                # This would be the "get_opponent_move_value" logic
                # We assume player will respond optimally to the opponent's move
                eval_for_opp_move = self._get_player_response_value(
                    current_battle_state,
                    self.create_order(o_action_obj),
                    beta,
                    node_idx_ref,
                )
                min_eval = min(min_eval, eval_for_opp_move)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval

    # NOTE: The _minimax_score function gets more complex if it needs to handle all actions recursively.
    # The above implementation is a simplification where the deep recursion only considers move vs. move scenarios.
    # The following are placeholder helpers to illustrate the full recursive structure.

    def _get_player_response_value(
        self, battle_state, opp_order, beta, node_idx_counter
    ):
        max_score = -float("inf")
        player_moves = battle_state.available_moves
        if not player_moves:
            next_state = self._simulate_one_turn_local(battle_state, None, opp_order)
            return self._minimax_score(
                next_state, 1, False, self.K, max_score, beta, node_idx_counter
            )
        for p_move in player_moves:
            p_order = self.create_order(p_move)
            next_state = self._simulate_one_turn_local(battle_state, p_order, opp_order)
            score = self._minimax_score(
                next_state, 1, False, self.K, max_score, beta, node_idx_counter
            )
            max_score = max(max_score, score)
            if beta <= max_score:  # This alpha check is against the parent's beta
                break
        return max_score

    def choose_max_damage_move(self, battle: Battle) -> BattleOrder:
        if battle.available_moves:
            best_move = max(
                battle.available_moves, key=lambda move: move.base_power * move.accuracy
            )
            return self.create_order(best_move)
        return self.choose_random_move(battle)
