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
        # TODO: edge case
        if battle.active_pokemon.fainted:
            return BattleOrder(battle.available_switches[0])

        # TODO: edge case
        if battle.opponent_active_pokemon.fainted:
            return self.choose_random_move(battle)

        if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            return BattleOrder(battle.available_switches[0])
        elif (
            not battle.active_pokemon.fainted
            and len(battle.available_moves) == 1
            and len(battle.available_switches) == 0
        ):
            return self.choose_random_move(battle)

        try:
            start_time = time.time()
            # TODO: probably explored_nodes will be useless
            action, explored_nodes = self.minimax(battle)
            end_time = time.time()

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
            return self.choose_max_damage_move(battle)

    # TODO: make it more complex
    def _evaluate_state(self, battle_state: AbstractBattle) -> int:
        """
        Evaluates the current battle state from the perspective of the player.
        Score = player's active Pokémon HP - opponent's active Pokémon HP.
        """
        player_hp = battle_state.active_pokemon.current_hp
        opponent_hp = battle_state.opponent_active_pokemon.current_hp

        # TODO: simple check to remove in the future
        if battle_state.active_pokemon.fainted:
            assert player_hp == 0
        if battle_state.opponent_active_pokemon.fainted:
            assert opponent_hp == 0

        return player_hp - opponent_hp

    def _get_opponent_possible_moves(self, battle_state: AbstractBattle) -> List[Move]:
        """
        Gets a list of possible moves for the opponent's active Pokémon.
        """
        if (
            battle_state.opponent_active_pokemon
            and not battle_state.opponent_active_pokemon.fainted
        ):
            opp_moves = list(battle_state.opponent_active_pokemon.moves.values())

            result = [move for move in opp_moves if move.current_pp > 0]
            if not result:
                result.append(Move("struggle", self.genNum))
            return result

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

    def _minimax_score(
        self,
        current_battle_state: AbstractBattle,
        depth: int,
        is_maximizing_player: bool,
        max_depth: int,
        alpha: float,
        beta: float,
        node_idx_ref: Dict,
    ) -> float:
        """
        Recursive Minimax function with alpha-beta pruning.
        """
        node_idx_ref["count"] += 1

        # Base Cases: game over or max depth reached
        if (
            depth == max_depth
            or (
                current_battle_state.active_pokemon
                and current_battle_state.active_pokemon.fainted
            )
            or (
                current_battle_state.opponent_active_pokemon
                and current_battle_state.opponent_active_pokemon.fainted
            )
            or current_battle_state.finished
        ):
            return self._evaluate_state(current_battle_state)

        if is_maximizing_player:  # Player's turn
            max_eval = -float("inf")
            player_actions = current_battle_state.available_moves
            if (
                not player_actions
            ):  # No moves available (e.g. Struggle, or trapped without moves)
                if current_battle_state.active_pokemon:
                    player_actions = [Move("struggle", self.genNum)]
                else:  # Should not happen if pokemon not fainted, or battle should end.
                    return self._evaluate_state(current_battle_state)

            for p_action_obj in player_actions:
                p_order = self.create_order(p_action_obj)

                # Assume opponent will make a move to minimize our score
                # This requires simulating opponent's responses
                current_worst_outcome_for_player = float("inf")
                opponent_possible_moves = self._get_opponent_possible_moves(
                    current_battle_state
                )
                if not opponent_possible_moves:  # Opponent is trapped or has no moves
                    next_battle_state = self._simulate_one_turn_local(
                        current_battle_state, p_order, None
                    )
                    eval_score = self._minimax_score(
                        next_battle_state,
                        depth + 1,
                        False,
                        max_depth,
                        alpha,
                        beta,
                        node_idx_ref,
                    )
                    current_worst_outcome_for_player = eval_score

                else:
                    for o_action_obj in opponent_possible_moves:
                        o_order = self.create_order(
                            o_action_obj
                        )  # Opponent orders are created normally
                        next_battle_state = self._simulate_one_turn_local(
                            current_battle_state, p_order, o_order
                        )
                        eval_score = self._minimax_score(
                            next_battle_state,
                            depth + 1,
                            False,
                            max_depth,
                            alpha,
                            beta,
                            node_idx_ref,
                        )
                        current_worst_outcome_for_player = min(
                            current_worst_outcome_for_player, eval_score
                        )
                        # Beta for opponent's choice against this p_order (not strictly part of this node's alpha/beta, but for sub-search)

                max_eval = max(max_eval, current_worst_outcome_for_player)
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval

        else:  # Minimizing player's turn (Opponent's turn)
            min_eval = float("inf")
            opponent_actions = self._get_opponent_possible_moves(current_battle_state)
            if not opponent_actions:  # Opponent has no moves (e.g., Struggle)
                if (
                    current_battle_state.opponent_active_pokemon
                    and current_battle_state.opponent_active_pokemon.must_struggle()
                ):
                    opponent_actions = [Move("struggle", self.genNum)]
                else:  # Should not happen if pokemon not fainted or battle should end
                    return self._evaluate_state(current_battle_state)

            for o_action_obj in opponent_actions:
                o_order = self.create_order(o_action_obj)

                # Assume player will make a move to maximize their score
                current_best_outcome_for_player = -float("inf")
                player_possible_moves = current_battle_state.available_moves
                if not player_possible_moves:  # Player is trapped or has no moves
                    next_battle_state = self._simulate_one_turn_local(
                        current_battle_state, None, o_order
                    )
                    eval_score = self._minimax_score(
                        next_battle_state,
                        depth + 1,
                        True,
                        max_depth,
                        alpha,
                        beta,
                        node_idx_ref,
                    )
                    current_best_outcome_for_player = eval_score
                else:
                    for p_action_obj in player_possible_moves:
                        p_order = self.create_order(p_action_obj)
                        next_battle_state = self._simulate_one_turn_local(
                            current_battle_state, p_order, o_order
                        )
                        eval_score = self._minimax_score(
                            next_battle_state,
                            depth + 1,
                            True,
                            max_depth,
                            alpha,
                            beta,
                            node_idx_ref,
                        )
                        current_best_outcome_for_player = max(
                            current_best_outcome_for_player, eval_score
                        )
                        # Alpha for player's choice against this o_order

                min_eval = min(min_eval, current_best_outcome_for_player)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

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

    def minimax(self, battle: AbstractBattle) -> Tuple[Optional[BattleOrder], int]:
        """
        Chooses the best move for the current player using the Minimax algorithm.
        Simulates 1v1 scenarios, does not consider switches.
        A leaf node is where a Pokémon faints or max depth is reached.
        """
        # Create a backup of opponent's Pokémon data before enrichment
        battle = deepcopy(battle)
        self._enrich_opponent_pokemon(battle)

        max_depth = self.K
        node_idx_counter = {"count": 0}
        best_action_order: Optional[BattleOrder] = None

        max_overall_score = -float("inf")
        initial_player_moves = battle.available_moves

        # TODO: sort moves by more damage to prune more nodes, do it in _minimax_score too
        for p_action_obj in initial_player_moves:
            p_order = self.create_order(p_action_obj)

            # For this p_order, what's the worst score the opponent can force on us?
            # This means we simulate p_order against all opponent's responses,
            # and for each resulting state, the opponent (minimizing player) starts their turn.
            min_score_for_current_p_action = float("inf")

            # TODO: enrich opponent pokemon
            opponent_possible_moves = self._get_opponent_possible_moves(battle)

            if not opponent_possible_moves:  # Opponent is trapped or has no moves
                # Simulate only player's move, then it's opponent's turn (minimizer) at depth 1
                sim_after_p_order_only_battle = self._simulate_one_turn_local(
                    battle, p_order, None
                )
                current_action_score = self._minimax_score(
                    sim_after_p_order_only_battle,
                    1,  # Depth is 1 after first set of moves
                    False,  # Opponent's turn (minimizer)
                    max_depth,
                    -float("inf"),
                    float("inf"),
                    node_idx_counter,
                )
                min_score_for_current_p_action = current_action_score
            else:
                for o_action_obj in opponent_possible_moves:
                    o_order = self.create_order(o_action_obj)  # Opponent order

                    # Simulate both moves happening in this turn
                    sim_after_both_moves_battle = self._simulate_one_turn_local(
                        battle, p_order, o_order
                    )

                    # After both moves, it's conceptually our (player's) turn again for the next depth level (depth 1)
                    # However, the minimax structure is: max (min (max (...)))
                    # So, after player makes p_order, and opponent makes o_order, the resulting state
                    # is evaluated from the perspective of whose turn it is NEXT.
                    # If depth in _minimax_score means "remaining depth", then it's max_depth-1.
                    # If depth means "current depth", it starts at 0 (current state) or 1 (after first action).

                    # The call should be: from sim_after_both_moves_battle, it's player's turn (maximizer) at depth 1.
                    score = self._minimax_score(
                        sim_after_both_moves_battle,
                        1,  # Current depth is 1
                        True,  # It's Player's (Maximizer's) turn from this new state
                        max_depth,
                        -float("inf"),
                        float("inf"),  # Initial alpha beta for this sub-problem
                        node_idx_counter,
                    )
                    min_score_for_current_p_action = min(
                        min_score_for_current_p_action, score
                    )

            if min_score_for_current_p_action > max_overall_score:
                max_overall_score = min_score_for_current_p_action
                best_action_order = p_order

            print(p_action_obj, min_score_for_current_p_action)

        if (
            best_action_order is None and initial_player_moves
        ):  # Should pick one if moves were available
            best_action_order = self.create_order(
                initial_player_moves[0]
            )  # Fallback to first available move

        print(f"explored {node_idx_counter["count"]} nodes")
        return best_action_order, node_idx_counter["count"]

    def choose_max_damage_move(self, battle: Battle) -> BattleOrder:
        if battle.available_moves:
            best_move = max(
                battle.available_moves, key=lambda move: move.base_power * move.accuracy
            )
            return self.create_order(best_move)
        return self.choose_random_move(battle)
