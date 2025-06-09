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
from poke_env.environment.effect import Effect
from poke_env.environment.field import Field
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
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

        # -- Constants for battle state evaluation
        self.POKEMON_ALIVE = 30.0
        self.POKEMON_HP = 100.0
        self.USED_TERA = -75.0

        self.POKEMON_ATTACK_BOOST = 30.0
        self.POKEMON_DEFENSE_BOOST = 15.0
        self.POKEMON_SPECIAL_ATTACK_BOOST = 30.0
        self.POKEMON_SPECIAL_DEFENSE_BOOST = 15.0
        self.POKEMON_SPEED_BOOST = 30.0

        self.BOOST_MULTIPLIERS = {
            6: 3.3,
            5: 3.15,
            4: 3.0,
            3: 2.5,
            2: 2.0,
            1: 1.0,
            0: 0.0,
            -1: -1.0,
            -2: -2.0,
            -3: -2.5,
            -4: -3.0,
            -5: -3.15,
            -6: -3.3,
        }

        self.POKEMON_FROZEN = -40.0
        self.POKEMON_ASLEEP = -25.0
        self.POKEMON_PARALYZED = -25.0
        self.POKEMON_TOXIC = -30.0
        self.POKEMON_POISONED = -10.0
        self.POKEMON_BURNED = -25.0

        self.LEECH_SEED = -30.0
        self.SUBSTITUTE = 40.0
        self.CONFUSION = -20.0

        self.REFLECT = 20.0
        self.LIGHT_SCREEN = 20.0
        self.AURORA_VEIL = 40.0
        self.SAFEGUARD = 5.0
        self.TAILWIND = 7.0
        self.HEALING_WISH = 30.0

        self.STEALTH_ROCK = -10.0
        self.SPIKES = -7.0
        self.TOXIC_SPIKES = -7.0
        self.STICKY_WEB = -25.0

    # ----

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

    def _evaluate_state(self, battle: AbstractBattle) -> float:
        score = 0.0

        # TODO: it is for debugging purposes, remove it later
        dict_of_values = {
            "p1": dict(),
            "p2": dict(),
        }

        # --- Player's Side Evaluation ---
        if any(p.terastallized for p in battle.team.values()):
            score += self.USED_TERA

        for pokemon in battle.team.values():
            if pokemon.fainted:
                continue

            pkmn_score = self.POKEMON_ALIVE + (
                self.POKEMON_HP * pokemon.current_hp_fraction
            )

            # Status Effects
            if pokemon.status == Status.BRN:
                pkmn_score += self._evaluate_burn_score(pokemon)
            elif pokemon.status == Status.FRZ:
                pkmn_score += self.POKEMON_FROZEN
            elif pokemon.status == Status.SLP:
                pkmn_score += self.POKEMON_ASLEEP
            elif pokemon.status == Status.PAR:
                pkmn_score += self.POKEMON_PARALYZED
            elif pokemon.status == Status.TOX:
                pkmn_score += self._evaluate_poison_score(pokemon, self.POKEMON_TOXIC)
            elif pokemon.status == Status.PSN:
                pkmn_score += self._evaluate_poison_score(
                    pokemon, self.POKEMON_POISONED
                )

            if pokemon.item:
                pkmn_score += 10.0

            pkmn_score += self._evaluate_hazards_score(pokemon, battle)

            # Active Pokémon Effects
            if pokemon.active:
                # Boosts
                for stat, value in pokemon.boosts.items():
                    multiplier = self._get_boost_multiplier(value)
                    if stat in ["atk", "spa", "spe"]:
                        pkmn_score += multiplier * self.POKEMON_ATTACK_BOOST
                    elif stat in ["def", "spd"]:
                        pkmn_score += multiplier * self.POKEMON_DEFENSE_BOOST

                # Volatile Statuses
                if Effect.LEECH_SEED in pokemon.effects:
                    pkmn_score += self.LEECH_SEED
                if Effect.SUBSTITUTE in pokemon.effects:
                    pkmn_score += self.SUBSTITUTE
                if Effect.CONFUSION in pokemon.effects:
                    pkmn_score += self.CONFUSION

            dict_of_values["p1"][pokemon.species] = pkmn_score
            score += pkmn_score

        # --- Opponent's Side Evaluation (scores are subtracted) ---
        if any(p.terastallized for p in battle.opponent_team.values()):
            score -= self.USED_TERA

        # TODO: add values for unseen pokemon
        for pokemon in battle.opponent_team.values():
            if pokemon.fainted:
                continue

            pkmn_score = self.POKEMON_ALIVE + (
                self.POKEMON_HP * pokemon.current_hp_fraction
            )

            # Status Effects
            if pokemon.status == Status.BRN:
                pkmn_score += self._evaluate_burn_score(pokemon)
            elif pokemon.status == Status.FRZ:
                pkmn_score += self.POKEMON_FROZEN
            elif pokemon.status == Status.SLP:
                pkmn_score += self.POKEMON_ASLEEP
            elif pokemon.status == Status.PAR:
                pkmn_score += self.POKEMON_PARALYZED
            elif pokemon.status == Status.TOX:
                pkmn_score += self._evaluate_poison_score(pokemon, self.POKEMON_TOXIC)
            elif pokemon.status == Status.PSN:
                pkmn_score += self._evaluate_poison_score(
                    pokemon, self.POKEMON_POISONED
                )

            if pokemon.item:
                pkmn_score += 10.0

            pkmn_score += self._evaluate_hazards_score(pokemon, battle)

            if pokemon.active:
                # Boosts
                for stat, value in pokemon.boosts.items():
                    multiplier = self._get_boost_multiplier(value)
                    if stat in ["atk", "spa", "spe"]:
                        pkmn_score += multiplier * self.POKEMON_ATTACK_BOOST
                    elif stat in ["def", "spd"]:
                        pkmn_score += multiplier * self.POKEMON_DEFENSE_BOOST
                # Volatile Statuses
                if Effect.LEECH_SEED in pokemon.effects:
                    pkmn_score += self.LEECH_SEED
                if Effect.SUBSTITUTE in pokemon.effects:
                    pkmn_score += self.SUBSTITUTE
                if Effect.CONFUSION in pokemon.effects:
                    pkmn_score += self.CONFUSION

            dict_of_values["p2"][pokemon.species] = pkmn_score
            score -= pkmn_score

        # --- Team-Wide Side Conditions & Hazards ---
        for sc, val in battle.side_conditions.items():
            if sc == SideCondition.REFLECT:
                score += self.REFLECT
            elif sc == SideCondition.LIGHT_SCREEN:
                score += self.LIGHT_SCREEN
            elif sc == SideCondition.AURORA_VEIL:
                score += self.AURORA_VEIL
            elif sc == SideCondition.SAFEGUARD:
                score += self.SAFEGUARD
            elif sc == SideCondition.TAILWIND:
                score += self.TAILWIND
            # Hazards affect the whole team, so we evaluate their potential impact
            elif sc == SideCondition.STEALTH_ROCK:
                score += self.STEALTH_ROCK * len(
                    [p for p in battle.team.values() if not p.fainted]
                )
            elif sc == SideCondition.SPIKES:
                score += (
                    self.SPIKES
                    * val
                    * len([p for p in battle.team.values() if not p.fainted])
                )
            # TODO: add other conditions

        for sc, val in battle.opponent_side_conditions.items():
            if sc == SideCondition.REFLECT:
                score -= self.REFLECT
            elif sc == SideCondition.LIGHT_SCREEN:
                score -= self.LIGHT_SCREEN
            elif sc == SideCondition.AURORA_VEIL:
                score -= self.AURORA_VEIL
            elif sc == SideCondition.SAFEGUARD:
                score -= self.SAFEGUARD
            elif sc == SideCondition.TAILWIND:
                score -= self.TAILWIND
            elif sc == SideCondition.STEALTH_ROCK:
                score -= self.STEALTH_ROCK * len(
                    [p for p in battle.opponent_team.values() if not p.fainted]
                )
            elif sc == SideCondition.SPIKES:
                score -= (
                    self.SPIKES
                    * val
                    * len([p for p in battle.opponent_team.values() if not p.fainted])
                )

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

    def _enrich_opponent_team(self, battle: AbstractBattle):
        actual_opponent_team = set()
        for mon in battle.opponent_team.values():
            actual_opponent_team.add(mon.species)

        for previewed_mon in battle._teampreview_opponent_team:
            if previewed_mon.species not in actual_opponent_team:
                mon = Pokemon(gen=self.genNum, species=previewed_mon.species)
                mon.set_hp_status("100/100")
                # TODO: in the opponent team the names are not normalized
                # so for example we have "p2: Gengar" instead of "p2: gengar"
                # maybe with should reverse the normalization
                battle.opponent_team["p2: " + mon.species] = mon

        for mon in battle.opponent_team.values():
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
                        data["items"][0]["name"]
                        .lower()
                        .replace(" ", "")
                        .replace("-", "")
                    )

            if mon._terastallized_type is None:
                if data.get("tera"):
                    mon._terastallized_type = PokemonType.from_name(
                        data["tera"][0]["name"]
                    )

    def choose_best_action(
        self, battle: AbstractBattle
    ) -> Tuple[Optional[BattleOrder], int]:
        """
        Top-level function to choose the best action (move or switch) using Minimax.
        It iterates through all possible actions and uses helper functions to evaluate them.
        """
        # Create a single, enriched copy of the battle state to be used for all simulations this turn.
        sim_battle = deepcopy(battle)

        self._enrich_opponent_team(sim_battle)

        best_action_order = self.choose_random_move(battle)  # Default action
        node_idx_counter = {"count": 0}

        # The alpha value for the root of our decision tree (a MAX node).
        root_alpha = -float("inf")

        # 1. Evaluate staying in and using a move
        if sim_battle.available_moves:
            # Sort moves to check more promising ones first, potentially leading to more pruning.
            # A simple heuristic is to check high-power moves first.
            sorted_moves = sorted(
                sim_battle.available_moves,
                key=lambda m: m.base_power * m.accuracy,
                reverse=True,
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
                cnt = node_idx_counter["count"]
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

    def _get_boost_multiplier(self, boost_level: int) -> float:
        return self.BOOST_MULTIPLIERS.get(boost_level, 0.0)

    def _evaluate_poison_score(self, pokemon: Pokemon, base_score: float) -> float:
        # In poke-env, abilities are strings. We use 'in' for abilities like "Poison Heal"
        ability = pokemon.ability or ""
        if "poisonheal" in ability:
            return 15.0
        if any(
            a in ability
            for a in ["guts", "marvelscale", "quickfeet", "toxicboost", "magicguard"]
        ):
            return 10.0
        return base_score

    def _evaluate_burn_score(self, pokemon: Pokemon) -> float:
        ability = pokemon.ability or ""
        if any(a in ability for a in ["guts", "marvelscale", "quickfeet"]):
            return -2.0 * self.POKEMON_BURNED

        # Penalize physical attackers more heavily for burn
        physical_move_count = 0
        for move in pokemon.moves.values():
            if move.category == MoveCategory.PHYSICAL:
                physical_move_count += 1

        # This is a simplification. The Rust code compares calculated stats.
        # We check the number of physical moves as a proxy.
        return (physical_move_count / 4.0) * self.POKEMON_BURNED

    def _is_grounded(self, mon: Pokemon, battle: AbstractBattle):
        if Field.GRAVITY in battle.fields:
            return True
        elif mon.item == "ironball":
            return True
        elif mon.ability == "levitate":
            return False
        elif mon.ability is None and "levitate" in mon.possible_abilities:
            return False
        elif mon.item == "airballoon":
            return False
        elif mon.type_1 == PokemonType.FLYING or mon.type_2 == PokemonType.FLYING:
            return False
        elif Effect.MAGNET_RISE in mon.effects:
            return False
        return True

    def _evaluate_hazards_score(
        self, pokemon: Pokemon, battle: AbstractBattle
    ) -> float:
        """
        Calculates the score penalty from entry hazards for a single Pokémon.
        This version correctly uses battle.is_grounded().
        """
        # Determine which side's conditions to check
        if pokemon in battle.team.values():
            side_conditions = battle.side_conditions
        elif pokemon in battle.opponent_team.values():
            side_conditions = battle.opponent_side_conditions
        else:
            return 0.0  # Should not happen

        score = 0.0

        # Use the battle's method to correctly determine if the Pokemon is grounded
        is_grounded = self._is_grounded(pokemon, battle)

        # Heavy-Duty Boots negates damage-dealing entry hazards
        if pokemon.item == "heavydutyboots":
            if is_grounded and SideCondition.STICKY_WEB in side_conditions:
                score += self.STICKY_WEB
            return score

        # Magic Guard negates all hazard damage
        if pokemon.ability == "magicguard":
            return score

        # Calculate hazard damage
        if SideCondition.STEALTH_ROCK in side_conditions:
            # Stealth Rock damage depends on the Pokemon's type effectiveness against Rock
            rock_multiplier = PokemonType.ROCK.damage_multiplier(
                pokemon.type_1, pokemon.type_2, type_chart=self.gen.type_chart
            )
            # The base penalty is for 1x effective, scaled by the multiplier
            score += self.STEALTH_ROCK * rock_multiplier

        if is_grounded:
            if SideCondition.SPIKES in side_conditions:
                # Assumes a simple scaling with layers. 1 layer = -7.0
                score += side_conditions.get(SideCondition.SPIKES, 0) * self.SPIKES
            if SideCondition.TOXIC_SPIKES in side_conditions:
                # This can be expanded to check for poison/steel types that remove it
                score += self.TOXIC_SPIKES
            if SideCondition.STICKY_WEB in side_conditions:
                score += self.STICKY_WEB

        return score
