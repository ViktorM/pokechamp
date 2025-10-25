import ast
from copy import copy, deepcopy
from collections import deque
from dataclasses import dataclass
import datetime
import json
import os
import random
import sys

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.player.player import Player, BattleOrder
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from poke_env.environment.move import Move
import time
import json
import hashlib
import re
from poke_env.data.gen_data import GenData
from pokechamp.gpt_player import GPTPlayer
from pokechamp.llama_player import LLAMAPlayer
from pokechamp.openrouter_player import OpenRouterPlayer
from pokechamp.gemini_player import GeminiPlayer
from pokechamp.ollama_player import OllamaPlayer
from pokechamp.data_cache import (
    get_cached_move_effect,
    get_cached_pokemon_move_dict,
    get_cached_ability_effect,
    get_cached_pokemon_ability_dict,
    get_cached_item_effect,
    get_cached_pokemon_item_dict,
    get_cached_pokedex
)
import logging
from pokechamp.minimax_optimizer import (
    get_minimax_optimizer,
    initialize_minimax_optimization,
    fast_battle_evaluation,
    create_battle_state_hash,
    OptimizedSimNode,
    CachedChildNode,
    mk_ttkey,
    canonical_action
)
from poke_env.player.local_simulation import LocalSim, SimNode
from difflib import get_close_matches
from pokechamp.prompts import get_number_turns_faint, get_status_num_turns_fnt, state_translate, get_gimmick_motivation
from pokechamp.gen1_quirks import Gen1Quirks


DEBUG = False
logger = logging.getLogger(__name__)


class LLMPlayer(Player):
    def __init__(self,
                 battle_format,
                 api_key="",
                 backend="gpt-4o",
                 temperature=0.3,
                 prompt_algo="io",
                 log_dir=None,
                 team=None,
                 save_replays=None,
                 account_configuration=None,
                 server_configuration=None,
                 K=3,
                 _use_strat_prompt=False,
                 prompt_translate: Callable=state_translate,
                 device=0,
                 llm_backend=None,
                 log_level=None,
                 move_time_limit_s: float = None,
                 max_tokens: int = 300,
                 reasoning_effort: str = "low",
                 temp_action: float = None,
                 mt_action: int = None,
                 temp_expand: float = None,
                 mt_expand: int = None,
                 profile: bool = False
                 ):

        super().__init__(battle_format=battle_format,
                         team=team,
                         save_replays=save_replays,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration,
                         log_level=log_level)

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action : Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # Momentum tracking per battle
        self._momentum: Dict[str, Dict] = {}  # battle_tag -> momentum metrics
        self.backend = backend
        self.temperature = temperature
        self.log_dir = log_dir
        self.api_key = api_key
        self.prompt_algo = prompt_algo
        self.reasoning_effort = reasoning_effort
        # Minimax leaf evaluation config (can be overridden)
        self.minimax_llm_weight_midgame = 0.4  # 40% LLM in midgame  
        self.minimax_llm_weight_endgame = 0.8  # 80% LLM in endgame

        # Two-tier temperature and token settings with CLI override support
        # Tier 1: Action/Decision prompts (cold, structured JSON selection)
        self.temp_action = temp_action if temp_action is not None else 0.0
        self.mt_action = mt_action if mt_action is not None else 120

        # Tier 2: Expansion/Reasoning prompts (warmer, idea generation)
        self.temp_expand = temp_expand if temp_expand is not None else temperature
        self.mt_expand = mt_expand if mt_expand is not None else max_tokens

        # Legacy compatibility - map to expand tier by default
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Profiling flag
        self.profile = profile

        self.gen = GenData.from_format(battle_format)
        self.genNum = self.gen.gen
        self.prompt_translate = prompt_translate

        self.strategy_prompt = ""
        self.team_str = team
        self.use_strat_prompt = _use_strat_prompt

        # Use cached data instead of loading files repeatedly
        self.move_effect = get_cached_move_effect()
        # only used in old prompting method, replaced by statistcal sets data
        self.pokemon_move_dict = get_cached_pokemon_move_dict()
        self.ability_effect = get_cached_ability_effect()
        # only used is old prompting method
        self.pokemon_ability_dict = get_cached_pokemon_ability_dict()
        self.item_effect = get_cached_item_effect()
        # unused
        # with open(f"./poke_env/data/static/items/gen8pokemon_item_dict.json", "r") as f:
        #     self.pokemon_item_dict = json.load(f)
        self.pokemon_item_dict = get_cached_pokemon_item_dict()
        self._pokemon_dict = get_cached_pokedex(self.gen.gen)

        self.last_plan = ""

        if llm_backend is None:
            print(f"Initializing backend: {backend}")  # Debug logging
            if backend.startswith('ollama/'):
                # Ollama models - extract model name after 'ollama/'
                model_name = backend.replace('ollama/', '')
                print(f"Using Ollama with model: {model_name}")
                self.llm = OllamaPlayer(model=model_name, device=device)
            elif backend.startswith(('openai/', 'anthropic/', 'google/', 'meta/', 'mistral/', 'cohere/', 'perplexity/', 'deepseek/', 'microsoft/', 'nvidia/', 'huggingface/', 'together/', 'replicate/', 'fireworks/', 'localai/', 'vllm/', 'sagemaker/', 'vertex/', 'bedrock/', 'azure/', 'custom/', 'x-ai/', 'xai/')):
                # OpenRouter supports hundreds of models from various providers including xAI
                self.llm = OpenRouterPlayer(self.api_key)
            elif 'gpt' in backend:
                # Direct OpenAI API; allow service tier via env
                self.llm = GPTPlayer(self.api_key, service_tier='priority')
            elif 'llama' == backend:
                self.llm = LLAMAPlayer(device=device)
            elif 'gemini' in backend:
                self.llm = GeminiPlayer(self.api_key)
            else:
                raise NotImplementedError('LLM type not implemented:', backend)
        else:
            self.llm = llm_backend

        self.llm_value = self.llm
        self.K = K      # for minimax, SC, ToT
        self.use_optimized_minimax = True  # Enable optimized minimax by default
        self._minimax_initialized = False
        # Configuration for time optimization
        self.use_damage_calc_early_exit = True  # Use damage calculator to exit early when advantageous
        self.use_llm_value_function = True  # Use LLM for leaf node evaluation (vs fast heuristic)
        self.max_depth_for_llm_eval = 2  # Only use LLM evaluation for shallow depths to save time
        # Per-move time budget (seconds). Defaults to 8s if not provided
        self.move_time_limit_s = move_time_limit_s if move_time_limit_s is not None else 8.0
        self.max_tokens = int(max(1, max_tokens))

        # Metrics and KPIs
        self._move_metrics = []  # per-move rows
        self._metrics_file_initialized = False
        self._llm_calls_per_turn = {}  # battle_tag -> {turn: count}
        self._timeout_avoided = {}     # battle_tag -> count
        self._timeouts = {}            # battle_tag -> count
        self._last_action_label = {}   # battle_tag -> last action label for oscillation
        self._oscillation = {}         # battle_tag -> count
        self._progress_scores = {}     # battle_tag -> [scores]

        # Partial trap state tracking (Gen1)
        self._trap_states = {}  # battle_tag -> trap state dict

    def _recover_action_from_text(self, raw_text: str, battle: Battle):
        """Attempt to recover a legal action from non-JSON text.

        Strategy: find a legal move id or switch species mentioned in the text.
        Returns a BattleOrder or None.
        """
        try:
            text = (raw_text or '').lower().replace(' ', '').replace('-', '')
            for move in (battle.available_moves or []):
                mid = move.id.lower().replace(' ', '').replace('-', '')
                if mid and mid in text:
                    return self.create_order(move)
            for mon in (battle.available_switches or []):
                sid = (mon.species or '').lower().replace(' ', '').replace('-', '')
                if sid and sid in text:
                    return self.create_order(mon)
        except Exception:
            pass
        return None

    def _robust_value(self, child_vals, opp_priors=None, eta=0.25):
        """
        Compute robust value for action considering opponent replies.
        
        Uses mean-of-worst-two for stability with small K (more monotone, less noise).

        Args:
            child_vals: List of values for different opponent replies
            opp_priors: Optional prior probabilities for opponent actions (unused with mean-of-worst-two)
            eta: Legacy parameter, kept for compatibility

        Returns:
            Mean of two worst opponent replies (or min if only 1 reply)
        """
        if not child_vals:
            return float('-inf')
        
        if len(child_vals) == 1:
            return child_vals[0]
        
        # Sort and take mean of worst two (more stable than eta-mixing)
        sorted_vals = sorted(child_vals)
        worst_two = sorted_vals[:2]
        return sum(worst_two) / len(worst_two)

    def _should_use_minimax(self, battle: Battle, dmg_calc_turns: float, sim=None) -> bool:
        """
        Deterministic heuristic to decide between damage calc and minimax.
        Much stricter now - only skips minimax in very clear endgame situations.

        Args:
            battle: Current battle state
            dmg_calc_turns: Turns to KO opponent with best damage calc move

        Returns:
            True if should use minimax (default), False if damage calc is sufficient
        """
        # Always use minimax if fainted
        if battle.active_pokemon.fainted:
            return True

        try:
            # Get opponent's best TTK against us
            _, their_best_ttk = self.estimate_matchup(
                sim, battle,
                battle.opponent_active_pokemon,
                battle.active_pokemon,
                is_opp=True
            )
        except:
            their_best_ttk = float('inf')

        # Check if we move first (speed advantage)
        try:
            we_move_first = battle.active_pokemon.stats['spe'] > battle.opponent_active_pokemon.stats['spe']
        except:
            we_move_first = False

        # Calculate effective TTK accounting for speed
        our_effective_ttk = dmg_calc_turns + (0 if we_move_first else 1)

        # Get our best damage move type for pivot checking
        best_move_type = None
        if battle.available_moves:
            damages = []
            for move in battle.available_moves:
                if move.base_power > 0:
                    dmg = move.base_power * move.accuracy
                    if battle.opponent_active_pokemon:
                        dmg *= battle.opponent_active_pokemon.damage_multiplier(move)
                    damages.append((dmg, move.type))
            if damages:
                damages.sort(key=lambda x: x[0], reverse=True)
                best_move_type = damages[0][1]

        # Check if opponent has healthy pivots that resist our best move
        opp_can_pivot = False
        if best_move_type and battle.opponent_team:
            for mon in battle.opponent_team.values():
                if not mon.fainted and mon != battle.opponent_active_pokemon:
                    hp_frac = mon.current_hp_fraction
                    if hp_frac is None or hp_frac > 0.6:
                        # Check type effectiveness
                        type_mult = mon.damage_multiplier(best_move_type)
                        if type_mult <= 0.5:  # Resists or immune
                            opp_can_pivot = True
                            break

        # Check game phase
        our_fainted = sum(1 for p in battle.team.values() if p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        endgame = (len(battle.team) - our_fainted <= 2) or (len(battle.opponent_team) - opp_fainted <= 2)

        # Check if we're at a disadvantage
        hazards_against_us = bool(battle.side_conditions)
        fainted_disadvantage = our_fainted > opp_fainted

        # Much stricter conditions for skipping minimax
        clear_edge = (dmg_calc_turns <= 2) and (our_effective_ttk < their_best_ttk - 1)

        # Only skip minimax in very clear endgame situations with no complications
        if endgame and clear_edge and not opp_can_pivot and not hazards_against_us and not fainted_disadvantage:
            print(f"Using damage calculator (clear endgame advantage: {our_effective_ttk} vs {their_best_ttk} turns)")
            return False

        # Default to minimax for all complex positions
        return True

    def legalize_choice(self, battle: Battle, choice: dict) -> Optional[BattleOrder]:
        """
        Robust normalizer to map LLM output to legal BattleOrder.

        Handles nickname/species/case mismatches.

        Args:
            battle: Current battle state
            choice: Dict like {"move": "X"} or {"switch": "Y"}

        Returns:
            BattleOrder if legal, None otherwise
        """
        try:
            # Normalize move selection
            if "move" in choice:
                want = choice["move"].replace(" ", "").replace("-", "").lower()
                for mv in battle.available_moves:
                    mv_id = mv.id.replace(" ", "").replace("-", "").lower()
                    if mv_id == want:
                        # Check for gimmick flags
                        tera = choice.get("tera") in (True, "true", "True", 1)
                        dmax = choice.get("dmax") in (True, "true", "True", 1)
                        return self.create_order(mv, terastallize=tera, dynamax=dmax)
                
                # Log normalization failure
                available_ids = [mv.id for mv in battle.available_moves]
                print(f"⚠️  Move '{choice['move']}' not in legal moves: {available_ids}")
                return None

            # Normalize switch selection
            if "switch" in choice:
                want = choice["switch"].replace(" ", "").replace("-", "").lower()
                # Try species match (not nickname)
                for mon in battle.available_switches:
                    species = (mon.species or "").replace(" ", "").replace("-", "").lower()
                    if species == want:
                        return self.create_order(mon)
                
                # Log normalization failure
                available_species = [mon.species for mon in battle.available_switches]
                print(f"⚠️  Switch '{choice['switch']}' not in legal switches: {available_species}")
                return None

        except Exception as e:
            print(f"⚠️  legalize_choice failed: {e}")
            return None

        return None

    def _safe_default(self, battle: Battle) -> BattleOrder:
        """
        Robust fallback when no action can be determined.
        
        Priority:
        1. Max-damage legal move
        2. First legal move
        3. First legal switch
        4. Random move (last resort)
        """
        # Try max damage move first
        md = self.choose_max_damage_move(battle)
        if md is not None:
            return md

        # Try any legal move
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])

        # Try any legal switch
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])

        # If server sent wait-state, higher layer should have early-returned;
        # otherwise do nothing (Showdown will ignore or we pick random)
        return self.choose_random_move(battle)

    def _finalize_action(self, selected: Optional[BattleOrder], raw_text: str, battle: Battle) -> BattleOrder:
        """
        Finalize action selection with robust fallbacks.

        Args:
            selected: The action selected by LLM/minimax (may be None)
            raw_text: Raw LLM output for text recovery
            battle: Current battle state

        Returns:
            A valid BattleOrder (guaranteed non-None)
        """
        # If we have a valid action, use it
        if selected is not None:
            return selected

        # Try to recover from raw text
        recovered = self._recover_action_from_text(raw_text, battle)
        if recovered is not None:
            return recovered

        # Use safe default as last resort
        return self._safe_default(battle)

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=None, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, llm=None, reasoning_effort=None) -> str:
        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort
        if temperature is None:
            temperature = self.temperature

        # Only pass reasoning_effort to GPTPlayer (supports it for o3-mini, o4-mini, gpt-5 models)
        llm_to_use = llm if llm is not None else self.llm
        if hasattr(llm_to_use, '__class__') and llm_to_use.__class__.__name__ == 'GPTPlayer':
            output, _ = llm_to_use.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions, reasoning_effort=reasoning_effort)
        else:
            # For OpenRouterPlayer, GeminiPlayer, etc. - don't pass reasoning_effort
            output, _ = llm_to_use.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions)
        return output

    def check_all_pokemon(self, pokemon_str: str) -> Pokemon:
        valid_pokemon = None
        if pokemon_str in self._pokemon_dict:
            valid_pokemon = pokemon_str
        else:
            closest = get_close_matches(pokemon_str, self._pokemon_dict.keys(), n=1, cutoff=0.8)
            if len(closest) > 0:
                valid_pokemon = closest[0]
        if valid_pokemon is None:
            return None
        pokemon = Pokemon(species=pokemon_str, gen=self.genNum)
        return pokemon

    def _update_trap_state(self, battle: AbstractBattle):
        """Update partial trap state based on battle conditions."""
        battle_tag = battle.battle_tag if hasattr(battle, 'battle_tag') else str(battle)

        # Initialize if needed
        if battle_tag not in self._trap_states:
            self._trap_states[battle_tag] = {
                'active': False,
                'source': None,
                'move': None,
                'turns': 0
            }

        trap_state = self._trap_states[battle_tag]

        # Check if we're trapped
        if battle.trapped:
            # We're currently trapped
            if not trap_state['active']:
                # Just got trapped
                trap_state['active'] = True
                trap_state['turns'] = 1
                if battle.opponent_active_pokemon:
                    trap_state['source'] = battle.opponent_active_pokemon.species
                    # Try to identify the move from opponent's known moves
                    trap_moves = ['wrap', 'bind', 'firespin', 'clamp']
                    for move in battle.opponent_active_pokemon.moves.values():
                        if move.id in trap_moves:
                            trap_state['move'] = move.id
                            break
            else:
                # Still trapped
                trap_state['turns'] += 1
        else:
            # Not trapped - reset if we were
            if trap_state['active']:
                trap_state['active'] = False
                trap_state['source'] = None
                trap_state['move'] = None
                trap_state['turns'] = 0

        return trap_state

    def choose_move(self, battle: AbstractBattle):
        sim = LocalSim(battle, 
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
                    prompt_translate=self.prompt_translate
        )
        if battle.turn <=1 and self.use_strat_prompt:
            self.strategy_prompt = sim.get_llm_system_prompt(self.format, self.llm, team_str=self.team_str, model='gpt-4o-2024-05-13')

        # Update trap state
        trap_state = self._update_trap_state(battle)

        # Handle trapped Pokemon (e.g., by Wrap in Gen 1) - no moves or switches available
        if (battle.trapped or (len(battle.available_moves) == 0 and len(battle.available_switches) == 0)) and not battle.active_pokemon.fainted:
            # Pokemon is trapped and cannot do anything - return default order
            print(f"Pokemon {battle.active_pokemon.species} is trapped and cannot move or switch")
            return self.choose_default_move()

        # Type-suicide guard: force switch if 2x weak and likely to die
        if not battle.trapped and len(battle.available_switches) > 0:
            try:
                belief = self._get_belief_topk(battle, sim)
                opp_topk = belief.get("opp_move_topk", [])
                if opp_topk:
                    top_move_id, top_prob = opp_topk[0]
                    if top_prob > 0.35:
                        top_move = Move(top_move_id, gen=sim.gen.gen)
                        # Check if we're 2x weak to this move
                        my_mult = battle.active_pokemon.damage_multiplier(top_move.type)
                        if my_mult >= 2.0:
                            # Estimate danger
                            turns_opp_ko = get_number_turns_faint(
                                battle.opponent_active_pokemon, top_move, battle.active_pokemon,
                                sim, boosts1=battle.opponent_active_pokemon._boosts.copy(),
                                boosts2=battle.active_pokemon._boosts.copy()
                            )
                            danger = 1.0 / max(1, turns_opp_ko)
                            if danger > 0.6:  # Very likely to die
                                # Check if we can KO back
                                best_move, best_turns = self.estimate_matchup(sim, battle, battle.active_pokemon, battle.opponent_active_pokemon)
                                if best_turns > 1:  # Can't OHKO back
                                    # Force switch to best resist/immune
                                    best_switch = None
                                    best_resist = 999
                                    for sw_mon in battle.available_switches:
                                        resist = sw_mon.damage_multiplier(top_move.type)
                                        if resist < best_resist:
                                            best_resist = resist
                                            best_switch = sw_mon

                                    if best_switch and best_resist < 1.0:
                                        print(f"Type-suicide guard: switching to {best_switch.species} (resists {top_move.id})")
                                        return self.create_order(best_switch)
            except Exception:
                pass

        # Gen1 anti-wrap switch recommendation (only if not trapped)
        if sim.gen.gen == 1 and not battle.trapped and len(battle.available_switches) > 0:
            switch_target = Gen1Quirks.get_switch_recommendation(battle)
            if switch_target:
                for pokemon in battle.available_switches:
                    if pokemon.species.lower() == switch_target.lower():
                        print(f"Gen1 anti-wrap: switching to {pokemon.species}")
                        return self.create_order(pokemon)

        # HACK/TEMPORARY: Gen1 Anti-Wrap Strategy - DISABLED due to repeated crashes
        # This hack was causing TypeErrors when stats were None
        # The minimax/io should handle wrap strategy through proper evaluation
        pass

        if battle.active_pokemon:
            if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
                next_action = BattleOrder(battle.available_switches[0])
                return next_action
            elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1 and len(battle.available_switches) == 0:
                return self.choose_max_damage_move(battle)
        elif len(battle.available_moves) <= 1 and len(battle.available_switches) == 0:
            return self.choose_max_damage_move(battle)

        # Pass trap state to the prompt generator
        if hasattr(sim, 'set_trap_state'):
            sim.set_trap_state(trap_state)
        system_prompt, state_prompt, state_action_prompt = sim.state_translate(battle) # add lower case
        moves = [move.id for move in battle.available_moves]
        switches = [pokemon.species for pokemon in battle.available_switches]
        actions = [moves, switches]

        # Determine gimmick availability as structured flags (not standalone actions)
        tera_available = False
        dmax_available = False
        if 'pokellmon' not in self.ps_client.account_configuration.username:
            if battle._data.gen == 9 and battle.can_tera:
                tera_available = True
            if battle._data.gen == 8 and battle.can_dynamax:
                dmax_available = True

        # Build gimmick flags for JSON output (coupled with moves, not standalone)
        gimmick_flags = ''
        if tera_available:
            gimmick_flags += ', "tera": true|false'
        if dmax_available:
            gimmick_flags += ', "dmax": true|false'

        if battle.active_pokemon.fainted or (len(battle.available_moves) == 0 and len(battle.available_switches) > 0 and not battle.trapped):
            # Pokemon is fainted or can't move but can switch (not trapped)
            constraint_prompt_io = '''Choose the most suitable pokemon to switch. Pick exactly one action. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>", "why":"<=20 tokens (optional)"}. If low on budget, omit "why".\n'''
            constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
        elif len(battle.available_switches) == 0 or battle.trapped:
            constraint_prompt_io = f'''Choose the best action. Pick exactly one action. Your output MUST be a JSON like: {{"move":"<move_name>"{gimmick_flags}, "why":"<=20 tokens (optional)"}}. If low on budget, omit "why". Set tera/dmax to true only if beneficial.\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move>", "target":"<move_name>"}, ..., "option_k":{"action":"<move>", "target":"<move_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move>", "target":"<move_name>"}\n'''
        else:
            constraint_prompt_io = f'''Choose the best action. Pick exactly one action. Your output MUST be a JSON like: {{"move":"<move_name>"{gimmick_flags}, "why":"<=20 tokens (optional)"}} or {{"switch":"<switch_pokemon_name>", "why":"<=20 tokens (optional)"}}. If low on budget, omit "why". Set tera/dmax to true only if beneficial.\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}\n'''

        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io
        state_prompt_cot = state_prompt + state_action_prompt + constraint_prompt_cot
        state_prompt_tot_1 = state_prompt + state_action_prompt + constraint_prompt_tot_1
        state_prompt_tot_2 = state_prompt + state_action_prompt + constraint_prompt_tot_2

        retries = 5
        # Chain-of-thought
        if self.prompt_algo == "io":
            return self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions)

        # Improved IO with numbered contract and early gate
        elif self.prompt_algo == "io2":
            return self.io2(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions)

        # Self-consistency with k = 3
        elif self.prompt_algo == "sc":
            # Per-move timeout accounting for SC
            start_time = time.time()
            try:
                return self.sc(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim)
            except Exception as e:
                print(f'SC failed with error: {e}')
                # Fallback to damage calculator
                move, _ = self.dmg_calc_move(battle)
                if move is not None:
                    progress_score = self._record_progress(battle)
                    self._record_metric(battle, start_time, json_ok=False, fallback_reason='sc_exception', 
                                        action_kind='move', action_label=str(getattr(move, 'order', '')), 
                                        max_tokens=self.max_tokens, tokens_prompt=0, tokens_completion=0, 
                                        near_timeout=False, progress_score=progress_score)
                    return move

                # Last resort
                progress_score = self._record_progress(battle)
                self._record_metric(battle, start_time, json_ok=False, fallback_reason='sc_exception_maxdmg', 
                                    action_kind='max_damage', action_label='', max_tokens=self.max_tokens, 
                                    tokens_prompt=0, tokens_completion=0, near_timeout=False, 
                                    progress_score=progress_score)
                return self.choose_max_damage_move(battle)

        # Single-call rank-and-pick
        elif self.prompt_algo == "rank":
            return self.rank_and_pick(retries, system_prompt, state_prompt, state_action_prompt, battle, sim, actions=actions)

        # Batched IO ranker (scores all actions in one call)
        elif self.prompt_algo == "ranker":
            start_time = time.time()
            ranked_actions = self.rank_actions_batched(battle, sim=sim, start_time=start_time)

            if ranked_actions:
                # Return top-1 action
                best_action, best_score = ranked_actions[0]
                # Only print in debug mode (if logger level is DEBUG)
                if self.logger.level <= logging.DEBUG:
                    print(f"[Ranker] Selected {best_action} with score {best_score:.1f}")
                return best_action
            else:
                # Fallback to damage calculator
                if self.logger.level <= logging.DEBUG:
                    print("[Ranker] Failed, using damage calculator")
                move, _ = self.dmg_calc_move(battle)
                return move

        # Tree of thought, k = 3
        elif self.prompt_algo == "tot":
            llm_output1 = ""
            next_action = None

            for i in range(retries):
                try:
                    llm_output1 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_1,
                                               model=self.backend,
                                               temperature=self.temp_expand,
                                               max_tokens=self.mt_expand,
                                               json_format=True)
                    break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if llm_output1 == "":
                return self.choose_max_damage_move(battle)

            for i in range(retries):
                try:
                    llm_output2 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_2.replace("[OPTIONS]", llm_output1),
                                               model=self.backend,
                                               temperature=self.temp_expand,
                                               max_tokens=self.mt_expand,
                                               json_format=True)

                    next_action = self.parse_new(llm_output2, battle, sim)
                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(json.dumps({"turn": battle.turn,
                                            "system_prompt": system_prompt,
                                            "user_prompt1": state_prompt_tot_1,
                                            "user_prompt2": state_prompt_tot_2,
                                            "llm_output1": llm_output1,
                                            "llm_output2": llm_output2,
                                            "battle_tag": battle.battle_tag
                                            }) + "\n")
                    if next_action is not None:
                        break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            return next_action

        elif self.prompt_algo == "minimax":
            start_time = time.time()

            if not self._ensure_minimax_ready(battle):
                logger.warning("Minimax optimizer disabled, falling back to damage calculator")
                return self._minimax_fallback(battle, start_time, reason='minimax_not_ready')

            action = self.tree_search_optimized(
                retries, battle, sim=sim, return_opp=False, start_time=start_time
            )

            progress_score = self._record_progress(battle)
            elapsed = time.time() - start_time
            self._record_metric(
                battle,
                start_time,
                json_ok=True,
                fallback_reason='',
                action_kind='minimax',
                action_label=str(action.message if hasattr(action, 'message') else action.order),
                max_tokens=self.max_tokens,
                tokens_prompt=0,
                tokens_completion=0,
                near_timeout=(elapsed > self.move_time_limit_s - 2.0),
                progress_score=progress_score,
            )
            return action
        
    def _init_metrics_file(self):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, 'metrics.csv')
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(
                    'battle_tag,turn,backend,algorithm,player_role,player_username,latency_ms,json_ok,'
                    'fallback_reason,available_moves,available_switches,'
                    'action_kind,action_label,max_tokens,tokens_prompt,tokens_completion,'
                    'near_timeout,progress_score\n'
                )
        self._metrics_file_initialized = True

    def _record_metric(self, battle: Battle, start_time: float, json_ok: bool, fallback_reason: str, action_kind: str, action_label: str, max_tokens: int, tokens_prompt: int, tokens_completion: int, near_timeout: bool, progress_score: int):
        latency_ms = int((max(0.0, time.time() - start_time)) * 1000)
        avail_moves = len(battle.available_moves) if getattr(battle, 'available_moves', None) is not None else 0
        avail_switches = len(battle.available_switches) if getattr(battle, 'available_switches', None) is not None else 0
        row = [
            getattr(battle, 'battle_tag', ''),
            getattr(battle, 'turn', 0),
            self.backend,
            self.prompt_algo,
            getattr(battle, 'player_role', ''),  # p1 or p2
            self.username,  # player_username
            latency_ms,
            int(bool(json_ok)),
            fallback_reason,
            avail_moves,
            avail_switches,
            action_kind,
            action_label,
            max_tokens,
            tokens_prompt,
            tokens_completion,
            int(near_timeout),
            progress_score,
        ]
        self._move_metrics.append(row)
        if self.log_dir:
            if not self._metrics_file_initialized:
                self._init_metrics_file()
            try:
                with open(os.path.join(self.log_dir, 'metrics.csv'), 'a') as f:
                    f.write(','.join(map(str, row)) + "\n")
            except Exception:
                pass

    def _bump_llm_calls(self, battle: Battle):
        bt = getattr(battle, 'battle_tag', '')
        turn = getattr(battle, 'turn', 0)
        self._llm_calls_per_turn.setdefault(bt, {})
        self._llm_calls_per_turn[bt][turn] = self._llm_calls_per_turn[bt].get(turn, 0) + 1

    def _record_action_label(self, battle: Battle, action_kind: str, action_label: str):
        bt = getattr(battle, 'battle_tag', '')
        last = self._last_action_label.get(bt)
        if last is not None and last == (action_kind, action_label) and action_kind == 'switch':
            self._oscillation[bt] = self._oscillation.get(bt, 0) + 1
        self._last_action_label[bt] = (action_kind, action_label)

    def _record_progress(self, battle: Battle):
        try:
            score = int(self._get_fast_heuristic_evaluation(battle))
        except Exception:
            score = 50
        bt = getattr(battle, 'battle_tag', '')
        self._progress_scores.setdefault(bt, []).append(score)
        return score

    def _record_timeout(self, battle: Battle, avoided: bool):
        bt = getattr(battle, 'battle_tag', '')
        if avoided:
            self._timeout_avoided[bt] = self._timeout_avoided.get(bt, 0) + 1
        else:
            self._timeouts[bt] = self._timeouts.get(bt, 0) + 1

    def _battle_finished_callback(self, battle: AbstractBattle):
        # Aggregate KPIs per battle when finished
        bt = getattr(battle, 'battle_tag', '')
        moves = [r for r in self._move_metrics if r[0] == bt]
        if not moves:
            return
        # Latency is now at index 4 (after battle_tag, turn, backend, algorithm)
        # Use try-except to handle both old format (index 3) and new format (index 4)
        try:
            latencies = [int(r[4]) for r in moves]
        except (IndexError, ValueError):
            # Fallback for old format or corrupted data
            try:
                latencies = [int(r[3]) for r in moves]
            except (IndexError, ValueError):
                latencies = [0]
        avg_ms = sum(latencies) / len(latencies) if latencies else 0
        llm_calls = sum(self._llm_calls_per_turn.get(bt, {}).values()) if bt in self._llm_calls_per_turn else 0
        timeouts = self._timeouts.get(bt, 0)
        avoided = self._timeout_avoided.get(bt, 0)
        osc = self._oscillation.get(bt, 0)
        prog = self._progress_scores.get(bt, [])
        prog_trend = (prog[-1] - prog[0]) if len(prog) >= 2 else 0
        
        # Calculate additional useful statistics
        json_errors = sum(1 for r in moves if not r[7])  # json_ok is at index 7
        switches = sum(1 for r in moves if r[9] == 'switch')  # action_kind is at index 9
        
        # Calculate belief entropy trend (placeholder for now - can be enhanced later)
        belief_entropy_trend = 0  # TODO: Calculate from Bayesian predictor if available
        
        if self.log_dir:
            try:
                with open(os.path.join(self.log_dir, 'metrics_summary.csv'), 'a') as f:
                    f.write(','.join(map(str, [
                        bt,
                        getattr(battle, 'player_role', ''),  # p1 or p2
                        self.backend,
                        self.prompt_algo,
                        self.username,  # player_username
                        int(battle.won),
                        getattr(battle, 'turn', 0),  # turns - actual battle length
                        len(moves),  # total_moves - number of move records logged
                        f"{avg_ms:.1f}",  # avg_latency
                        llm_calls,
                        json_errors,
                        switches,
                        timeouts,
                        avoided,
                        osc,
                        prog_trend,
                        belief_entropy_trend,
                    ])) + "\n")
            except Exception:
                pass

    def io(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle: Battle, sim, dont_verify=False, actions=None):
        next_action = None
        cot_prompt = 'In fewer than 3 sentences, let\'s think step by step:'
        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io + cot_prompt

        # Stricter JSON handling with bounded retries and sanitization
        def _parse_llm_json(s: str):
            # Strip common wrappers/fences and trailing commas
            s = s.strip()
            if s.startswith('```'):
                s = s.strip('`')
                # Remove potential language hint like ```json
                if s.startswith('json'):
                    s = s[4:]
            s = s.strip()
            # Attempt direct parse
            try:
                return json.loads(s)
            except Exception:
                # Try to extract first JSON object
                start = s.find('{')
                end = s.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(s[start:end+1])
                    except Exception:
                        pass
            raise ValueError('Invalid JSON from LLM')

        # Per-move timeout start and accounting
        start_time = time.time()
        near_timeout_threshold = max(0.0, self.move_time_limit_s - 2.0)
        max_tokens = 300
        prev_prompt_tokens = getattr(self.llm, 'prompt_tokens', 0)
        prev_completion_tokens = getattr(self.llm, 'completion_tokens', 0)

        for i in range(retries):
            self._bump_llm_calls(battle)
            # Timeout fallback: choose deterministic damage move
            if time.time() - start_time > self.move_time_limit_s:
                print('LLM timeout, using damage calculator fallback')
                move, _ = self.dmg_calc_move(battle)
                if move is not None:
                    progress_score = self._record_progress(battle)
                    self._record_timeout(battle, avoided=False)
                    self._record_metric(battle, start_time, json_ok=False, fallback_reason='timeout', action_kind='move', action_label=str(getattr(move, 'order', '')), max_tokens=self.max_tokens,
                                        tokens_prompt=0, tokens_completion=0, near_timeout=True, progress_score=progress_score)
                    return move
                progress_score = self._record_progress(battle)
                self._record_timeout(battle, avoided=False)
                self._record_metric(battle, start_time, json_ok=False, fallback_reason='timeout', action_kind='max_damage', action_label='', max_tokens=self.max_tokens,
                                    tokens_prompt=0, tokens_completion=0, near_timeout=True, progress_score=progress_score)
                return self.choose_max_damage_move(battle)

            try:
                llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                            user_prompt=state_prompt_io,
                                            model=self.backend,
                                            temperature=self.temperature,
                                            max_tokens=self.max_tokens,
                                            # stop=["reason"],
                                            json_format=True,
                                            actions=actions)

                if DEBUG:
                    print(f"Raw LLM output: {llm_output}")
                llm_action_json = _parse_llm_json(llm_output)
                if DEBUG:
                    print(f"Parsed JSON: {llm_action_json}")
                next_action = None

                # Check for gimmicks as structured flags (new format) or standalone keys (old format)
                dynamax = False
                tera = False
                
                # New format: {"move": "X", "dmax": true} or {"move": "X", "tera": true}
                if "dmax" in llm_action_json:
                    dynamax = llm_action_json["dmax"] in (True, "true", "True", 1)
                if "tera" in llm_action_json:
                    tera = llm_action_json["tera"] in (True, "true", "True", 1)
                
                # Old format (backward compat): {"dynamax": "move_name"} or {"terastallize": "move_name"}
                if "dynamax" in llm_action_json:
                    dynamax = True
                if "terastallize" in llm_action_json:
                    tera = True
                
                is_a_move = dynamax or tera

                if "move" in llm_action_json.keys() or is_a_move:
                    # Extract move name from appropriate key
                    if "dynamax" in llm_action_json and isinstance(llm_action_json["dynamax"], str):
                        llm_move_id = llm_action_json["dynamax"].strip()  # Old format
                    elif "terastallize" in llm_action_json and isinstance(llm_action_json["terastallize"], str):
                        llm_move_id = llm_action_json["terastallize"].strip()  # Old format
                    else:
                        llm_move_id = llm_action_json["move"].strip()  # Standard or new format
                    move_list = battle.available_moves
                    if dont_verify: # opponent
                        move_list = battle.opponent_active_pokemon.moves.values()
                    
                    # Debug: print available moves
                    if DEBUG:
                        print(f"LLM requested move: '{llm_move_id}'")
                        print(f"Available moves: {[move.id for move in move_list]}")
                    
                    for i, move in enumerate(move_list):
                        if move.id.lower().replace(' ', '') == llm_move_id.lower().replace(' ', ''):
                            #next_action = self.create_order(move, dynamax=sim._should_dynamax(battle), terastallize=sim._should_terastallize(battle))
                            next_action = self.create_order(move, dynamax=dynamax, terastallize=tera)
                            if DEBUG:
                                print(f"Move match found: {move.id}")
                            break
                    
                    if next_action is None and dont_verify:
                        # unseen move so just check if it is in the action prompt
                        if llm_move_id.lower().replace(' ', '') in state_action_prompt:
                            next_action = self.create_order(Move(llm_move_id.lower().replace(' ', ''), self.gen.gen), dynamax=dynamax, terastallize=tera)
                    
                    if next_action is None and DEBUG:
                        print(f"No move match found for '{llm_move_id}'")
                elif "switch" in llm_action_json.keys():
                    llm_switch_species = llm_action_json["switch"].strip()
                    switch_list = battle.available_switches
                    if dont_verify: # opponent prediction
                        observable_switches = []
                        for _, opponent_pokemon in battle.opponent_team.items():
                            if not opponent_pokemon.active:
                                observable_switches.append(opponent_pokemon)
                        switch_list = observable_switches
                    
                    # Debug: print available switches
                    if DEBUG:
                        print(f"LLM requested switch: '{llm_switch_species}'")
                        print(f"Available switches: {[pokemon.species for pokemon in switch_list]}")
                    
                    for i, pokemon in enumerate(switch_list):
                        if pokemon.species.lower().replace(' ', '') == llm_switch_species.lower().replace(' ', ''):
                            next_action = self.create_order(pokemon)
                            if DEBUG:
                                print(f"Switch match found: {pokemon.species}")
                            break
                    
                else:
                    raise ValueError('No valid action')
                
                # with open(f"{self.log_dir}/output.jsonl", "a") as f:
                #     f.write(json.dumps({"turn": battle.turn,
                #                         "system_prompt": system_prompt,
                #                         "user_prompt": state_prompt_io,
                #                         "llm_output": llm_output,
                #                         "battle_tag": battle.battle_tag
                #                         }) + "\n")
                
                if next_action is not None:
                    # tokens and near-timeout
                    tp = getattr(self.llm, 'prompt_tokens', 0)
                    tc = getattr(self.llm, 'completion_tokens', 0)
                    d_prompt = max(0, tp - prev_prompt_tokens)
                    d_comp = max(0, tc - prev_completion_tokens)
                    near_timeout = (time.time() - start_time) > near_timeout_threshold
                    if near_timeout:
                        self._record_timeout(battle, avoided=True)
                    action_kind = 'move' if ("move" in llm_action_json or "dynamax" in llm_action_json or "terastallize" in llm_action_json) else ('switch' if "switch" in llm_action_json else 'unknown')
                    action_label = llm_action_json.get('move') or llm_action_json.get('switch') or llm_action_json.get('dynamax') or llm_action_json.get('terastallize') or ''
                    self._record_action_label(battle, action_kind, str(action_label))
                    progress_score = self._record_progress(battle)
                    self._record_metric(battle, start_time, json_ok=True, fallback_reason='', action_kind=action_kind, action_label=str(action_label), max_tokens=self.max_tokens,
                                        tokens_prompt=d_prompt, tokens_completion=d_comp, near_timeout=near_timeout, progress_score=progress_score)
                    break
            except Exception as e:
                print(f'Exception (JSON/selection) [{self.backend}]: {e}', 'passed')
                # Try forgiving recovery from free-form text
                try:
                    recovered = self._recover_action_from_text(llm_output if 'llm_output' in locals() else '', battle)
                    if recovered is not None:
                        progress_score = self._record_progress(battle)
                        near_timeout = (time.time() - start_time) > near_timeout_threshold
                        if near_timeout:
                            self._record_timeout(battle, avoided=True)
                        self._record_metric(battle, start_time, json_ok=False, fallback_reason='json_recovery', action_kind='move_or_switch', action_label=str(recovered.order if hasattr(recovered,'order') else ''), max_tokens=self.max_tokens,
                                            tokens_prompt=0, tokens_completion=0, near_timeout=near_timeout, progress_score=progress_score)
                        return recovered
                except Exception:
                    pass
                continue

        if next_action is None:
            # Check if we're trapped with no valid actions
            if len(battle.available_moves) == 0 and len(battle.available_switches) == 0:
                print(f'Pokemon {battle.active_pokemon.species} is trapped - no valid actions available')
                progress_score = self._record_progress(battle)
                self._record_metric(battle, start_time, json_ok=False, fallback_reason='trapped_no_actions', action_kind='default', action_label='', max_tokens=self.max_tokens,
                                    tokens_prompt=0, tokens_completion=0, near_timeout=False, progress_score=progress_score)
                return self.choose_default_move()
            
            print('No action found. Using robust fallback chain')
            try:
                print('No action found debug:', llm_action_json if 'llm_action_json' in locals() else 'no json', 
                      actions if 'actions' in locals() else 'no actions', dont_verify)
            except:
                pass
            
            # Use robust finalization with text recovery and safe default
            raw_text = llm_output if 'llm_output' in locals() else ''
            next_action = self._finalize_action(None, raw_text, battle)
            
            progress_score = self._record_progress(battle)
            near_timeout = (time.time() - start_time) > near_timeout_threshold
            if near_timeout:
                self._record_timeout(battle, avoided=True)
            
            # Determine action label for metrics
            action_label = ''
            action_kind = 'safe_default'
            if next_action and hasattr(next_action, 'move'):
                action_label = next_action.move.id if next_action.move else ''
                action_kind = 'move'
            elif next_action and hasattr(next_action, 'pokemon'):
                action_label = next_action.pokemon.species if next_action.pokemon else ''
                action_kind = 'switch'
            
            self._record_metric(battle, start_time, json_ok=False, fallback_reason='no_valid_action', 
                              action_kind=action_kind, action_label=action_label, max_tokens=self.max_tokens,
                              tokens_prompt=0, tokens_completion=0, near_timeout=near_timeout, progress_score=progress_score)
        
        return next_action

    def io2(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle: Battle, sim, dont_verify=False, actions=None):
        """
        Improved IO with:
        - Numbered action contract {"pick": <int>}
        - Action-tier temperature/tokens (colder, shorter)
        - Early deterministic KO-race gate
        - Robust fallbacks
        """
        next_action = None
        # Cap retries for IO to avoid long stalls
        retries = min(3, retries)

        # Early deterministic fast-path: skip LLM if we clearly win TTK race
        try:
            dmg_move, our_ttk = self.dmg_calc_move(battle, return_move=True)
            # If dmg_calc_move returns valid move, check if it's a clear win
            if dmg_move is not None and our_ttk is not None and our_ttk != np.inf:
                try:
                    _, their_ttk = self.estimate_matchup(sim, battle, battle.opponent_active_pokemon, battle.active_pokemon, is_opp=True)
                except Exception:
                    their_ttk = float("inf")
                try:
                    we_move_first = battle.active_pokemon.stats['spe'] > battle.opponent_active_pokemon.stats['spe']
                except Exception:
                    we_move_first = False
                our_effective_ttk = our_ttk + (0 if we_move_first else 1)
                # Clear race edge: we kill in <=2 and strictly before their best line
                if our_ttk <= 2 and our_effective_ttk < (their_ttk - 1):
                    # Return the BattleOrder directly (dmg_move is already an order from dmg_calc_move with return_move=True)
                    result_order = self.create_order(dmg_move) if hasattr(dmg_move, "id") else dmg_move
                    if self.logger.level <= logging.DEBUG:
                        print(f"IO2: Early KO-race exit (our TTK: {our_ttk}, theirs: {their_ttk})")
                    return result_order
        except Exception:
            pass

        # Deterministic, low-token contract: enumerate actions and request {"pick": <int>}
        action_map = []
        idx_lines = []
        for mv in (battle.available_moves or []):
            idx_lines.append(f"{len(action_map)}. move:{mv.id}")
            action_map.append(self.create_order(mv))
        for sw in (battle.available_switches or []):
            idx_lines.append(f"{len(action_map)}. switch:{sw.species}")
            action_map.append(self.create_order(sw))
        
        if not action_map:
            # No legal actions - should be handled upstream but be safe
            return self.choose_default_move()

        indexed_block = "\n\nLEGAL_ACTIONS_INDEXED:\n" + "\n".join(idx_lines)

        # Add concise tactical guardrails for stronger first-choice quality
        tactical_guardrails = (
            "\n\nTactical heuristics: "
            "Avoid status into immunities (Thunder Wave→Ground/Electric, Toxic→Steel/Poison, powders→Grass); "
            "prefer reliable KOs over low-accuracy gambles unless lethal; "
            "avoid type-suicide into obvious OHKOs; "
            "if a safe switch prevents a 2HKO while your attack takes ≥3 turns to KO, prefer the switch."
        )
        
        # Keep reasoning cheap; "why" short
        cot_prompt = '\nIn fewer than 3 sentences, think step by step:'
        constraint_json = '\n\nReturn ONLY JSON: {"pick": <int>, "why":"<=20 tokens"}. Set pick to the index of your chosen action.'
        state_prompt_io = state_prompt + state_action_prompt + indexed_block + tactical_guardrails + cot_prompt + constraint_json

        # Per-move timeout start and accounting
        start_time = time.time()
        near_timeout_threshold = max(0.0, self.move_time_limit_s - 2.0)

        for i in range(retries):
            self._bump_llm_calls(battle)
            # Timeout fallback
            if time.time() - start_time > self.move_time_limit_s:
                print('IO2 timeout, using damage calculator fallback')
                move, _ = self.dmg_calc_move(battle)
                if move is not None:
                    return move
                return self.choose_max_damage_move(battle)

            try:
                llm_output = self.get_LLM_action(
                    system_prompt=system_prompt,
                    user_prompt=state_prompt_io,
                    model=self.backend,
                    # Use action-tier knobs: colder temperature, reasonable token limit
                    temperature=self.temp_action,
                    max_tokens=max(80, min(self.mt_action, 160)),  # At least 80 for JSON response
                    json_format=True,
                    actions=actions
                )

                if DEBUG:
                    print(f"IO2 Raw LLM output: {llm_output}")
                
                # Use forgiving JSON extraction (handles markdown fences, etc.)
                llm_action_json = self._extract_json_obj(llm_output)
                if not llm_action_json:
                    if self.logger.level <= logging.DEBUG:
                        print(f"IO2: Failed to extract JSON from: {llm_output[:200]}")
                    raise ValueError("No valid JSON found in response")
                
                if DEBUG:
                    print(f"IO2 Parsed JSON: {llm_action_json}")

                # Support numbered contract: {"pick": <int>}
                if "pick" in llm_action_json:
                    try:
                        pick = int(llm_action_json["pick"])
                        if 0 <= pick < len(action_map):
                            next_action = action_map[pick]
                            if DEBUG:
                                print(f"IO2: Selected action {pick}")
                    except Exception:
                        pass

                # Fallback to old format if pick failed
                if next_action is None and ("move" in llm_action_json or "switch" in llm_action_json):
                    # Try to match using original logic (for backward compat)
                    if "move" in llm_action_json:
                        llm_move_id = llm_action_json["move"].strip()
                        for move in battle.available_moves:
                            if move.id.lower().replace(' ', '') == llm_move_id.lower().replace(' ', ''):
                                next_action = self.create_order(move)
                                break
                    elif "switch" in llm_action_json:
                        llm_switch_species = llm_action_json["switch"].strip()
                        for pokemon in battle.available_switches:
                            if pokemon.species.lower().replace(' ', '') == llm_switch_species.lower().replace(' ', ''):
                                next_action = self.create_order(pokemon)
                                break

                if next_action is not None:
                    # Dominance override: prefer clearly faster KOs
                    try:
                        mon = battle.active_pokemon
                        opp = battle.opponent_active_pokemon
                        # Best deterministic TTK from damage calc
                        best_action, best_ttk = self.dmg_calc_move(battle)
                        if best_action is not None and mon and opp and hasattr(next_action, "move") and next_action.move:
                            if next_action.move.category != MoveCategory.STATUS:
                                # TTK of the selected move
                                sel_ttk = get_number_turns_faint(
                                    mon, next_action.move, opp, sim,
                                    boosts1=mon._boosts.copy(),
                                    boosts2=opp.boosts.copy()
                                )
                                # If selected is ≥1 turn slower, override
                                if sel_ttk > best_ttk + 1:
                                    next_action = best_action
                                    if self.logger.level <= logging.DEBUG:
                                        print(f"IO2: Dominance override - selected TTK {sel_ttk} vs best {best_ttk}")
                        # If LLM chose a switch but best_ttk ≤ 2 and we're safe, prefer the hit
                        if best_action is not None and hasattr(next_action, "pokemon") and next_action.pokemon and best_ttk <= 2:
                            next_action = best_action
                            if self.logger.level <= logging.DEBUG:
                                print(f"IO2: Dominance override - switch vs clean 2HKO")
                    except Exception:
                        pass

                    # Record metrics
                    near_timeout = (time.time() - start_time) > near_timeout_threshold
                    if near_timeout:
                        self._record_timeout(battle, avoided=True)
                    action_kind = 'move' if any(hasattr(next_action, 'move') and next_action.move for _ in [1]) else 'switch'
                    action_label = ''
                    if hasattr(next_action, 'move') and next_action.move:
                        action_label = next_action.move.id
                    elif hasattr(next_action, 'pokemon') and next_action.pokemon:
                        action_label = next_action.pokemon.species
                    progress_score = self._record_progress(battle)
                    self._record_metric(battle, start_time, json_ok=True, fallback_reason='', 
                                      action_kind=action_kind, action_label=action_label, 
                                      max_tokens=max(80, min(self.mt_action, 160)), tokens_prompt=0, tokens_completion=0, 
                                      near_timeout=near_timeout, progress_score=progress_score)
                    break

            except Exception as e:
                print(f'IO2 Exception [{self.backend}]: {e}')
                continue

        # Final fallback chain
        if next_action is None:
            print('IO2: No valid action found, using robust fallback')
            raw_text = llm_output if 'llm_output' in locals() else ''
            next_action = self._finalize_action(None, raw_text, battle)
            progress_score = self._record_progress(battle)
            self._record_metric(battle, start_time, json_ok=False, fallback_reason='io2_fallback',
                              action_kind='safe_default', action_label='', 
                              max_tokens=min(self.mt_action, 160), tokens_prompt=0, tokens_completion=0,
                              near_timeout=False, progress_score=progress_score)

        return next_action

    def _extract_json_obj(self, s: str) -> Optional[Dict[str, Any]]:
        """Be tolerant: find first '{' and last '}' and parse that slice."""
        try:
            i = s.find("{")
            j = s.rfind("}")
            if i == -1 or j == -1 or j <= i:
                return None
            return json.loads(s[i:j+1])
        except Exception:
            return None

    def _extract_pick_from_text(self, s: str) -> Optional[int]:
        """Fallback: try to extract a pick id from free-form text."""
        try:
            # 1) Try to find "pick": <int>
            m = re.search(r'"pick"\s*:\s*(\d+)', s)
            if m:
                return int(m.group(1))
            # 2) Try to find standalone integer after the word pick
            m = re.search(r'pick[^\d]*(\d+)', s, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def _compute_state_hash(self, battle: AbstractBattle) -> str:
        """Compute a deterministic sha1 hash of key battle state fields."""
        try:
            active = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            state_obj = {
                "tag": battle.battle_tag,
                "turn": battle.turn,
                "me": {
                    "species": getattr(active, "species", ""),
                    "hp": getattr(active, "current_hp_fraction", 0.0),
                    "status": str(getattr(active, "status", "")),
                },
                "opp": {
                    "species": getattr(opp, "species", ""),
                    "hp": getattr(opp, "current_hp_fraction", 0.0),
                    "status": str(getattr(opp, "status", "")),
                },
                "trapped": bool(getattr(battle, "trapped", False)),
                "moves": [m.id for m in (battle.available_moves or [])],
                "switches": [p.species for p in (battle.available_switches or [])],
            }
            payload = json.dumps(state_obj, sort_keys=True, separators=(",", ":")).encode()
            return hashlib.sha1(payload).hexdigest()  # stable across runs
        except Exception:
            # Fallback to simple tag-turn string if anything goes wrong
            return f"{battle.battle_tag}-{battle.turn}"

    def _update_momentum(self, battle: AbstractBattle):
        """Update momentum metrics for the battle."""
        tag = battle.battle_tag
        if tag not in self._momentum:
            self._momentum[tag] = {
                "last_switch_turn": 0,
                "my_switch_count": 0,
                "opp_switch_count": 0,
                "damage_history": [],  # (turn, net_damage)
                "last_actions": []  # Recent action strings for oscillation detection
            }
        
        mom = self._momentum[tag]
        
        # Track switches (simplified - would need message parsing for accuracy)
        # Increment counts based on battle events (placeholder logic)
        
        # Track damage (net damage last turn)
        try:
            my_hp_lost = 1.0 - getattr(battle.active_pokemon, 'current_hp_fraction', 1.0)
            opp_hp_lost = 1.0 - getattr(battle.opponent_active_pokemon, 'current_hp_fraction', 1.0)
            net_dmg = opp_hp_lost - my_hp_lost
            mom["damage_history"].append((battle.turn, net_dmg))
            # Keep only last 5 turns
            mom["damage_history"] = mom["damage_history"][-5:]
        except Exception:
            pass

    def _get_momentum(self, battle: AbstractBattle) -> Dict[str, Any]:
        """Get momentum metrics."""
        tag = battle.battle_tag
        if tag not in self._momentum:
            self._update_momentum(battle)
        
        mom = self._momentum[tag]
        
        # Calculate net chip last 3 turns
        try:
            recent = [d for t, d in mom["damage_history"] if battle.turn - t <= 3]
            net_chip_last3 = sum(recent) * 100 if recent else 0.0
        except Exception:
            net_chip_last3 = 0.0
        
        # Status advantage
        try:
            my_status_count = sum(1 for mon in battle.team.values() if getattr(mon, 'status', None) is not None)
            opp_status_count = sum(1 for mon in battle.opponent_team.values() if getattr(mon, 'status', None) is not None)
            status_adv = opp_status_count - my_status_count
        except Exception:
            status_adv = 0
        
        # Switch delta
        switch_delta = mom.get("opp_switch_count", 0) - mom.get("my_switch_count", 0)
        
        # Oscillation detection (simplified)
        oscillation_flag = len(mom.get("last_actions", [])) >= 4 and len(set(mom["last_actions"][-4:])) <= 2
        
        return {
            "pp_adv": 0,  # TODO: needs PP tracking
            "status_adv": status_adv,
            "net_chip_last3": round(net_chip_last3, 1),
            "switch_delta": switch_delta,
            "oscillation_flag": oscillation_flag
        }

    def _get_progress_score(self, battle: AbstractBattle) -> Dict[str, float]:
        """Get current progress metrics."""
        return self._get_momentum(battle)

    def _get_budget_class(self) -> str:
        """Rough budget class from configured move time limit.
        C0: tight budget; C1/C2: more relaxed.
        """
        try:
            t = float(self.move_time_limit_s or 8.0)
        except Exception:
            t = 8.0
        if t >= 12.0:
            return "C2"
        if t >= 9.0:
            return "C1"
        return "C0"

    def _is_para_immune(self, opponent: Pokemon, move_id: str) -> bool:
        """Gen1: Thunder Wave fails on Ground; treat Electric as immune for safety."""
        if move_id != "thunderwave":
            return False
        t1 = getattr(opponent, 'type_1', None)
        t2 = getattr(opponent, 'type_2', None)
        names = set()
        if t1: names.add(t1.name.upper())
        if t2: names.add(t2.name.upper())
        return ("GROUND" in names) or ("ELECTRIC" in names)

    def _is_powder_immune(self, opponent: Pokemon, move_id: str) -> bool:
        """Conservative: treat Grass as immune to Sleep/Paralysis powders."""
        if move_id not in {"sleeppowder", "stunspore"}:
            return False
        t1 = getattr(opponent, 'type_1', None)
        t2 = getattr(opponent, 'type_2', None)
        names = set()
        if t1: names.add(t1.name.upper())
        if t2: names.add(t2.name.upper())
        return "GRASS" in names

    def _get_phase(self, battle: AbstractBattle) -> str:
        """Determine battle phase: opening, mid, or endgame."""
        try:
            my_alive = sum(1 for mon in battle.team.values() if not getattr(mon, 'fainted', False))
            opp_alive = sum(1 for mon in battle.opponent_team.values() if not getattr(mon, 'fainted', False))
            total_hp_frac = sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in battle.team.values() if not getattr(mon, 'fainted', False))
            total_hp_frac += sum(getattr(mon, 'current_hp_fraction', 0.0) for mon in battle.opponent_team.values() if not getattr(mon, 'fainted', False))
        except Exception:
            my_alive, opp_alive, total_hp_frac = 6, 6, 10.0
        
        if battle.turn <= 5 and my_alive >= 5 and opp_alive >= 5:
            return "opening"
        elif my_alive <= 2 or opp_alive <= 2 or total_hp_frac <= 3.5:
            return "endgame"
        else:
            return "mid"

    def _get_endgame_flags(self, battle: AbstractBattle) -> Dict[str, Any]:
        """Compute endgame indicators."""
        try:
            my_alive = sum(1 for mon in battle.team.values() if not getattr(mon, 'fainted', False))
            opp_alive = sum(1 for mon in battle.opponent_team.values() if not getattr(mon, 'fainted', False))
        except Exception:
            my_alive, opp_alive = 6, 6
        return {
            "is_endgame": (my_alive <= 2 or opp_alive <= 2),
            "my_alive": my_alive,
            "opp_alive": opp_alive,
        }

    def _get_belief_topk(self, battle: AbstractBattle, sim) -> Dict[str, List]:
        """Get Bayesian predictor top-K opponent moves and switches."""
        try:
            opp_moves_str = sim.get_opponent_current_moves(mon=battle.opponent_active_pokemon)
            if not isinstance(opp_moves_str, list):
                opp_moves_str = list(opp_moves_str) if opp_moves_str else []
            opp_moves_topk = [(m, 1.0 / len(opp_moves_str)) for m in opp_moves_str[:3]]
        except Exception:
            opp_moves_topk = []

        try:
            opp_switches = [mon.species for mon in battle.opponent_team.values() 
                          if not getattr(mon, 'fainted', False) and not getattr(mon, 'active', False)]
            opp_switch_topk = [(s, 1.0 / len(opp_switches)) for s in opp_switches[:2]]
        except Exception:
            opp_switch_topk = []

        return {"opp_move_topk": opp_moves_topk, "opp_switch_topk": opp_switch_topk}

    def _belief_mix_score(self, action: Dict, belief: Dict, sim, battle: AbstractBattle) -> float:
        """Compute pure belief-mixed expected value over opponent responses."""
        # Use the improved implementation with fixed parameters
        from pokechamp.rank_improvements_v3_simple import compute_belief_mixed_ev
        return compute_belief_mixed_ev(battle, action, belief, sim)

    def _microsearch_pick(self, pruned_actions: List[Dict], sim, battle, belief: Dict) -> Optional[int]:
        """Two-ply micro-search with proper opponent modeling."""
        # Import the improved implementation
        from pokechamp.rank_improvements_v3_simple import microsearch_two_ply
        
        best_score = -1e9
        best_id = None
        
        for action in pruned_actions:
            score = microsearch_two_ply(battle, action, belief, sim)
            if score > best_score:
                best_score = score
                best_id = action["id"]
                
        return best_id

    def _categorize_move(self, move: Move) -> str:
        """Categorize move for diversity in pruning."""
        if move.category != MoveCategory.STATUS:
            if move.base_power and move.base_power >= 100:
                return "nuke"
            else:
                return "chip"
        else:
            mv_id = move.id
            sleep_moves = {"spore", "sleeppowder", "hypnosis", "sing", "lovelykiss"}
            para_moves = {"thunderwave", "stunspore", "glare"}
            healing_moves = {"softboiled", "recover", "rest", "milkdrink", "synthesis", "moonlight", "morningsun"}
            if mv_id in sleep_moves:
                return "sleep"
            elif mv_id in para_moves:
                return "para"
            elif mv_id in healing_moves:
                return "heal"
            else:
                return "setup"

    def _prune_candidates(self, battle: AbstractBattle, actions: List[Dict], pre_scores: Dict[str, float], max_actions: int = 4, sim=None) -> List[Dict]:
        """Smart pruning with dominance filtering and diversity."""
        from pokechamp.rank_improvements_v3_simple import dominance_filter
        
        # First, apply dominance filtering if we have belief data
        try:
            belief = self._get_belief_topk(battle, sim) if sim else {}
            if belief and (belief.get("opp_move_topk") or belief.get("opp_switch_topk")):
                # Build EV matrix for dominance checking
                ev_matrix = {}
                for action in actions:
                    action_id = action["id"]
                    ev_matrix[action_id] = {}
                    
                    # Compute EV against each opponent response
                    for opp_move_id, _ in belief.get("opp_move_topk", [])[:3]:
                        # Simplified - use pre-score as proxy for EV
                        ev_matrix[action_id][f"move_{opp_move_id}"] = pre_scores.get(str(action_id), 0.0)
                    
                    for opp_switch, _ in belief.get("opp_switch_topk", [])[:2]:
                        ev_matrix[action_id][f"switch_{opp_switch}"] = pre_scores.get(str(action_id), 0.0)
                
                # Apply dominance filter
                actions = dominance_filter(actions, ev_matrix)
        except Exception:
            pass  # Skip dominance filtering on error
        
        # Separate moves and switches
        moves = [(a, float(pre_scores.get(str(a["id"]), 0.0))) for a in actions if a["type"] in ["move", "dynamax", "terastallize"]]
        switches = [(a, float(pre_scores.get(str(a["id"]), 0.0))) for a in actions if a["type"] == "switch"]
        
        # Sort by score
        moves.sort(key=lambda x: x[1], reverse=True)
        switches.sort(key=lambda x: x[1], reverse=True)
        
        # Avoid pointless healing/status moves at high HP or no-effect statuses
        healing_moves = {"softboiled", "recover", "rest", "milkdrink", "synthesis", "moonlight", "morningsun"}
        para_moves = {"thunderwave", "stunspore", "glare"}
        opp_status = getattr(getattr(battle, 'opponent_active_pokemon', None), 'status', None)
        hp_frac = getattr(getattr(battle, 'active_pokemon', None), 'current_hp_fraction', 1.0)
        filtered_moves: List[Tuple[Dict, float]] = []
        for a, v in moves:
            if a.get("type") == "move":
                mv = a.get("move", "")
                if mv in healing_moves and hp_frac >= 0.9:
                    # Skip healing when essentially full HP
                    continue
                if mv in para_moves and opp_status is not None:
                    # Skip paralysis attempts if opponent already has a status
                    continue
                if mv in para_moves:
                    opp = getattr(battle, 'opponent_active_pokemon', None)
                    if opp is not None and self._is_para_immune(opp, mv):
                        continue
                if mv in {"sleeppowder", "stunspore"}:
                    opp = getattr(battle, 'opponent_active_pokemon', None)
                    if opp is not None and self._is_powder_immune(opp, mv):
                        continue
            filtered_moves.append((a, v))

        # Enforce diversity: at least one status move if it's good
        if sim is not None and len(filtered_moves) > 0:
            categories = {}
            for a, v in filtered_moves:
                try:
                    mv = Move(a["move"], gen=sim.gen.gen)
                    cat = self._categorize_move(mv)
                    if cat not in categories or v > categories[cat][1]:
                        categories[cat] = (a, v)
                except Exception:
                    pass
            # If we have a good status move and no status in top, force include it
            has_status_in_top = False
            for a, _ in filtered_moves[:max_actions-1]:
                try:
                    mv = Move(a["move"], gen=sim.gen.gen)
                    if self._categorize_move(mv) in ["sleep", "para", "setup"]:
                        has_status_in_top = True
                        break
                except Exception:
                    pass
            if not has_status_in_top:
                for cat in ["sleep", "para", "setup"]:
                    if cat in categories and categories[cat][1] >= 0.3:
                        # Force include this status move
                        filtered_moves = [categories[cat]] + [x for x in filtered_moves if x[0]["id"] != categories[cat][0]["id"]]
                        break

        # Sort by score
        filtered_moves.sort(key=lambda x: x[1], reverse=True)
        switches.sort(key=lambda x: x[1], reverse=True)

        # Take top moves and best switch
        pruned = [a for a, _ in filtered_moves[:max_actions-1]]
        if switches and len(pruned) < max_actions:
            pruned.append(switches[0][0])

        return pruned[:max_actions]

    def rank_and_pick(self, retries, system_prompt, state_prompt, state_action_prompt, battle, sim, actions=None):
        """Single-call rank-and-pick: LLM scores all actions in one call."""
        start_time = time.time()

        # Get available actions with indices
        legal_actions = []
        action_map = {}  # id -> BattleOrder
        idx = 0

        # Add moves with PP info
        for move in battle.available_moves:
            legal_actions.append({
                "id": idx,
                "type": "move",
                "move": move.id,
                "pp": move.current_pp if hasattr(move, 'current_pp') else None
            })
            action_map[idx] = self.create_order(move)
            idx += 1

        # Add switches
        for pokemon in battle.available_switches:
            legal_actions.append({
                "id": idx,
                "type": "switch",
                "to": pokemon.species
            })
            action_map[idx] = self.create_order(pokemon)
            idx += 1

        # Gimmicks (dynamax, tera, etc.)
        if battle.can_dynamax:
            for move in battle.available_moves:
                legal_actions.append({
                    "id": idx,
                    "type": "dynamax",
                    "move": move.id
                })
                action_map[idx] = self.create_order(move, dynamax=True)
                idx += 1
        
        if battle.can_tera:
            for move in battle.available_moves:
                legal_actions.append({
                    "id": idx,
                    "type": "terastallize",
                    "move": move.id
                })
                action_map[idx] = self.create_order(move, terastallize=True)
                idx += 1
        
        # If no legal actions, return default
        if not legal_actions:
            return self.choose_default_move()
        
        # Get endgame and belief context
        endgame = self._get_endgame_flags(battle)
        belief = self._get_belief_topk(battle, sim)

        # Calculate opponent danger once (estimated turns for opponent to KO us)
        try:
            _, opp_turns_to_ko = self.estimate_matchup(sim, battle, battle.opponent_active_pokemon, battle.active_pokemon, is_opp=True)
        except Exception:
            opp_turns_to_ko = np.inf

        # Calculate pre-scores using pure belief-mixed EV
        pre_scores = {}
        from pokechamp.rank_improvements_v3_simple import explosion_gate
        
        for action in legal_actions:
            try:
                # Check explosion gate
                if action["type"] in ["move", "dynamax", "terastallize"]:
                    if not explosion_gate(action, battle, belief, sim):
                        pre_scores[str(action["id"])] = 0.0
                        continue
                
                # Compute pure belief-mixed expected value
                pre_score = self._belief_mix_score(action, belief, sim, battle)
                
                # Apply Gen1 quirks if applicable
                if sim.gen.gen == 1 and action["type"] in ["move", "dynamax", "terastallize"]:
                    try:
                        move = Move(action["move"], gen=sim.gen.gen)
                        pre_score = Gen1Quirks.adjust_move_value(
                            move, battle.active_pokemon, battle.opponent_active_pokemon,
                            battle, pre_score
                        )
                    except:
                        pass

                # Don't round or clip - preserve margins
                pre_scores[str(action["id"])] = pre_score
            except Exception as e:
                # Fallback for errors
                pre_scores[str(action["id"])] = 0.15

        # Budget-aware pruning with dynamic adjustment based on danger/margin
        budget_class = self._get_budget_class()
        base_max = 4 if budget_class == "C0" else 6

        # Temporarily calculate margin for pruning decision
        temp_sorted = sorted(pre_scores.values(), reverse=True)
        temp_margin = temp_sorted[0] - (temp_sorted[1] if len(temp_sorted) > 1 else 0.0)
        temp_danger = 1.0 / max(1, opp_turns_to_ko) if opp_turns_to_ko != np.inf else 0.0

        # Adjust pruning based on situation
        if temp_margin > 0.5 and temp_danger < 0.3:
            # Clear decision, low danger - prune harder
            max_actions = max(3, base_max - 1)
        elif temp_margin < 0.1 and temp_danger > 0.5:
            # Tight decision, high danger - allow more options
            max_actions = min(8, base_max + 1)
        else:
            max_actions = base_max

        pruned_actions = self._prune_candidates(battle, legal_actions, pre_scores, max_actions=max_actions, sim=sim)

        # Reindex pruned actions 0..N-1 for a cleaner contract
        reindexed_actions = []
        reindex_map = {}
        for new_id, a in enumerate(pruned_actions):
            reindexed = dict(a)
            reindexed["id"] = new_id
            reindexed_actions.append(reindexed)
            reindex_map[new_id] = a["id"]  # new -> old
        pruned_actions = reindexed_actions
        pruned_map = {new_id: action_map[old_id] for new_id, old_id in reindex_map.items()}

        # Build battle context
        state_hash = self._compute_state_hash(battle)

        # Check sleep/freeze clause status
        sleep_count = sum(1 for mon in battle.opponent_team.values() 
                         if mon.status == Status.SLP and not mon.fainted)
        freeze_count = sum(1 for mon in battle.opponent_team.values() 
                          if mon.status == Status.FRZ and not mon.fainted)

        # Update momentum before building context
        self._update_momentum(battle)

        # Get all context components
        phase = self._get_phase(battle)
        momentum = self._get_momentum(battle)

        # Calculate danger using proper KO probability
        from pokechamp.rank_improvements_v3_simple import compute_ko_probability
        danger_next_turn = 0.0

        # Compute danger from opponent's top move
        opp_topk = belief.get("opp_move_topk", [])
        if opp_topk:
            try:
                opp_move_id, opp_move_prob = opp_topk[0]
                opp_move = Move(opp_move_id, gen=sim.gen.gen)
                ko_prob = compute_ko_probability(battle.opponent_active_pokemon, opp_move, battle.active_pokemon, sim)
                danger_next_turn = ko_prob * opp_move_prob
            except Exception:
                # Fallback to simple calculation
                danger_next_turn = 1.0 / max(1, opp_turns_to_ko) if opp_turns_to_ko != np.inf else 0.0
        else:
            # No belief, use simple calculation
            danger_next_turn = 1.0 / max(1, opp_turns_to_ko) if opp_turns_to_ko != np.inf else 0.0

        sorted_pre = sorted(pre_scores.values(), reverse=True)
        q_margin = sorted_pre[0] - (sorted_pre[1] if len(sorted_pre) > 1 else 0.0)

        # Filter pre-scores to only pruned actions (reindexed) - preserve margins!
        pruned_pre_scores = {str(a["id"]): pre_scores.get(str(reindex_map[a["id"]]), 0.0) for a in pruned_actions}

        # Rich JSON user payload per spec
        user_ctx = {
            "version": "ctx-1.3",
            "format": getattr(self, "format", "gen9ou"),
            "turn": battle.turn,
            "state_hash": state_hash,
            "phase": phase,
            "endgame": endgame["is_endgame"],
            "mechanics": {
                "tera": False,
                "sleep_clause": True,
                "freeze_clause": True,
                "partial_trap_active": bool(getattr(battle, "trapped", False)),
                "reflect_active": False,  # TODO: track from battle events
                "lightscreen_active": False,
            },
            "clause_state": {
                "sleep_clause_used_by_me": sleep_count >= 1,
                "freeze_clause_engaged": freeze_count >= 1,
            },
            "belief": {
                "opp_move_topk": belief.get("opp_move_topk", [])[:3],
                "opp_switch_topk": belief.get("opp_switch_topk", [])[:2],
                "entropy": 0.0,  # TODO: compute entropy from belief distribution
            },
            "momentum": momentum,
            "risk": {
                "danger_next_turn": round(danger_next_turn, 2),
                "q_margin": round(q_margin, 2),
            },
            "budget_class": budget_class,
            "legal_actions": pruned_actions,
            "pre_scores": pruned_pre_scores,
            "request": "rank_actions_by_winprob",
        }

        rank_system_prompt = (
            "You are a ranking function for a competitive Pokemon battle.\n"
            "Return ONLY a compact JSON object with fields: state_hash, scores, pick.\n"
            "Do not include any text before or after JSON. Do not explain your reasoning.\n"
            "Scores must be floats in [0,1]. pick must be one of the provided action ids.\n"
            "If two actions are tied, prefer lower variance (avoid Hyper Beam unless it KOs)."
        )
        state_prompt_rank = json.dumps(user_ctx, separators=(",", ":"))

        # Single LLM call
        best_pick = None
        model_scores: Dict[int, float] = {}
        for i in range(retries):
            try:
                # TODO: make temp and max_tokens a parameter, and/or add to the documentation
                temp_for_rank = 0.25
                max_tokens_for_rank = min(int(self.max_tokens), 220)
                llm_output = self.get_LLM_action(
                    system_prompt=rank_system_prompt,
                    user_prompt=state_prompt_rank,
                    model=self.backend,
                    temperature=temp_for_rank,
                    max_tokens=max_tokens_for_rank,
                    json_format=True,
                    reasoning_effort=self.reasoning_effort
                )

                # Extract JSON tolerantly
                llm_json = self._extract_json_obj(llm_output)
                if not llm_json:
                    raise ValueError("No valid JSON found in response")

                # Validate response
                scores = llm_json.get("scores", {})
                pick = llm_json.get("pick")

                # Check state hash (soft warning only)
                if llm_json.get("state_hash") != state_hash:
                    print("Warning: rank state_hash mismatch")

                # Priority 1: Use the explicit pick if valid
                if pick is not None:
                    try:
                        pick = int(pick)
                        if pick in pruned_map:
                            best_pick = pick
                            # Also extract scores for micro-search
                            if scores:
                                model_scores = {int(k): float(v) for k, v in scores.items() if int(k) in pruned_map}
                            break
                    except:
                        pass

                # Priority 2: Use scores argmax
                if scores:
                    try:
                        valid_scores = {int(k): float(v) for k, v in scores.items() if int(k) in pruned_map}
                        if valid_scores:
                            best_pick = max(valid_scores.items(), key=lambda x: x[1])[0]
                            model_scores = valid_scores
                            break
                    except:
                        pass

                # Priority 3: Last resort - try text extraction (avoid if possible)
                if not best_pick:
                    pick_only = self._extract_pick_from_text(llm_output)
                    if pick_only is not None and pick_only in pruned_map:
                        best_pick = pick_only
                        break

            except Exception as e:
                print(f"Rank attempt {i+1} failed: {e}")
                if 'llm_output' in locals():
                    print(f"LLM output was: {llm_output[:200]}...")
                if i < retries - 1:
                    continue

        # Use the pick if we got one
        if best_pick is not None and best_pick in pruned_map:
            # Optional micro-search when margin small, danger high, and budget allows
            try:
                if model_scores and budget_class in ["C1", "C2"]:
                    sorted_scores = sorted(model_scores.values(), reverse=True)
                    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
                    # Only trigger on very tight margins AND high danger
                    if margin < 0.02 and danger_next_turn > 0.5:
                        ms_pick = self._microsearch_pick(pruned_actions, sim, battle, belief)
                        if ms_pick is not None and ms_pick in pruned_map:
                            best_pick = ms_pick
            except Exception:
                pass

            # Record metrics
            elapsed = time.time() - start_time
            near_timeout = elapsed > (self.move_time_limit_s * 0.8)
            if near_timeout:
                self._record_timeout(battle, avoided=True)

            picked_action = next(a for a in pruned_actions if a["id"] == best_pick)
            action_kind = picked_action["type"]
            action_label = picked_action.get("move") or picked_action.get("to", "")

            progress_score = self._record_progress(battle)
            self._record_metric(battle, start_time, json_ok=True, fallback_reason='', 
                              action_kind=action_kind, action_label=action_label,
                              max_tokens=self.max_tokens, tokens_prompt=0, tokens_completion=0,
                              near_timeout=near_timeout, progress_score=progress_score)

            return pruned_map[best_pick]

        # Fallback to pre-scores argmax
        if pruned_pre_scores:
            best_id = int(max(pruned_pre_scores.items(), key=lambda x: float(x[1]))[0])
            if best_id in pruned_map:
                print("Using pre-scores argmax fallback")
                return pruned_map[best_id]

        # Final fallback chain with robust finalization
        print("Rank algorithm failed after all retries, using robust fallback chain")
        
        # Try damage calculator first
        move, _ = self.dmg_calc_move(battle)

        # Finalize with text recovery and safe default
        raw_text = llm_output if 'llm_output' in locals() else ''
        final_action = self._finalize_action(move, raw_text, battle)

        # Determine action label for metrics
        action_label = ''
        action_kind = 'safe_default'
        fallback_reason = 'rank_failed_safe_default'

        if move is not None:
            # Damage calc worked
            action_kind = 'dmg_calc'
            fallback_reason = 'rank_failed_dmgcalc'
            if hasattr(move, 'move') and move.move:
                action_label = move.move.id
        elif final_action:
            # Finalization produced something
            if hasattr(final_action, 'move') and final_action.move:
                action_label = final_action.move.id
                action_kind = 'move'
            elif hasattr(final_action, 'pokemon') and final_action.pokemon:
                action_label = final_action.pokemon.species
                action_kind = 'switch'

        progress_score = self._record_progress(battle)
        self._record_metric(battle, start_time, json_ok=False, fallback_reason=fallback_reason,
                          action_kind=action_kind, action_label=action_label, max_tokens=self.max_tokens,
                          tokens_prompt=0, tokens_completion=0, near_timeout=False,
                          progress_score=progress_score)

        return final_action

    def sc(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim):
        # Track timeout for SC algorithm
        start_time = time.time()
        actions = []

        # SC needs tighter deadline since it makes multiple calls
        # Reserve at least 2 seconds for fallback
        sc_deadline = self.move_time_limit_s - 2.0

        # For SC with K>1, we need even tighter control
        per_sample_budget = sc_deadline / max(self.K, 1)

        # Collect K samples, but with timeout protection
        for i in range(self.K):
            # Check if we've exceeded time limit (with buffer for fallback)
            elapsed = time.time() - start_time
            if elapsed > sc_deadline or (i > 0 and elapsed > per_sample_budget * (i + 0.5)):
                print(f'SC timeout after {i} samples, using fallback')
                # If we have at least one action, use it
                if actions:
                    progress_score = self._record_progress(battle)
                    self._record_timeout(battle, avoided=False)
                    self._record_metric(battle, start_time, json_ok=False, fallback_reason='sc_timeout_partial', 
                                        action_kind='first_action', action_label=str(actions[0].message), 
                                        max_tokens=self.max_tokens, tokens_prompt=0, tokens_completion=0, 
                                        near_timeout=True, progress_score=progress_score)
                    return actions[0]
                # Otherwise fall back to damage calculator
                move, _ = self.dmg_calc_move(battle)
                if move is not None:
                    progress_score = self._record_progress(battle)
                    self._record_timeout(battle, avoided=False)
                    self._record_metric(battle, start_time, json_ok=False, fallback_reason='sc_timeout_dmgcalc', 
                                        action_kind='move', action_label=str(getattr(move, 'order', '')), 
                                        max_tokens=self.max_tokens, tokens_prompt=0, tokens_completion=0, 
                                        near_timeout=True, progress_score=progress_score)
                    return move
                # Last resort: max damage move
                progress_score = self._record_progress(battle)
                self._record_timeout(battle, avoided=False)
                self._record_metric(battle, start_time, json_ok=False, fallback_reason='sc_timeout_maxdmg', 
                                    action_kind='max_damage', action_label='', max_tokens=self.max_tokens, 
                                    tokens_prompt=0, tokens_completion=0, near_timeout=True, 
                                    progress_score=progress_score)
                return self.choose_max_damage_move(battle)

            # Get one sample
            action = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, 
                            state_action_prompt, battle, sim)
            actions.append(action)

        # Vote on the most common action
        action_message = [action.message for action in actions]
        _, counts = np.unique(action_message, return_counts=True)
        index = np.argmax(counts)

        # Record successful SC completion
        progress_score = self._record_progress(battle)
        near_timeout = (time.time() - start_time) > max(0.0, self.move_time_limit_s - 2.0)
        if near_timeout:
            self._record_timeout(battle, avoided=True)
        self._record_metric(battle, start_time, json_ok=True, fallback_reason='', 
                            action_kind='sc_vote', action_label=str(actions[index].message), 
                            max_tokens=self.max_tokens, tokens_prompt=0, tokens_completion=0, 
                            near_timeout=near_timeout, progress_score=progress_score)

        return actions[index]

    def estimate_matchup(self, sim: LocalSim, battle: Battle, mon: Pokemon, mon_opp: Pokemon, is_opp: bool=False) -> Tuple[Move, int]:
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
                t = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
            else:
                t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())

            # Apply Gen1 quirks to evaluation
            if sim.gen.gen == 1 and not is_opp:
                # Convert turns to value (inverse relationship)
                move_value = 1.0 / max(1, t)
                # Apply quirks
                move_value = Gen1Quirks.adjust_move_value(move, mon, mon_opp, battle, move_value)
                # Convert back to effective turns (lower is better)
                t = 1.0 / max(0.01, move_value)

            hp_remaining.append(t)
            # _, hp2, _, _ = sim.calculate_remaining_hp(battle.active_pokemon, battle.opponent_active_pokemon, move, None)
            # hp_remaining.append(hp2)
        hp_best_index = np.argmin(hp_remaining)
        best_move = moves[hp_best_index]
        best_move_turns = hp_remaining[hp_best_index]
        best_move = Move(best_move, gen=sim.gen.gen)
        best_move = self.create_order(best_move)
        # check special moves: tera/dyna
        # dyna for gen 8
        if sim.battle._data.gen == 8 and sim.battle.can_dynamax:
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen).dynamaxed
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, dynamax=True)
                        best_move_turns = t
        # tera for gen 9
        elif sim.battle._data.gen == 9 and sim.battle.can_tera:
            mon.terastallize()
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen)
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, terastallize=True)
                        best_move_turns = t
            mon.unterastallize()

        return best_move, best_move_turns

    def dmg_calc_move(self, battle: AbstractBattle, return_move: bool=False):
        sim = LocalSim(deepcopy(battle), 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    format=self.format
        )
        best_action = None
        best_action_turns = np.inf
        if battle.available_moves and not battle.active_pokemon.fainted:
            # try moves and find hp remaining for opponent
            mon = battle.active_pokemon
            mon_opp = battle.opponent_active_pokemon
            best_action, best_action_turns = self.estimate_matchup(sim, battle, mon, mon_opp)
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

    def _get_fast_heuristic_evaluation(self, battle_state):
        """Fast heuristic evaluation for leaf nodes when LLM is not used."""
        try:
            player_hp = int(battle_state.active_pokemon.current_hp_fraction * 100) if battle_state.active_pokemon else 0
            opp_hp = int(battle_state.opponent_active_pokemon.current_hp_fraction * 100) if battle_state.opponent_active_pokemon else 0
            player_remaining = len([p for p in battle_state.team.values() if not p.fainted])
            opp_remaining = len([p for p in battle_state.opponent_team.values() if not p.fainted])
            
            # Use cached fast evaluation
            return fast_battle_evaluation(
                player_hp, opp_hp, 
                player_remaining, opp_remaining,
                battle_state.turn
            )
        except Exception:  # noqa: BLE001
            # Ultimate fallback to basic hp difference
            try:
                from poke_env.player.local_simulation import LocalSim

                sim = LocalSim(
                    battle_state,
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
                return sim.get_hp_diff()
            except Exception:  # noqa: BLE001
                return 50  # Neutral fallback score

    def _initialize_minimax_optimizer(self, battle):
        """Initialize the minimax optimizer with current battle state."""
        initialize_minimax_optimization(
            battle=battle,
            move_effect=self.move_effect,
            pokemon_move_dict=self.pokemon_move_dict,
            ability_effect=self.ability_effect,
            pokemon_ability_dict=self.pokemon_ability_dict,
            item_effect=self.item_effect,
            pokemon_item_dict=self.pokemon_item_dict,
            gen=self.gen,
            _dynamax_disable=self._dynamax_disable,
            format=self.format,
            prompt_translate=self.prompt_translate
        )
        self._minimax_initialized = True
        logger.info("Minimax optimizer initialized")

    def _ensure_minimax_ready(self, battle) -> bool:
        if not self.use_optimized_minimax:
            return False
        if not self._minimax_initialized:
            try:
                self._initialize_minimax_optimizer(battle)
            except Exception as exc:
                logger.warning("Failed to initialize minimax optimizer: %s", exc)
                self.use_optimized_minimax = False
                return False
        return self.use_optimized_minimax

    def _minimax_fallback(self, battle, start_time, reason: str):
        """
        Minimax fallback with robust finalization.

        Uses damage calculator, then safe default chain if needed.
        """
        # Try damage calculator
        move, _ = self.dmg_calc_move(battle)

        # Use robust finalization to ensure we always return a valid action
        final_action = self._finalize_action(move, '', battle)

        # Determine action label for metrics
        action_label = ''
        action_kind = 'safe_default'
        fallback_reason = f'{reason}_safe_default'

        if move is not None:
            # Damage calc worked
            action_kind = 'dmg_calc'
            fallback_reason = reason
            if hasattr(move, 'move') and move.move:
                action_label = move.move.id
        elif final_action:
            # Finalization produced something
            if hasattr(final_action, 'move') and final_action.move:
                action_label = final_action.move.id
                action_kind = 'move'
            elif hasattr(final_action, 'pokemon') and final_action.pokemon:
                action_label = final_action.pokemon.species
                action_kind = 'switch'

        progress_score = self._record_progress(battle)
        self._record_metric(
            battle,
            start_time,
            json_ok=False,
            fallback_reason=fallback_reason,
            action_kind=action_kind,
            action_label=action_label,
            max_tokens=self.max_tokens,
            tokens_prompt=0,
            tokens_completion=0,
            near_timeout=False,
            progress_score=progress_score,
        )

        return final_action

    def check_timeout(self, start_time, battle):
        if time.time() - start_time > 30:
            print('default due to time')
            move, _ = self.dmg_calc_move(battle)
            return move
        else:
            return None

    def _deserialize_action(self, battle, act_dict, is_opp=False):
        """
        Convert action dict from propose_actions_topk to BattleOrder.
        act_dict: {"action":"move|switch", "id":"<move_id|species>", "score":float}
        """
        try:
            action_type = act_dict.get("action", "")
            action_id = act_dict.get("id", "").lower().replace(" ", "").replace("-", "")
            
            if action_type == "move":
                if is_opp:
                    # Build a Move object for opponent orders (not a raw string)
                    mv = Move(action_id, gen=getattr(self.gen, "gen", 9))
                    return self.create_order(mv)
                else:
                    # Find matching move for player
                    moves = battle.available_moves
                    for move in moves:
                        move_id = getattr(move, 'id', '').replace("-", "")
                        if move_id == action_id:
                            return self.create_order(move)
            elif action_type == "switch":
                if is_opp:
                    # Map species id to the actual opponent Pokemon object
                    for mon in battle.opponent_team.values():
                        species = getattr(mon, 'species', '').lower().replace(" ", "").replace("-", "")
                        if species == action_id and not getattr(mon, 'fainted', False):
                            return self.create_order(mon)
                    return None
                else:
                    # Find matching switch for player
                    switches = battle.available_switches
                    for mon in switches:
                        species = getattr(mon, 'species', '').lower().replace(" ", "").replace("-", "")
                        if species == action_id:
                            return self.create_order(mon)
            
            if self.logger.level <= logging.DEBUG:
                print(f"DEBUG: _deserialize_action returning None for {act_dict} (is_opp={is_opp})")
            return None
        except Exception as e:
            if self.logger.level <= logging.DEBUG:
                print(f"DEBUG: _deserialize_action exception: {e} for {act_dict} (is_opp={is_opp})")
            return None

    def propose_actions_topk(self, battle, sim, k_player=3, k_opp=3, temperature=0.4, max_tokens=140):
        """
        Batched root proposal: Get top-K actions for both player and opponent in ONE call.

        Returns:
          {
            "player": [{"action":"move|switch","id":"<move_id|species>","score":float}, ...],
            "opponent": [{"action":"move|switch","id":"<move_id|species>","score":float}, ...]
          }
        """
        try:
            system_prompt, state_prompt, acts_me, acts_opp, _, _, _ = sim.get_player_prompt(return_actions=True)
            
            prompt = f"""{state_prompt}

TASK: Return top {k_player} actions for us (player) and top {k_opp} actions for opponent.

For each action, provide:
- action: "move" or "switch"
- id: move name (e.g., "earthquake") or species (e.g., "gliscor")
- score: 0.0-1.0 (win probability)

Return JSON ONLY:
{{"player": [{{"action":"move","id":"earthquake","score":0.75}}, ...], "opponent": [{{"action":"move","id":"icebeam","score":0.65}}, ...]}}

Our legal actions: {', '.join(acts_me[:10])}
Opponent likely actions: {', '.join(acts_opp[:10])}"""

            llm_output = self.get_LLM_action(
                system_prompt=system_prompt,
                user_prompt=prompt,
                model=self.backend,
                temperature=temperature,
                max_tokens=max_tokens,
                json_format=True
            )

            result = self._extract_json_obj(llm_output)
            if not result:
                return {"player": [], "opponent": []}

            # Clamp to K
            player_actions = result.get("player", [])[:k_player]
            opp_actions = result.get("opponent", [])[:k_opp]

            return {"player": player_actions, "opponent": opp_actions}

        except Exception as e:
            if self.logger.level <= logging.DEBUG:
                print(f"Batched proposal failed: {e}")
            return {"player": [], "opponent": []}

    def score_leaves_batch(self, leaves, temperature=0.0, max_tokens=180):
        """
        Batched leaf scoring: Score multiple leaf states in ONE call.
        
        Args:
            leaves: List of nodes with .simulation.battle
            
        Returns:
            List[float] - scores in [0,1] for each leaf
        """
        if not leaves:
            return []

        try:
            # Build compact state representation for each leaf
            compact_leaves = []
            for node in leaves:
                b = node.simulation.battle
                us = b.active_pokemon
                opp = b.opponent_active_pokemon

                leaf_state = {
                    "us": {
                        "species": getattr(us, 'species', ''),
                        "hp": int((getattr(us, 'current_hp_fraction', 0) or 0) * 100),
                        "status": str(getattr(us, 'status', None)) if getattr(us, 'status', None) else None
                    },
                    "opp": {
                        "species": getattr(opp, 'species', ''),
                        "hp": int((getattr(opp, 'current_hp_fraction', 0) or 0) * 100),
                        "status": str(getattr(opp, 'status', None)) if getattr(opp, 'status', None) else None
                    },
                    "endgame": len([p for p in b.team.values() if not p.fainted]) <= 2
                }
                compact_leaves.append(leaf_state)

            payload = {"leaves": compact_leaves}

            system_prompt = "You are a strong Gen9 OU evaluator that returns JSON only."
            # user_prompt = f"""For each leaf position, return a win probability for our side (0.0 = certain loss, 1.0 = certain win).

            user_prompt = f"""For each leaf position, return a position score in [0,100] for our side (0 = certain loss, 50 = equal, 100 = certain win).
Return JSON exactly as specified: {{"scores":[42, 73, ...]}}

Positions to evaluate:
{chr(10).join(f"{i}. Us:{leaf['us']['species']}@{leaf['us']['hp']}% vs Opp:{leaf['opp']['species']}@{leaf['opp']['hp']}%" for i, leaf in enumerate(compact_leaves))}"""

            llm_output = self.get_LLM_action(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.backend,
                temperature=temperature,
                max_tokens=max_tokens,
                json_format=True
            )

            result = self._extract_json_obj(llm_output)
            if not result or "scores" not in result:
                return [50.0] * len(leaves)  # Neutral fallback on 0-100 scale

            scores = result.get("scores", [])
            # Clamp to [0,100] and pad if needed
            clamped = [float(max(0.0, min(100.0, s))) for s in scores]
            while len(clamped) < len(leaves):
                clamped.append(50.0)  # Neutral on 0-100 scale

            return clamped[:len(leaves)]

        except Exception as e:
            if self.logger.level <= logging.DEBUG:
                print(f"Batched leaf scoring failed: {e}")
            return [50.0] * len(leaves)  # Neutral fallback on 0-100 scale

    def rank_actions_batched(self, battle, sim=None, start_time=None):
        """
        Batched IO ranker: Get scored ranking for all legal actions in one LLM call.
        Returns: List[(action, score)] sorted by score descending
        """
        if start_time is None:
            start_time = time.time()

        # Get or create sim
        if sim is None:
            from poke_env.player.local_simulation import LocalSimulation
            sim = LocalSimulation(battle, self.pokemon_move_dict, self.move_effect, self.ability_effect, 
                                 self.item_effect, self.type_chart, self.type_list)

        try:
            # Collect all legal actions
            legal_actions = []
            action_map = {}  # index -> BattleOrder

            # Add moves
            for idx, move in enumerate(battle.available_moves):
                move_name = getattr(move, 'id', str(move))
                legal_actions.append(f"move:{move_name}")
                action_map[len(action_map)] = self.create_order(move)

            # Add switches
            for mon in battle.available_switches:
                species = getattr(mon, 'species', str(mon)).lower()
                legal_actions.append(f"switch:{species}")
                action_map[len(action_map)] = self.create_order(mon)

            if not legal_actions:
                return []

            # Build prompt
            from pokechamp.prompts import state_translate2
            state_desc = state_translate2(sim, battle)

            ranking_prompt = f"""{state_desc}

TASK: Score ALL legal actions below on a 0-100 scale, where:
- 0 = certain loss
- 50 = neutral/unclear
- 100 = likely win

Assume opponent plays optimally. Return a JSON object with this exact format:
{{"ranking": [{{"action": "move:moonblast", "score": 75}}, {{"action": "switch:gliscor", "score": 45}}, ...]}}

Legal actions to score:
{chr(10).join(f"{i}. {act}" for i, act in enumerate(legal_actions))}

Score each action and return the complete ranking array."""

            # Make single LLM call
            llm_output = self.get_LLM_action(
                system_prompt="You are an expert Pokemon battler. Provide strategic evaluations.",
                user_prompt=ranking_prompt,
                model=self.backend,
                temperature=self.temp_expand,  # Use expand tier for evaluation
                max_tokens=min(self.mt_expand, 400),  # Cap at 400 for ranking
                json_format=True,
                reasoning_effort=self.reasoning_effort
            )

            # Parse JSON
            llm_json = self._extract_json_obj(llm_output)
            if not llm_json or "ranking" not in llm_json:
                raise ValueError("No ranking in LLM output")

            # Extract scores
            ranked_results = []
            ranking_list = llm_json["ranking"]

            for entry in ranking_list:
                action_str = entry.get("action", "")
                score = entry.get("score", 0)

                # Map action string back to BattleOrder
                if action_str in legal_actions:
                    idx = legal_actions.index(action_str)
                    battle_order = action_map.get(idx)
                    if battle_order:
                        ranked_results.append((battle_order, float(score)))

            # Sort by score descending
            ranked_results.sort(key=lambda x: x[1], reverse=True)

            return ranked_results

        except Exception as e:
            print(f"Batched ranker failed: {e}")
            # Fallback: return empty list (caller should handle)
            return []

    def tree_search_optimized(self, retries, battle, sim=None, return_opp=False, start_time=None) -> BattleOrder:
        """
        Optimized version of tree_search using object pooling and caching.

        This version provides significant performance improvements for minimax:
        - Object pooling for LocalSim instances
        - LLM choice between damage calculator and minimax upfront
        - Battle state caching to avoid repeated computations
        """
        optimizer = get_minimax_optimizer()
        if start_time is None:
            start_time = time.time()
        internal_start = time.time()

        # Hierarchical timing with mutual exclusivity
        from pokechamp.timing_utils import MinimaxTimer
        timer = MinimaxTimer()
        timer.start()
        timer.push('initialization')

        root = optimizer.create_optimized_root(battle)

        try:  # Ensure cleanup happens even on errors
            # NOTE: Avoid heavy prompt generation here; defer to leaf eval or root proposal
            # Quick toggle (disabled by default): get_player_prompt at minimax start
            if False:
                system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = root.simulation.get_player_prompt(return_actions=True)
            # system_prompt, state_prompt, _, _, _, _, _ = root.simulation.get_player_prompt(return_actions=True)

            # Deterministic gate for minimax vs damage calculator
            if not battle.active_pokemon.fainted and len(battle.available_moves) > 0:
                # Get dmg calc move for potential early return
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(battle)
                if dmg_calc_out is not None:
                    # Use deterministic heuristic to decide
                    if not self._should_use_minimax(battle, dmg_calc_turns, sim=root.simulation):
                        print(f'Using damage calculator (clear advantage in {dmg_calc_turns} turns)')
                        progress_score = self._record_progress(battle)
                        self._record_metric(battle, start_time, json_ok=True, fallback_reason='dmg_calc_deterministic', 
                                          action_kind='move', action_label=str(getattr(dmg_calc_out, 'order', '')), 
                                          max_tokens=0, tokens_prompt=0, tokens_completion=0, 
                                          near_timeout=False, progress_score=progress_score)
                        if return_opp:
                            try:
                                action_opp, _ = self.estimate_matchup(root.simulation, battle, 
                                                                   battle.opponent_active_pokemon, 
                                                                   battle.active_pokemon, is_opp=True)
                                return dmg_calc_out, action_opp if action_opp else None
                            except:
                                return dmg_calc_out, None
                        return dmg_calc_out

            print("Using minimax tree search")

            # Set a hard deadline for minimax - leave 1s for final processing
            _time_budget = self.move_time_limit_s or 8.0
            minimax_deadline = start_time + _time_budget - 1.0

            # Use adaptive depth based on time remaining (simple iterative deepening)
            time_budget_per_move = minimax_deadline - time.time()
            if time_budget_per_move < 3.0:
                effective_K = 1  # Shallow search when time-constrained
            elif time_budget_per_move < 6.0:
                effective_K = min(2, self.K)  # Medium depth
            else:
                effective_K = self.K  # Full depth when we have time

            q = deque([root])  # O(1) popleft instead of O(n) pop(0)
            leaf_nodes = []
            expanded_any = False  # Track if we've done at least one expansion

            # End initialization, start expansion loop
            timer.pop()  # initialization
            timer.push('expansion_loop')

            while len(q) != 0:
                # If time ran out AND we've expanded at least once, bail to selection
                if time.time() > minimax_deadline and expanded_any:
                    print(f'Minimax timeout: explored {len(leaf_nodes)} leaf nodes, returning best so far')
                    # If we have any children, proceed to selection with heuristics
                    if leaf_nodes or root.children:
                        break
                    # Otherwise fallback
                    dmg_calc_out, _ = self.dmg_calc_move(battle)
                    if dmg_calc_out is not None:
                        return dmg_calc_out
                    return self.choose_max_damage_move(battle)

                node = q.popleft()

                # Unify battle reference and initialize action lists once per node
                b = node.simulation.battle
                player_actions = []
                opponent_actions = []
                # Quick toggle (disabled by default): per-node prompt generation (expensive)
                if False:
                    _sys, _state, _cot, _io, _sap, _asw, _apm = node.simulation.get_player_prompt(return_actions=True)

                # Check if terminal node or reached depth limit (use effective_K for adaptive depth)
                if node.simulation.is_terminal() or node.depth == effective_K:
                    # Check state-value cache first (post-step state)
                    child_tt_key = mk_ttkey(node.simulation.battle)
                    depth_remaining = self.K - node.depth
                    cached_state_value = optimizer.get_state_value(child_tt_key, min_depth=depth_remaining)

                    if cached_state_value is not None:
                        # State-value cache hit! Skip evaluation entirely
                        node.hp_diff = cached_state_value
                        leaf_nodes.append(node)
                        continue

                    # Cache miss - need to evaluate
                    try:
                        # Use fast heuristic at interior nodes (depth < K)
                        # Mix with LLM at deepest leaves (always, not just endgame)
                        b = node.simulation.battle
                        team_us = sum(not p.fainted for p in b.team.values())
                        team_opp = sum(not p.fainted for p in b.opponent_team.values())
                        is_endgame = (team_us <= 2 or team_opp <= 2)
                        
                        # Only use LLM at deepest leaves in critical situations
                        # This actually reduces LLM calls for real speed improvement
                        # Also check we have time for LLM call (need at least 2s for GPT-4o)
                        time_for_llm = (time.time() + 2.0) < minimax_deadline
                        use_llm = time_for_llm and (node.depth == self.K) and (
                            is_endgame or  # Critical endgame positions
                            abs(node.simulation.get_hp_diff()) < 20  # Close positions
                        )

                        if use_llm:
                            # LLM critic for deepest leaves - generate prompt just-in-time for this leaf
                            value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' +\
                                            'Add points based on the effectiveness of current available moves.' +\
                                            'Award points for each pokemon remaining on the player\'s team, weighted by their strength' +\
                                            'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' +\
                                            'Subtract points for excessive switching.' +\
                                            'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' +\
                                            'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                            cot_prompt = 'Briefly justify your total score, up to 100 words. Then, conclude with the score in the JSON format: {"score": <total_points>}. '
                            # Build state prompt for this specific node
                            _sys_leaf, _state_leaf, *_ = node.simulation.get_player_prompt(return_actions=True)
                            state_prompt_io = _state_leaf + value_prompt + cot_prompt
                            timer.push('llm_calls')
                            llm_output = self.get_LLM_action(system_prompt=_sys_leaf,
                                                            user_prompt=state_prompt_io,
                                                            model=self.backend,
                                                            temperature=self.temp_expand,
                                                            max_tokens=self.mt_expand,
                                                            json_format=True,
                                                            llm=self.llm_value
                                                            )
                            timer.pop()  # llm_calls
                            # Load when llm does heavylifting for parsing
                            llm_action_json = json.loads(llm_output)
                            llm_score = int(llm_action_json['score'])

                            # Get fast heuristic for mixing
                            b = node.simulation.battle
                            fast_score = fast_battle_evaluation(
                                int((b.active_pokemon.current_hp_fraction or 0) * 100),
                                int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100),
                                6 - sum(p.fainted for p in b.team.values()),
                                6 - sum(p.fainted for p in b.opponent_team.values()),
                                b.turn
                            )

                            # Use pure LLM score when we do call it (for critical positions)
                            node.hp_diff = llm_score
                        else:
                            # Use enhanced heuristic with switch penalty
                            b = node.simulation.battle
                            base_score = fast_battle_evaluation(
                                int((b.active_pokemon.current_hp_fraction or 0) * 100),
                                int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100),
                                6 - sum(p.fainted for p in b.team.values()),
                                6 - sum(p.fainted for p in b.opponent_team.values()),
                                b.turn
                            )

                            # Add switch penalty - check opponent's best switch
                            switch_penalty = 0
                            if b.opponent_team and b.active_pokemon and b.available_moves:
                                # Find our strongest move type for checking resistances
                                our_move_types = []
                                for move in b.available_moves:
                                    if move.base_power > 0 and move.type is not None:
                                        our_move_types.append(move.type)

                                if our_move_types:
                                    # Find opponent's best defensive switch
                                    best_resist = 1.0
                                    for mon in b.opponent_team.values():
                                        if not mon.fainted and mon != b.opponent_active_pokemon:
                                            # Check resistance to each of our move types
                                            for move_type in our_move_types:
                                                resist = mon.damage_multiplier(move_type)
                                                best_resist = min(best_resist, resist)

                                    # Penalty if opponent has good switch (resist or immune)
                                    if best_resist <= 0.5:
                                        switch_penalty = 15 * (0.5 - best_resist)  # 0-15 penalty

                            node.hp_diff = base_score - switch_penalty
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # LLM leaf evaluation failed, use heuristic fallback
                        b = node.simulation.battle
                        hp_p = int((b.active_pokemon.current_hp_fraction or 0) * 100)
                        hp_o = int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100)
                        team_p = sum(not p.fainted for p in b.team.values())
                        team_o = sum(not p.fainted for p in b.opponent_team.values())
                        node.hp_diff = fast_battle_evaluation(hp_p, hp_o, team_p, team_o, b.turn)
                        if self.logger.level <= logging.DEBUG:
                            print(f"LLM leaf eval failed, using heuristic: {e}")

                    # Cache the state value (post-step) for transposition table
                    depth_remaining = max(0, self.K - node.depth)
                    cached = optimizer.cache_state_value(child_tt_key, float(node.hp_diff), depth_remaining)
                    if not cached and self.logger.level <= logging.DEBUG:
                        print(f"[TT] Skipped write: deeper entry exists at depth {depth_remaining}")

                    # Also cache by (parent, action, opp) for micro-memoization
                    if node.parent_node and node.parent_node.simulation:
                        parent_tt_key = mk_ttkey(node.parent_node.simulation.battle)
                        p_canonical = canonical_action(node.action, tera=getattr(node.simulation.battle, '_tera_intent', False))
                        o_canonical = canonical_action(node.action_opp, tera=False)
                        # Calculate remaining depth for caching
                        depth_remaining = self.K - node.depth if hasattr(self, 'K') else 0
                        cached = optimizer.cache_evaluation(parent_tt_key, p_canonical, o_canonical, float(node.hp_diff), depth_remaining)
                        if not cached and self.logger.level <= logging.DEBUG:
                            print(f"[Q-cache] Skipped write: deeper entry exists")

                    leaf_nodes.append(node)
                    continue

                # Use batched proposal for root node, fallback to old method for deeper nodes
                if node.depth == 0:
                    # Root batched proposal with soft budget: enable when we have enough actions and time
                    time_remaining_root = minimax_deadline - time.time()
                    total_legal_root = len(b.available_moves or []) + len(b.available_switches or [])
                    use_batched_proposal = time_remaining_root > 1.5 and total_legal_root > 6
                    if use_batched_proposal:
                        if self.logger.level <= logging.DEBUG:
                            print("ROOT PROPOSAL ACTIVE")
                        try:
                            timer.push('llm_calls')
                            seed = self.propose_actions_topk(node.simulation.battle, node.simulation, 
                                                            k_player=2, k_opp=2,
                                                            temperature=self.temp_expand, 
                                                            max_tokens=100)
                            timer.pop()  # llm_calls
                            # Deserialize player actions
                            for act_dict in seed.get("player", []):
                                action = self._deserialize_action(node.simulation.battle, act_dict, is_opp=False)
                                if action and action not in player_actions:
                                    player_actions.append(action)

                            # Deserialize opponent actions
                            opponent_actions = []
                            for act_dict in seed.get("opponent", []):
                                action = self._deserialize_action(node.simulation.battle, act_dict, is_opp=True)
                                if action and action not in opponent_actions:
                                    opponent_actions.append(action)
                        except Exception as e:
                            if self.logger.level <= logging.DEBUG:
                                print(f"Batched proposal failed at root, using fallback: {e}")
                    else:
                        if self.logger.level <= logging.DEBUG:
                            print("ROOT PROPOSAL SKIPPED (timeout guard)")

                    # If proposal failed or returned empty, enumerate all legal actions
                    if not player_actions:
                        # Add all legal moves
                        for move in b.available_moves:
                            player_actions.append(self.create_order(move))
                        # Add all legal switches
                        for switch in b.available_switches:
                            player_actions.append(self.create_order(switch))

                        # If still no actions, try damage calc
                        if not player_actions and not b.active_pokemon.fainted:
                            dmg_calc_out, _ = self.dmg_calc_move(b)
                            if dmg_calc_out:
                                player_actions.append(dmg_calc_out)

                        # Absolute fallback: pick the first legal action
                        if not player_actions:
                            if b.available_moves:
                                player_actions.append(self.create_order(b.available_moves[0]))
                            elif b.available_switches:
                                player_actions.append(self.create_order(b.available_switches[0]))

                    # For opponent, use top-2 estimates and add safety nets
                    if not opponent_actions:
                        # Best damage move (guarded)
                        action_opp, _ = None, float('inf')
                        if b and b.opponent_active_pokemon and b.active_pokemon:
                            opp_moves = getattr(b.opponent_active_pokemon, 'moves', None)
                            if opp_moves:
                                action_opp, _ = self.estimate_matchup(node.simulation, b,
                                                                      b.opponent_active_pokemon,
                                                                      b.active_pokemon, is_opp=True)
                        if action_opp:
                            opponent_actions.append(action_opp)

                        # Best switch (heuristic) - higher score = better for opponent
                        best_score = -np.inf
                        best_switch = None
                        for mon in b.opponent_team.values():
                            if mon.species != b.opponent_active_pokemon.species and not mon.fainted:
                                score = self._estimate_matchup(mon, b.active_pokemon)
                                if score > best_score:
                                    best_score = score
                                    best_switch = mon
                        if best_switch and len(opponent_actions) < 2:
                            opponent_actions.append(self.create_order(best_switch))

                    # Safety net: if still < 1, add any opponent move/switch available
                    if not opponent_actions and b.opponent_active_pokemon:
                        # Try opponent's known moves
                        for mv in (getattr(b.opponent_active_pokemon, 'moves', {}).values() or []):
                            opponent_actions.append(self.create_order(mv))
                            break
                        # If none, add any healthy opponent pokemon as a switch
                        if not opponent_actions:
                            for mon in b.opponent_team.values():
                                if not mon.fainted and mon != b.opponent_active_pokemon:
                                    opponent_actions.append(self.create_order(mon))
                                    break
                else:
                    # Non-root: use original logic (damage calc + estimate)
                    action_opp, opp_turns = None, float('inf')
                    if b and b.opponent_active_pokemon and b.active_pokemon:
                        opp_moves = getattr(b.opponent_active_pokemon, 'moves', None)
                        if opp_moves:
                            action_opp, opp_turns = self.estimate_matchup(
                                node.simulation, b, 
                                b.opponent_active_pokemon, 
                                b.active_pokemon, 
                                is_opp=True
                            )

                    # For inner nodes (depth > 0), use heuristic action generation
                    # Add top damage move and one additional high-power move to increase tactical variety
                    if not b.active_pokemon.fainted and b.available_moves:
                        dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(b)
                        if dmg_calc_out is not None:
                            player_actions.append(dmg_calc_out)
                        # Add a second high-power move if it differs from the dmg-calc choice
                        if b.available_moves:
                            sorted_moves = sorted(b.available_moves, key=lambda mv: (getattr(mv, 'base_power', 0) or 0), reverse=True)
                            for mv in sorted_moves:
                                candidate = self.create_order(mv)
                                if not any(getattr(a, 'order', None) == candidate.order for a in player_actions):
                                    player_actions.append(candidate)
                                    break
                        # Add one STATUS/setup/hazard move for tactical variety (if present)
                        for mv in b.available_moves:
                            if getattr(mv, 'category', None) == MoveCategory.STATUS:
                                candidate = self.create_order(mv)
                                if not any(getattr(a, 'order', None) == candidate.order for a in player_actions):
                                    player_actions.append(candidate)
                                    break

                    # Add best switch if available
                    if b.available_switches:
                        best_switch_score = -np.inf
                        best_switch = None
                        for switch_mon in b.available_switches:
                            # Estimate matchup score (higher = better for switch_mon)
                            score = self._estimate_matchup(switch_mon, b.opponent_active_pokemon)
                            if score > best_switch_score:
                                best_switch_score = score
                                best_switch = switch_mon

                        if best_switch:
                            player_actions.append(self.create_order(best_switch))

                    # Safety net 1: If still empty, enumerate all legal moves
                    if not player_actions:
                        for mv in (b.available_moves or []):
                            player_actions.append(self.create_order(mv))

                    # Safety net 2: If still empty, add 1-2 best switches
                    if not player_actions and b.available_switches:
                        switches = sorted(b.available_switches,
                                        key=lambda s: self._estimate_matchup(s, b.opponent_active_pokemon), reverse=True)[:2]
                        for sw in switches:
                            player_actions.append(self.create_order(sw))

                    # Safety net 3: Absolute last resort
                    if not player_actions:
                        # When moves are disabled and no switches, this should never happen
                        # but if it does, log it clearly
                        if self.logger.level <= logging.DEBUG:
                            print(f"WARNING: No legal actions for inner node at depth {node.depth}")

                    # Generate opponent actions (always ensure ≥2: damage + pivot)
                    if action_opp is not None:
                        opponent_actions.append(action_opp)

                    # Add best defensive pivot/switch - picks mon that resists our best move type
                    if len(opponent_actions) < 2 and b.available_moves:
                        # Find our strongest move type for opponent to resist
                        our_best_move_type = None
                        max_bp = 0
                        for move in b.available_moves:
                            bp = getattr(move, 'base_power', 0) or 0
                            if bp > max_bp and move.type is not None:
                                max_bp = bp
                                our_best_move_type = move.type
                        
                        # Find opponent mon with best resistance to our threat
                        best_score = -np.inf
                        best_switch = None
                        for mon in b.opponent_team.values():
                            if mon != b.opponent_active_pokemon and not mon.fainted:
                                # Bonus for resisting our best move type
                                resist_bonus = 0
                                if our_best_move_type:
                                    multiplier = mon.damage_multiplier(our_best_move_type)
                                    if multiplier <= 0.5:
                                        resist_bonus = (0.5 - multiplier) * 2.0  # 0-2 bonus for resists/immunities
                                
                                score = self._estimate_matchup(mon, b.active_pokemon) + resist_bonus
                                if score > best_score:
                                    best_score = score
                                    best_switch = mon
                        if best_switch:
                            opponent_actions.append(self.create_order(best_switch))

                    # Add one safe opponent damage move (if not already present)
                    if len(opponent_actions) < 2 and getattr(b.opponent_active_pokemon, 'moves', None):
                        opp_moves = list(b.opponent_active_pokemon.moves.values())
                        if opp_moves:
                            opp_sorted = sorted(opp_moves, key=lambda mv: (getattr(mv, 'base_power', 0) or 0), reverse=True)
                            for mv in opp_sorted:
                                cand = self.create_order(mv)
                                if not any(getattr(a, 'order', None) == cand.order for a in opponent_actions):
                                    opponent_actions.append(cand)
                                    break

                    # Safety net: If still < 2, add any available opponent moves/switches
                    if len(opponent_actions) < 2 and b.opponent_active_pokemon:
                        # Try opponent's best damage move
                        if len(opponent_actions) < 2:
                            for mv in (getattr(b.opponent_active_pokemon, 'moves', {}).values() or []):
                                if len(opponent_actions) >= 2:
                                    break
                                opponent_actions.append(self.create_order(mv))

                        # If still < 2, add any healthy opponent pokemon
                        if len(opponent_actions) < 2:
                            for mon in b.opponent_team.values():
                                if len(opponent_actions) >= 2:
                                    break
                                if not mon.fainted and mon != b.opponent_active_pokemon:
                                    opponent_actions.append(self.create_order(mon))
                
                # Create child nodes efficiently (if not at depth limit)
                if node.depth < effective_K:
                    if not player_actions or not opponent_actions:
                        if self.logger.level <= logging.DEBUG:
                            print(f"DEBUG: No actions to expand at depth {node.depth} - player_actions: {len(player_actions) if player_actions else 0}, opponent_actions: {len(opponent_actions) if opponent_actions else 0}")

                    if player_actions and opponent_actions:
                        # Order by 1-ply threat proxy: prioritize high-impact expansions
                        def _threat_score(order, us=True):
                            try:
                                if hasattr(order, "move") and order.move:
                                    mv = order.move
                                    bp = float(getattr(mv, "base_power", 0) or 0)
                                    acc = float(getattr(mv, "accuracy", 1.0) or 1.0)
                                    b_ = node.simulation.battle
                                    atk = b_.active_pokemon if us else b_.opponent_active_pokemon
                                    dfn = b_.opponent_active_pokemon if us else b_.active_pokemon
                                    stab = 1.5 if (mv.type and mv.type in getattr(atk, "types", [])) else 1.0
                                    eff = dfn.damage_multiplier(mv.type) if (dfn and mv.type) else 1.0
                                    return bp * acc * stab * eff
                            except Exception:
                                pass
                            return 0.0
                        
                        player_actions.sort(key=lambda a: _threat_score(a, us=True), reverse=True)
                        opponent_actions.sort(key=lambda a: _threat_score(a, us=False), reverse=True)

                        # Further limit branching if we're running low on time
                        time_remaining = minimax_deadline - time.time()
                        total_legal = len(player_actions) + len(opponent_actions)
                        if time_remaining < 2.0:  # Less than 2 seconds left
                            # Only explore 1 action each when low on time
                            player_limit, opponent_limit = 1, 1
                        elif total_legal > 8:
                            # Coarse pre-limit for fat positions
                            player_limit, opponent_limit = 1, 2
                        else:
                            # Normal limits
                            player_limit, opponent_limit = 2, 2

                        for action_p in player_actions[:player_limit]:
                            for action_o in opponent_actions[:opponent_limit]:
                                # First try parent-action cache (micro-memorization)
                                parent_tt_key = mk_ttkey(node.simulation.battle)
                                p_canonical = canonical_action(action_p, tera=getattr(node.simulation.battle, '_tera_intent', False))
                                o_canonical = canonical_action(action_o, tera=False)
                                child_min_depth = self.K - (node.depth + 1)  # depth remaining from child
                                timer.push('cache_ops')
                                cached_value = optimizer.get_cached_evaluation(parent_tt_key, p_canonical, o_canonical, min_depth=child_min_depth)
                                timer.pop()  # cache_ops

                                if cached_value is not None:
                                    # Parent-action cache hit!
                                    # Create a minimal child node without acquiring sim (no stepping needed)
                                    child_node = CachedChildNode(action_p, action_o, cached_value, node)
                                    node.children.append(child_node)
                                    # Count cache-satisfied children at target depth as leaves
                                    if child_node.depth >= effective_K:
                                        leaf_nodes.append(child_node)
                                    expanded_any = True
                                    # Don't increment nodes_created or add to queue - this is a cache hit
                                else:
                                    # Parent-action cache miss - create child and check state-value cache
                                    child_node = node.create_child_node(action_p, action_o)
                                    optimizer.stats['nodes_created'] += 1

                                    # Check state-value cache for the child state (post-step)
                                    child_tt_key = mk_ttkey(child_node.simulation.battle)
                                    child_min_depth = self.K - (node.depth + 1)  # depth remaining from child
                                    timer.push('cache_ops')
                                    state_value = optimizer.get_state_value(child_tt_key, min_depth=child_min_depth)
                                    timer.pop()  # cache_ops

                                    if state_value is not None:
                                        # State-value cache hit! Use cached value, don't add to queue
                                        child_node.hp_diff = state_value
                                        # Still add as child but mark as evaluated
                                        node.children.append(child_node)
                                        # Count cache-satisfied children at target depth as leaves
                                        if child_node.depth >= effective_K:
                                            leaf_nodes.append(child_node)
                                        expanded_any = True

                                        # Also cache the (parent, action_p, action_o) edge
                                        p_key = mk_ttkey(node.simulation.battle)
                                        p_can = canonical_action(action_p, tera=getattr(node.simulation.battle, '_tera_intent', False))
                                        o_can = canonical_action(action_o, tera=False)
                                        optimizer.cache_evaluation(p_key, p_can, o_can, float(state_value), child_min_depth)
                                    else:
                                        # Full cache miss - assign cheap heuristic immediately for root children
                                        if node.depth == 0 and child_node.simulation is not None:
                                            b = child_node.simulation.battle
                                            hp_p = int((b.active_pokemon.current_hp_fraction or 0) * 100)
                                            hp_o = int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100)
                                            team_p = sum(not p.fainted for p in b.team.values())
                                            team_o = sum(not p.fainted for p in b.opponent_team.values())
                                            base_score = fast_battle_evaluation(hp_p, hp_o, team_p, team_o, b.turn)

                                            # Add speed edge bonus
                                            spe_bonus = 0.0
                                            me, opp = b.active_pokemon, b.opponent_active_pokemon
                                            if me and opp and me.stats and opp.stats:
                                                my_spe = me.stats.get('spe')
                                                if my_spe is None:
                                                    if self.logger.level <= logging.DEBUG:
                                                        print(f"⚠️  Speed stat is None for {me.species}")
                                                    my_spe = 0.0
                                                opp_spe = opp.stats.get('spe')
                                                if opp_spe is None:
                                                    if self.logger.level <= logging.DEBUG:
                                                        print(f"⚠️  Speed stat is None for opponent {opp.species}")
                                                    opp_spe = 0.0
                                                if my_spe > opp_spe:
                                                    spe_bonus = 5.0
                                                elif my_spe < opp_spe:
                                                    spe_bonus = -5.0

                                            # Add type pressure edge
                                            type_bonus = 0.0
                                            def _best_eff(attacker, defender, moves):
                                                effs = []
                                                for mv in (moves or []):
                                                    try:
                                                        if getattr(mv, 'base_power', 0) and getattr(mv, 'type', None):
                                                            effs.append(defender.damage_multiplier(mv.type))
                                                    except Exception:
                                                        pass
                                                return max(effs) if effs else 1.0

                                            our_eff = _best_eff(b.active_pokemon, b.opponent_active_pokemon, b.available_moves)
                                            their_eff = _best_eff(b.opponent_active_pokemon, b.active_pokemon,
                                                                 getattr(b.opponent_active_pokemon, 'moves', {}).values())
                                            type_bonus = max(-10.0, min(10.0, 5.0 * (our_eff - their_eff)))

                                            child_node.hp_diff = base_score + spe_bonus + type_bonus
                                            # Cache this immediate heuristic
                                            optimizer.cache_state_value(child_tt_key, float(child_node.hp_diff), child_min_depth)

                                        # Add to queue for further expansion
                                        q.append(child_node)
                                        node.children.append(child_node)
                                        expanded_any = True

            # End expansion loop
            timer.pop()  # expansion_loop

            # Batch evaluate any unevaluated leaf nodes (only if we have time)
            time_remaining = minimax_deadline - time.time()
            unevaluated_leaves = [n for n in leaf_nodes if n.hp_diff is None]

            # Start batch eval timing
            timer.push('batch_eval')
            if unevaluated_leaves and time_remaining > 1.5:
                # Score all leaves in one LLM call (score_leaves_batch builds compact representation)
                timer.push('llm_calls')
                scores = self.score_leaves_batch(unevaluated_leaves, temperature=self.temp_expand or 0.0, max_tokens=min(180, self.mt_expand or 180))
                timer.pop()  # llm_calls

                # Apply scores and cache
                for node, score in zip(unevaluated_leaves, scores):
                    if score is not None:
                        raw_score = float(score)
                        if not (0.0 <= raw_score <= 100.0):
                            print(f"⚠️  Leaf score out of range: {raw_score}, clamping to [0,100]")
                        node.hp_diff = max(0.0, min(100.0, raw_score))
                        # Cache both TT and parent-edge
                        depth_remaining = self.K - node.depth
                        child_tt_key = mk_ttkey(node.simulation.battle)
                        ok = optimizer.cache_state_value(child_tt_key, node.hp_diff, depth_remaining)
                        if not ok and self.logger.level <= logging.DEBUG:
                            print(f"[TT] Skipped write: deeper entry exists at depth {depth_remaining}")

                        if node.parent_node and hasattr(node, 'action') and hasattr(node, 'action_opp'):
                            parent_tt_key = mk_ttkey(node.parent_node.simulation.battle)
                            p_canonical = canonical_action(node.action, tera=getattr(node.simulation.battle, '_tera_intent', False))
                            o_canonical = canonical_action(node.action_opp, tera=False)
                            ok2 = optimizer.cache_evaluation(parent_tt_key, p_canonical, o_canonical, node.hp_diff, depth_remaining)
                            if not ok2 and self.logger.level <= logging.DEBUG:
                                print(f"[Q-cache] Skipped write: deeper entry exists")

                    elif node.hp_diff is None or node.hp_diff == 0:
                        # Fallback to heuristic if batch scoring returned None (normalize + cache)
                        b = node.simulation.battle
                        hp_p = int((b.active_pokemon.current_hp_fraction or 0) * 100)
                        hp_o = int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100)
                        team_p = sum(not p.fainted for p in b.team.values())
                        team_o = sum(not p.fainted for p in b.opponent_team.values())
                        v = fast_battle_evaluation(hp_p, hp_o, team_p, team_o, b.turn)  # 0..100
                        raw_v = float(v)
                        if not (0.0 <= raw_v <= 100.0):
                            print(f"⚠️  Heuristic out of range: {raw_v}, clamping to [0,100]")
                        node.hp_diff = max(0.0, min(100.0, raw_v))

                        # Cache the fallback evaluation
                        depth_remaining = self.K - node.depth
                        child_tt_key = mk_ttkey(b)
                        ok = optimizer.cache_state_value(child_tt_key, node.hp_diff, depth_remaining)
                        if not ok and self.logger.level <= logging.DEBUG:
                            print(f"[TT] Skipped write: deeper entry exists at depth {depth_remaining}")

                        if node.parent_node and hasattr(node, 'action') and hasattr(node, 'action_opp'):
                            parent_key = mk_ttkey(node.parent_node.simulation.battle)
                            p = canonical_action(node.action, tera=getattr(b, '_tera_intent', False))
                            o = canonical_action(node.action_opp, tera=False)
                            ok2 = optimizer.cache_evaluation(parent_key, p, o, node.hp_diff, depth_remaining)
                            if not ok2 and self.logger.level <= logging.DEBUG:
                                print(f"[Q-cache] Skipped write: deeper entry exists")
            else:
                # Skipped batch eval due to time - ensure all leaves have heuristic values (normalize + cache)
                if unevaluated_leaves:
                    if self.logger.level <= logging.DEBUG:
                        print(f"DEBUG: Skipping batch eval ({len(unevaluated_leaves)} leaves) - only {time_remaining:.1f}s left")

                    for node in unevaluated_leaves:
                        if node.hp_diff is None or node.hp_diff == 0:
                            b = node.simulation.battle
                            hp_p = int((b.active_pokemon.current_hp_fraction or 0) * 100)
                            hp_o = int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100)
                            team_p = sum(not p.fainted for p in b.team.values())
                            team_o = sum(not p.fainted for p in b.opponent_team.values())
                            v = fast_battle_evaluation(hp_p, hp_o, team_p, team_o, b.turn)  # 0..100
                            raw_v = float(v)
                            if not (0.0 <= raw_v <= 100.0):
                                print(f"⚠️  Heuristic out of range: {raw_v}, clamping to [0,100]")
                            node.hp_diff = max(0.0, min(100.0, raw_v))

                            # Cache
                            depth_remaining = self.K - node.depth
                            child_tt_key = mk_ttkey(b)
                            optimizer.cache_state_value(child_tt_key, node.hp_diff, depth_remaining)

                            if node.parent_node and hasattr(node, 'action') and hasattr(node, 'action_opp'):
                                parent_key = mk_ttkey(node.parent_node.simulation.battle)
                                p = canonical_action(node.action, tera=getattr(b, '_tera_intent', False))
                                o = canonical_action(node.action_opp, tera=False)
                                optimizer.cache_evaluation(parent_key, p, o, node.hp_diff, depth_remaining)

            # End batch eval
            timer.pop()  # batch_eval

            # Ensure root children have values before selection (fill with richer heuristic if missing)
            if root.children:
                for child in root.children:
                    if getattr(child, 'hp_diff', None) is None and getattr(child, 'simulation', None) is not None:
                        b = child.simulation.battle
                        hp_p = int((b.active_pokemon.current_hp_fraction or 0) * 100)
                        hp_o = int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100)
                        team_p = sum(not p.fainted for p in b.team.values())
                        team_o = sum(not p.fainted for p in b.opponent_team.values())
                        base_score = fast_battle_evaluation(hp_p, hp_o, team_p, team_o, b.turn)

                        # Add speed edge bonus
                        spe_bonus = 0.0
                        me, opp = b.active_pokemon, b.opponent_active_pokemon
                        if me and opp and me.stats and opp.stats:
                            my_spe = me.stats.get('spe')
                            if my_spe is None:
                                if self.logger.level <= logging.DEBUG:
                                    print(f"⚠️  Speed stat is None for {me.species}")
                                my_spe = 0.0
                            opp_spe = opp.stats.get('spe')
                            if opp_spe is None:
                                if self.logger.level <= logging.DEBUG:
                                    print(f"⚠️  Speed stat is None for opponent {opp.species}")
                                opp_spe = 0.0
                            if my_spe > opp_spe:
                                spe_bonus = 5.0
                            elif my_spe < opp_spe:
                                spe_bonus = -5.0

                        # Add type pressure edge
                        type_bonus = 0.0
                        def _best_eff(attacker, defender, moves):
                            effs = []
                            for mv in (moves or []):
                                try:
                                    if getattr(mv, 'base_power', 0) and getattr(mv, 'type', None):
                                        effs.append(defender.damage_multiplier(mv.type))
                                except Exception:
                                    pass
                            return max(effs) if effs else 1.0

                        our_eff = _best_eff(b.active_pokemon, b.opponent_active_pokemon, b.available_moves)
                        their_eff = _best_eff(b.opponent_active_pokemon, b.active_pokemon,
                                             getattr(b.opponent_active_pokemon, 'moves', {}).values())
                        type_bonus = max(-10.0, min(10.0, 5.0 * (our_eff - their_eff)))

                        child.hp_diff = base_score + spe_bonus + type_bonus
                        # Cache minimal info to help future reuse
                        depth_remaining = max(0, self.K - child.depth)
                        child_tt_key = mk_ttkey(b)
                        optimizer.cache_state_value(child_tt_key, child.hp_diff, depth_remaining)
                        if hasattr(child, 'action') and hasattr(child, 'action_opp') and child.parent_node and child.parent_node.simulation:
                            parent_key = mk_ttkey(child.parent_node.simulation.battle)
                            p = canonical_action(child.action, tera=getattr(b, '_tera_intent', False))
                            o = canonical_action(child.action_opp, tera=False)
                            optimizer.cache_evaluation(parent_key, p, o, child.hp_diff, depth_remaining)

            # Guard against empty root
            if not root.children:
                # Force-enumerate legal actions if pruning/estimates returned nothing
                legal_moves = battle.available_moves or []
                legal_switches = battle.available_switches or []

                # Provide a default opponent reply so LocalSim.step() can advance
                opp_default = None
                b0 = root.simulation.battle
                if b0 and b0.opponent_active_pokemon and b0.active_pokemon:
                    opp_moves = getattr(b0.opponent_active_pokemon, 'moves', None)
                    if opp_moves:
                        opp_default, _ = self.estimate_matchup(root.simulation, b0, b0.opponent_active_pokemon, b0.active_pokemon, is_opp=True)

                if self.logger.level <= logging.DEBUG:
                    print(f"DEBUG: Root has no children, forcing enumeration of {len(legal_moves)} moves and {len(legal_switches)} switches")

                for mv in legal_moves:
                    try:
                        child_node = root.create_child_node(self.create_order(mv), opp_default)
                        optimizer.stats['nodes_created'] += 1
                        root.children.append(child_node)
                    except Exception as e:
                        if self.logger.level <= logging.DEBUG:
                            print(f"DEBUG: Failed to create child for move {mv}: {e}")

                for sw in legal_switches:
                    try:
                        child_node = root.create_child_node(self.create_order(sw), opp_default)
                        optimizer.stats['nodes_created'] += 1
                        root.children.append(child_node)
                    except Exception as e:
                        if self.logger.level <= logging.DEBUG:
                            print(f"DEBUG: Failed to create child for switch {sw}: {e}")

                if not root.children:
                    # Absolute last resort: max damage move
                    print("WARNING: No valid actions could be created, falling back to max damage")
                    return self.choose_max_damage_move(battle)

            # Choose best action using original logic
            def get_tree_action(root_node):
                if len(root_node.children) == 0:
                    # Root node has no children - fallback to damage calc or first legal move
                    if root_node.action is None:
                        # Emergency fallback: get any legal action
                        if battle.available_moves:
                            fallback_action = self.create_order(battle.available_moves[0])
                            print(f"WARNING: Minimax produced no actions, using first legal move: {fallback_action}")
                            return fallback_action, 0, None
                        elif battle.available_switches:
                            fallback_action = self.create_order(battle.available_switches[0])
                            print(f"WARNING: Minimax produced no actions, using first legal switch: {fallback_action}")
                            return fallback_action, 0, None
                        else:
                            # This should never happen - no legal actions at all
                            raise RuntimeError("No legal actions available")
                    return root_node.action, root_node.hp_diff, root_node.action_opp

                score_dict = {}
                action_dict = {}
                opp_dict = {}

                for child in root_node.children:
                    if child.action is None:
                        if self.logger.level <= logging.DEBUG:
                            print(f"DEBUG: Found None action in child node at depth {child.depth}")
                        continue
                    # Skip unevaluated children (ensure only numeric values go into aggregation)
                    if getattr(child, 'hp_diff', None) is None:
                        continue

                    action = str(child.action.order)
                    if action not in score_dict:
                        score_dict[action] = []
                        action_dict[action] = child.action
                        opp_dict[action] = child.action_opp
                    score_dict[action].append(child.hp_diff)

                # Use MINIMAX (worst-case opponent reply) or robust mix
                for action in list(score_dict.keys()):
                    # Filter out any non-numeric scores defensively
                    vals = [v for v in score_dict[action] if isinstance(v, (int, float))]
                    if vals:
                        score_dict[action] = self._robust_value(vals, eta=0.25)
                    else:
                        score_dict[action] = float('-inf')

                if not score_dict:
                    # Fallback if nothing evaluated
                    if battle.available_moves:
                        return self.create_order(battle.available_moves[0]), 0, None
                    if battle.available_switches:
                        return self.create_order(battle.available_switches[0]), 0, None
                    raise RuntimeError("No evaluable actions available for selection")

                best_action_str = max(score_dict, key=score_dict.get)
                best_value = score_dict[best_action_str]

                # Cache the interior node value (after aggregating children)
                if hasattr(root_node, 'simulation') and root_node.simulation:
                    try:
                        interior_tt_key = mk_ttkey(root_node.simulation.battle)
                        depth_remaining = max(0, self.K - root_node.depth)
                        cached = optimizer.cache_state_value(interior_tt_key, float(best_value), depth_remaining)
                        if not cached and self.logger.level <= logging.DEBUG:
                            print(f"[TT] Interior node skipped: deeper entry exists")
                    except Exception as e:
                        print(f"⚠️  [TT] Interior node cache failed: {e}")

                return action_dict[best_action_str], best_value, opp_dict[best_action_str]

            # Time action selection
            timer.push('action_selection')
            action, _, action_opp = get_tree_action(root)
            timer.pop()  # action_selection

            # Log performance stats with instrumentation
            end_time = time.time()
            timer.finish()  # Finalize timing and calculate 'other'

            # Calculate instrumentation metrics
            leaf_total = len(leaf_nodes)
            leaf_llm_scored = sum(1 for n in leaf_nodes if hasattr(n, 'hp_diff') and n.hp_diff is not None and n.hp_diff != 0)

            # Count opponent replies per node
            opp_replies_counts = []
            for child in root.children:
                if hasattr(child, 'action_opp') and child.action_opp is not None:
                    opp_replies_counts.append(1)
            avg_opp_replies = sum(opp_replies_counts) / len(opp_replies_counts) if opp_replies_counts else 0

            stats = optimizer.get_performance_stats()
            cs = stats['cache_stats']
            print(f"⚡ Optimized minimax: {end_time - internal_start:.2f}s, "
                  f"Pool reuse: {stats['pool_stats']['reuse_rate']:.2f}, "
                  f"TT hits: {cs['state_value_hits']}/{cs['state_value_hits'] + cs['state_value_misses']}, "
                  f"Q-cache: {cs['parent_action_hits']}/{cs['parent_action_hits'] + cs['parent_action_misses']}, "
                  f"Leaf LLM: {leaf_llm_scored}/{leaf_total}")

            # Print timing breakdown if profiling is enabled
            if self.profile:
                timer.report()

            # Record successful minimax completion
            progress_score = self._record_progress(battle)
            near_timeout = (end_time - start_time) > max(0.0, self.move_time_limit_s - 2.0)
            if near_timeout:
                self._record_timeout(battle, avoided=True)
            self._record_metric(battle, start_time, json_ok=True, fallback_reason='', 
                                action_kind='minimax', action_label=str(action.message if hasattr(action, 'message') else action.order), 
                                max_tokens=self.max_tokens, tokens_prompt=0, tokens_completion=0, 
                                near_timeout=near_timeout, progress_score=progress_score)

            if return_opp:
                return action, action_opp
            return action

        except Exception as e:
            print(f"Optimized minimax failed: {e}, falling back to damage calculator")
            # Fallback to damage calculator instead of original tree search
            try:
                dmg_calc_move, _ = self.dmg_calc_move(battle)
                if dmg_calc_move is not None:
                    if return_opp:
                        try:
                            action_opp, _ = self.estimate_matchup(None, battle, 
                                                               battle.opponent_active_pokemon, 
                                                               battle.active_pokemon, is_opp=True)
                            return dmg_calc_move, action_opp if action_opp else None
                        except:
                            return dmg_calc_move, None
                    return dmg_calc_move
            except:
                pass
            # Ultimate fallback to max damage move
            return self.choose_max_damage_move(battle)

        finally:
            # Finalize timing if not already done
            if 'timer' in locals():
                timer.finish()
                if self.profile and 'action' in locals() and not action:
                    # Show timing even on failure
                    timer.report()

            # Deterministic cleanup - always return sims to pool
            # Don't rely on __del__ (brittle under cycles/GC)
            try:
                optimizer.cleanup_tree(root)
            except:
                pass  # Cleanup failure shouldn't crash

    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: Battle):
        """
        Choose move with highest expected damage, accounting for:
        - Type effectiveness
        - Accuracy
        - Disabled status
        """
        if battle.available_moves:
            opp = getattr(battle, 'opponent_active_pokemon', None)

            def _score(m):
                # Skip disabled moves
                if getattr(m, 'disabled', False):
                    return -1.0
                bp = float(getattr(m, 'base_power', 0) or 0)
                acc = float(getattr(m, 'accuracy', 1.0) or 1.0)
                mult = 1.0
                try:
                    if opp is not None and getattr(m, 'type', None) is not None:
                        mult = float(opp.damage_multiplier(m.type))
                except Exception:
                    mult = 1.0
                return bp * acc * mult

            best_move = max(battle.available_moves, key=_score)
            return self.create_order(best_move)
        return self.choose_random_move(battle)
