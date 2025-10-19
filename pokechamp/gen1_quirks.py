"""Gen1-specific battle quirks and evaluation adjustments."""

from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status
from poke_env.environment.abstract_battle import AbstractBattle
from typing import Dict, Optional, Tuple


class Gen1Quirks:
    """Handles Gen1-specific mechanics for better decision making."""
    
    @staticmethod
    def evaluate_hyper_beam(attacker: Pokemon, defender: Pokemon, move: Move, 
                          base_damage: float, can_ko: bool) -> float:
        """
        Hyper Beam in Gen1 doesn't require recharge if it KOs.
        Returns adjusted value for the move.
        """
        if move.id != "hyperbeam":
            return base_damage
            
        if can_ko:
            # Huge bonus for KO - no recharge turn
            return base_damage * 2.0
        else:
            # Penalty for non-KO - lose next turn
            return base_damage * 0.5
    
    @staticmethod
    def evaluate_freeze_status(target: Pokemon) -> float:
        """
        Freeze in Gen1 is essentially a KO (no thaw moves except Haze).
        Returns value penalty for getting frozen.
        """
        if target.status == Status.FRZ:
            # Already frozen = dead
            return -1000.0
        
        # Risk of getting frozen (from Ice moves)
        # This should be factored into move evaluation
        return 0.0
    
    @staticmethod
    def check_sleep_clause(battle: AbstractBattle, move: Move) -> bool:
        """
        Check if Sleep Clause prevents using a sleep move.
        Returns True if the move violates sleep clause.
        """
        # Sleep-inducing moves
        sleep_moves = {"spore", "sleeppowder", "hypnosis", "sing", "lovelykiss"}
        
        if move.id not in sleep_moves:
            return False
        
        # Count sleeping opponents
        sleeping_opponents = sum(1 for mon in battle.opponent_team.values() 
                               if mon.status == Status.SLP and not mon.fainted)
        
        # Violates clause if already put one to sleep
        return sleeping_opponents >= 1
    
    @staticmethod
    def evaluate_partial_trap(attacker: Pokemon, defender: Pokemon, move: Move,
                            battle: AbstractBattle) -> float:
        """
        Evaluate partial trapping moves (Wrap, Bind, Clamp, Fire Spin).
        These are very strong in Gen1 due to preventing all actions.
        """
        trap_moves = {"wrap", "bind", "clamp", "firespin"}
        
        if move.id not in trap_moves:
            return 0.0
        
        # Check speed - trap is much better if you're faster
        attacker_speed = attacker.base_stats.get('spe', 100) if attacker.base_stats else 100
        defender_speed = defender.base_stats.get('spe', 100) if defender.base_stats else 100
        
        if attacker_speed > defender_speed:
            # Can trap lock indefinitely
            return 100.0  # Very high value
        else:
            # Still useful but can be broken by switching
            return 20.0
    
    @staticmethod
    def evaluate_wrap_counter(pokemon: Pokemon, opponent: Pokemon, 
                            battle: AbstractBattle) -> Optional[str]:
        """
        Suggest switch if facing a wrapper and we're slower.
        Returns switch target or None.
        """
        # Known wrappers
        wrapper_species = {
            "dragonite", "arbok", "tentacruel", "tangela", 
            "rapidash", "moltres", "cloyster"
        }
        
        if opponent.species.lower() not in wrapper_species:
            return None
            
        # Check if we're slower
        my_speed = pokemon.base_stats.get('spe', 100) if pokemon.base_stats else 100
        opp_speed = opponent.base_stats.get('spe', 100) if opponent.base_stats else 100
        
        if my_speed >= opp_speed:
            return None  # We're faster, no problem
            
        # Look for Normal-resistant switch (Rock/Ghost/Steel - but no Steel in Gen1)
        best_switch = None
        for mon in battle.available_switches:
            if mon.type_1 and mon.type_1.name in ["ROCK", "GHOST"]:
                return mon.species
            # Also consider faster Pokemon that can threaten back
            mon_speed = mon.base_stats.get('spe', 100) if mon.base_stats else 100
            if mon_speed > opp_speed:
                best_switch = mon.species
                
        return best_switch
    
    @staticmethod
    def adjust_move_value(move: Move, attacker: Pokemon, defender: Pokemon,
                         battle: AbstractBattle, base_value: float) -> float:
        """
        Apply all Gen1-specific adjustments to a move's evaluation.
        """
        value = base_value
        
        # Check if move can KO for Hyper Beam adjustment
        if move.id == "hyperbeam":
            # Estimate if it KOs (rough calculation)
            estimated_damage = base_value * defender.current_hp
            can_ko = estimated_damage >= defender.current_hp
            value = Gen1Quirks.evaluate_hyper_beam(attacker, defender, move, 
                                                  value, can_ko)
        
        # Freeze moves get bonus (except vs Ice types)
        freeze_chance_moves = {
            "icebeam": 0.1, "blizzard": 0.1, "icepunch": 0.1
        }
        if move.id in freeze_chance_moves and defender.type_1.name != "ICE":
            # Freeze = KO, so even 10% chance is valuable
            value *= 1.3
        
        # Sleep moves - check clause
        if Gen1Quirks.check_sleep_clause(battle, move):
            value = 0  # Can't use it
        
        # Partial trap evaluation
        trap_bonus = Gen1Quirks.evaluate_partial_trap(attacker, defender, move, battle)
        if trap_bonus > 0:
            value += trap_bonus
            
        # Critical hit rate in Gen1 (based on base speed)
        crit_moves = {"slash", "crabhammer", "razorleaf", "karatechop"}
        if move.id in crit_moves:
            # These have 8x crit rate in Gen1
            value *= 1.5
        
        return value
    
    @staticmethod
    def get_switch_recommendation(battle: AbstractBattle) -> Optional[str]:
        """
        Get Gen1-specific switch recommendations.
        """
        if not battle.active_pokemon or battle.active_pokemon.fainted:
            return None
            
        opponent = battle.opponent_active_pokemon
        if not opponent:
            return None
            
        # Anti-wrap switching
        wrap_counter = Gen1Quirks.evaluate_wrap_counter(
            battle.active_pokemon, opponent, battle
        )
        if wrap_counter:
            return wrap_counter
            
        return None
