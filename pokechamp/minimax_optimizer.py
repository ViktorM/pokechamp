"""
Minimax Performance Optimization System

This module provides optimizations specifically for the minimax tree search
to reduce LocalSim node creation overhead and improve performance.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.local_simulation import LocalSim


@dataclass(frozen=True)
class TTKey:
    """
    Transposition table key for minimax caching.

    Only includes stable, value-driving features.
    Excludes turn_number to allow transposition across turns.
    HP coarse-grained to deciles to improve hit rate.
    """
    active_hp_deciles: Tuple[int, int]              # (us, opp), 0..5 (quintiles)
    active_species: Tuple[str, str]                 # normalized species ids
    fainted_count: Tuple[int, int]                  # (our_fainted, opp_fainted)
    our_side: Tuple[Tuple[str, int], ...]           # hazards/screens on our side
    opp_side: Tuple[Tuple[str, int], ...]           # hazards/screens on opp side
    field: Tuple[str, ...]                          # weather/terrain (normalized)
    boosts: Tuple[int, int, int, int, int, int, int, int, int, int]  # stat boosts (10 values, no accuracy)
    tera_flags: Tuple[int, int]                     # (we_tera_used, opp_tera_used)


def normalize_side(side_dict) -> Tuple[Tuple[str, int], ...]:
    """Normalize side conditions to sorted tuples with bucketed counters."""
    if not side_dict:
        return ()

    def _bucket(n: int) -> int:
        """Bucket counters to reduce state space: 0, 1-2, 3+"""
        if n <= 0: return 0
        if n <= 2: return 1
        return 2

    items = []
    for k, v in side_dict.items():
        name = getattr(k, "name", str(k)).lower().replace(" ", "").replace("-", "")
        items.append((name, _bucket(int(v))))
    return tuple(sorted(items))


def normalize_field(battle) -> Tuple[str, ...]:
    """Normalize weather/terrain/field effects to sorted tuple of names."""
    xs = []
    try:
        if getattr(battle, "weather", None):
            weather_name = getattr(battle.weather, "name", str(battle.weather)).lower().replace(" ", "").replace("-", "")
            if weather_name and weather_name != "none":
                xs.append(weather_name)
    except:
        pass

    try:
        if getattr(battle, "fields", None):
            for k, v in getattr(battle, "fields", {}).items():
                if v:
                    field_name = getattr(k, "name", str(k)).lower().replace(" ", "").replace("-", "")
                    xs.append(field_name)
    except:
        pass

    return tuple(sorted(xs))


def canonical_action(order, tera: bool = False) -> str:
    """
    Canonicalize action to stable string for cache key.
    
    Examples: "M:shadowball", "S:zapdos", "M:icespinner+tera"
    """
    if order is None:
        return "none"

    # Move action
    if hasattr(order, 'move') and order.move:
        move_id = getattr(order.move, "id", "").lower().replace(" ", "").replace("-", "")
        suffix = "+tera" if tera else ""
        return f"M:{move_id}{suffix}"

    # Switch action
    if hasattr(order, 'pokemon') and order.pokemon:
        species = getattr(order.pokemon, "species", "").lower().replace(" ", "").replace("-", "")
        return f"S:{species}"

    return "none"


@dataclass
class BattleStateHash:
    """
    OLD reference implementation - kept for comparison.
    
    Issues with this approach:
    - Includes turn_number (prevents transpositions)
    - Uses exact HP (too precise, poor cache hits)
    - Stringifies weather/terrain (repr noise)
    
    Use TTKey instead for actual caching.
    """
    active_pokemon_hp: Tuple[int, int]  # Exact HP percentage
    active_pokemon_species: Tuple[str, str]
    team_remaining: Tuple[int, int]
    turn_number: int  # <-- Problem: prevents transpositions
    weather: str  # <-- Problem: string repr noise
    terrain: str  # <-- Problem: string repr noise
    hazards: Tuple[str, ...]
    boosts: Tuple[int, int, int, int, int, int, int, int, int, int, int, int]
    
    def __hash__(self):
        return hash((
            self.active_pokemon_hp,
            self.active_pokemon_species,
            self.team_remaining,
            self.turn_number,
            self.weather,
            self.terrain,
            self.hazards,
            self.boosts
        ))


def mk_ttkey(battle: Battle) -> TTKey:
    """
    Create a transposition table key for caching.
    
    Excludes turn_number to allow transpositions.
    Uses coarse-grained HP (deciles) to improve hit rate.
    Normalizes all fields to avoid string noise.
    """
    try:
        u = battle.active_pokemon
        o = battle.opponent_active_pokemon
        
        # HP as quintiles (0-5) for much better cache hits
        active_hp_deciles = (
            int((u.current_hp_fraction if u else 0) * 5),
            int((o.current_hp_fraction if o else 0) * 5)
        )

        # Normalized species
        active_species = (
            (u.species or "").lower().replace(" ", "").replace("-", ""),
            (o.species or "").lower().replace(" ", "").replace("-", "")
        )

        # Fainted count (not remaining - more stable)
        fainted_count = (
            sum(1 for p in battle.team.values() if p.fainted),
            sum(1 for p in battle.opponent_team.values() if p.fainted)
        )

        # Normalize side conditions
        our_side = normalize_side(battle.side_conditions)
        opp_side = normalize_side(battle.opponent_side_conditions)

        # Normalize field effects
        field = normalize_field(battle)

        # Extract stat boosts with bucketing for better TT reuse
        def _boost_bucket(z: int) -> int:
            """Bucket boosts: <=-2, -1, 0, 1-2, 3+"""
            if z <= -2: return -2
            if z == -1: return -1
            if z == 0: return 0
            if z <= 2: return 1
            return 2

        try:
            player_boosts = u.boosts if u else {}
            opp_boosts = o.boosts if o else {}
            
            # Only track value-driving boosts (drop accuracy for most gens)
            boosts = (
                _boost_bucket(player_boosts.get('atk', 0)), 
                _boost_bucket(player_boosts.get('def', 0)),
                _boost_bucket(player_boosts.get('spa', 0)), 
                _boost_bucket(player_boosts.get('spd', 0)),
                _boost_bucket(player_boosts.get('spe', 0)),
                _boost_bucket(opp_boosts.get('atk', 0)), 
                _boost_bucket(opp_boosts.get('def', 0)),
                _boost_bucket(opp_boosts.get('spa', 0)), 
                _boost_bucket(opp_boosts.get('spd', 0)),
                _boost_bucket(opp_boosts.get('spe', 0))
            )
        except:
            boosts = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Tera flags
        try:
            tera_flags = (
                int(getattr(u, "terastallized", False) if u else False or not battle.can_tera),
                int(getattr(o, "terastallized", False) if o else False or not battle.opponent_can_tera)
            )
        except:
            tera_flags = (0, 0)
        
        return TTKey(
            active_hp_deciles=active_hp_deciles,
            active_species=active_species,
            fainted_count=fainted_count,
            our_side=our_side,
            opp_side=opp_side,
            field=field,
            boosts=boosts,
            tera_flags=tera_flags
        )
    except Exception as e:
        # Fallback to minimal key
        return TTKey((0, 0), ("", ""), (0, 0), (), (), (), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0))


# Backward compatibility alias
def create_battle_state_hash(battle: Battle) -> TTKey:
    """Legacy name - redirects to mk_ttkey."""
    return mk_ttkey(battle)


class LocalSimPool:
    """Object pool for LocalSim instances to avoid repeated creation."""

    def __init__(self, initial_size: int = 16):
        self._available_sims: List[LocalSim] = []
        self._in_use_sims: List[LocalSim] = []
        self._initial_size = initial_size
        self._creation_template: Optional[Dict[str, Any]] = None
        self._reuse_count = 0  # Track how many times we reused a sim from pool

    def initialize_pool(self, template_battle: Battle, **localsim_kwargs):
        """Initialize the pool with template LocalSim instances."""
        self._creation_template = localsim_kwargs
        
        print(f"🔄 Initializing LocalSim pool with {self._initial_size} instances...")
        for i in range(self._initial_size):
            sim = LocalSim(
                battle=deepcopy(template_battle),
                **localsim_kwargs
            )
            self._available_sims.append(sim)
        print(f"✅ LocalSim pool initialized with {len(self._available_sims)} instances")

    def acquire_sim(self, battle: Battle) -> LocalSim:
        """Get a LocalSim from the pool, creating new one if needed."""
        if self._available_sims:
            sim = self._available_sims.pop()
            self._reuse_count += 1  # Increment reuse counter
            # Reset the simulation with new battle state
            sim.battle = deepcopy(battle)
            self._in_use_sims.append(sim)
            return sim
        else:
            # Pool exhausted, create new instance
            if self._creation_template is None:
                raise RuntimeError("Pool not initialized - call initialize_pool() first")

            sim = LocalSim(
                battle=deepcopy(battle),
                **self._creation_template
            )
            self._in_use_sims.append(sim)
            return sim

    def release_sim(self, sim: LocalSim):
        """Return a LocalSim to the pool for reuse."""
        if sim in self._in_use_sims:
            self._in_use_sims.remove(sim)
            # Clean up the simulation state
            sim.battle = None  # Clear battle reference
            self._available_sims.append(sim)

    def release_all(self):
        """Release all in-use sims back to the pool."""
        for sim in self._in_use_sims[:]:  # Copy list to avoid modification during iteration
            self.release_sim(sim)

    def get_stats(self) -> Tuple[int, int, int]:
        """Get pool statistics: (available, in_use, total)."""
        return len(self._available_sims), len(self._in_use_sims), len(self._available_sims) + len(self._in_use_sims)

    def get_reuse_count(self) -> int:
        """Get the number of times a sim was reused from the pool."""
        return self._reuse_count

    def reset_metrics(self):
        """Reset reuse counter for new search."""
        self._reuse_count = 0


class MinimaxCache:
    """Cache for minimax evaluation results to avoid recomputation."""

    def __init__(self, max_size: int = 1000):
        # Store (value, depth) to avoid using shallow eval for deeper nodes
        self._cache: Dict[Tuple[BattleStateHash, str, str], Tuple[float, int]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str, min_depth: int = 0) -> Optional[float]:
        """Get cached evaluation for a battle state + action combination."""
        key = (battle_state, player_action, opp_action)
        if key in self._cache:
            val, depth = self._cache[key]
            if depth >= min_depth:
                self._hits += 1
                return val
            # treat as miss if shallower than required
        self._misses += 1
        return None

    def set_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str, value: float, depth: int) -> bool:
        """
        Cache an evaluation result.

        Returns:
            True if cached, False if skipped (existing entry was deeper)
        """
        key = (battle_state, player_action, opp_action)

        # Only cache if deeper than existing entry
        if key in self._cache:
            existing_val, existing_depth = self._cache[key]
            if depth <= existing_depth:
                return False  # Skip, existing is deeper

        # Simple LRU: if cache is full, remove oldest 25% of entries
        if len(self._cache) >= self._max_size:
            items_to_remove = list(self._cache.keys())[:self._max_size // 4]
            for item in items_to_remove:
                del self._cache[item]

        self._cache[key] = (float(value), int(depth))
        return True

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Tuple[int, int, float]:
        """Get cache statistics: (hits, misses, hit_rate)."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return self._hits, self._misses, hit_rate


class OptimizedSimNode:
    """Optimized version of SimNode that uses object pooling and caching."""

    def __init__(self, battle: Battle, sim_pool: LocalSimPool, depth: int = 0):
        self.simulation = sim_pool.acquire_sim(battle)
        self.sim_pool = sim_pool
        self.depth = depth
        self.action: Optional[BattleOrder] = None
        self.action_opp: Optional[BattleOrder] = None
        self.parent_node = None
        self.parent_action = None
        self.hp_diff = None
        self.children: List['OptimizedSimNode'] = []

    def __del__(self):
        """Return simulation to pool when node is destroyed."""
        if hasattr(self, 'simulation') and self.simulation is not None:
            self.sim_pool.release_sim(self.simulation)

    def create_child_node(self, player_action: BattleOrder, opp_action: BattleOrder) -> 'OptimizedSimNode':
        """Create a child node efficiently."""
        # Create new battle state by stepping forward
        child_battle = deepcopy(self.simulation.battle)

        # Create child node
        child_node = OptimizedSimNode(child_battle, self.sim_pool, self.depth + 1)
        child_node.action = player_action
        child_node.action_opp = opp_action
        child_node.parent_node = self
        child_node.parent_action = self.action

        # Step the simulation forward
        child_node.simulation.step(player_action, opp_action)
        
        # Update relationships
        self.children.append(child_node)

        return child_node

    def cleanup(self):
        """Recursively cleanup all child nodes and return sims to pool."""
        for child in self.children:
            child.cleanup()

        if self.simulation is not None:
            self.sim_pool.release_sim(self.simulation)
            self.simulation = None


class MinimaxOptimizer:
    """Main optimizer for minimax tree search."""

    def __init__(self):
        self.sim_pool = LocalSimPool(initial_size=4)  # Larger pool for minimax
        self.cache = MinimaxCache(max_size=2000)  # (parent, action, opp) cache
        self.state_value_cache: Dict[TTKey, Tuple[float, int]] = {}  # child state → (value, depth) cache
        self.stats = {
            'nodes_created': 0,
            'cache_hits': 0,  # Parent-action cache hits
            'pool_reuses': 0,
            'total_time': 0.0,
            'cache_stats': {
                'state_value_hits': 0,
                'state_value_misses': 0,
                'parent_action_hits': 0,
                'parent_action_misses': 0
            }
        }

    def initialize(self, battle: Battle, **localsim_kwargs):
        """Initialize the optimizer with battle template."""
        self.sim_pool.initialize_pool(battle, **localsim_kwargs)
        print(f"🚀 MinimaxOptimizer initialized")

    def create_optimized_root(self, battle: Battle) -> OptimizedSimNode:
        """Create an optimized root node for minimax search."""
        root = OptimizedSimNode(battle, self.sim_pool, depth=0)
        self.stats['nodes_created'] += 1
        return root

    def cleanup_tree(self, root: OptimizedSimNode):
        """Cleanup entire tree and return all sims to pool."""
        root.cleanup()
        self.sim_pool.release_all()

    def evaluate_leaf(self, node: OptimizedSimNode) -> float:
        """Heuristic leaf eval (fast, bounded 0..100)."""
        b = node.simulation.battle
        hp_p = int((b.active_pokemon.current_hp_fraction or 0) * 100)
        hp_o = int((b.opponent_active_pokemon.current_hp_fraction or 0) * 100)
        team_p = len([p for p in b.team.values() if not p.fainted])
        team_o = len([p for p in b.opponent_team.values() if not p.fainted])
        return fast_battle_evaluation(hp_p, hp_o, team_p, team_o, b.turn)

    def get_cached_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str, min_depth: int = 0) -> Optional[float]:
        """Try to get cached evaluation for this state."""
        result = self.cache.get_evaluation(battle_state, player_action, opp_action, min_depth=min_depth)
        if result is not None:
            self.stats['cache_hits'] += 1  # Legacy counter
            self.stats['cache_stats']['parent_action_hits'] += 1
        else:
            self.stats['cache_stats']['parent_action_misses'] += 1
        return result

    def cache_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str, value: float, depth: int = 0) -> bool:
        """
        Cache an evaluation result (parent-action cache).
        
        Returns:
            True if cached, False if skipped
        """
        return self.cache.set_evaluation(battle_state, player_action, opp_action, value, depth)

    def get_state_value(self, key: TTKey, min_depth: int = 0) -> Optional[float]:
        """Get cached value for a child state (post-step) if deep enough."""
        v = self.state_value_cache.get(key)
        if v is None:
            self.stats['cache_stats']['state_value_misses'] += 1
            return None

        val, depth = v
        if depth >= min_depth:
            self.stats['cache_stats']['state_value_hits'] += 1
            return val
        # Too shallow; treat as miss for this query
        self.stats['cache_stats']['state_value_misses'] += 1
        return None

    def cache_state_value(self, key: TTKey, value: float, depth: int) -> bool:
        """
        Cache a state value (post-step) with depth info.
        
        Returns:
            True if value was cached, False if existing entry was deeper
        """
        # Limit cache size to prevent memory explosion
        if len(self.state_value_cache) > 10000:
            # Remove 20% of oldest entries
            to_remove = int(len(self.state_value_cache) * 0.2)
            for k in list(self.state_value_cache.keys())[:to_remove]:
                del self.state_value_cache[k]

        cur = self.state_value_cache.get(key)
        if (cur is None) or (depth > cur[1]):  # keep the deepest eval
            self.state_value_cache[key] = (float(value), int(depth))
            return True
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        pool_available, pool_in_use, pool_total = self.sim_pool.get_stats()
        reuse = self.sim_pool.get_reuse_count()
        cache_hits, cache_misses, cache_hit_rate = self.cache.get_stats()

        return {
            'nodes_created': self.stats['nodes_created'],
            'pool_stats': {
                'available': pool_available,
                'in_use': pool_in_use,
                'total': pool_total,
                'reuse': reuse,
                'reuse_rate': reuse / max(1, self.stats['nodes_created'])
            },
            'cache_stats': {
                'hits': cache_hits,
                'misses': cache_misses,
                'hit_rate': cache_hit_rate,
                'state_value_hits': self.stats['cache_stats']['state_value_hits'],
                'state_value_misses': self.stats['cache_stats']['state_value_misses'],
                'parent_action_hits': self.stats['cache_stats']['parent_action_hits'],
                'parent_action_misses': self.stats['cache_stats']['parent_action_misses']
            },
            'total_time': self.stats['total_time']
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'nodes_created': 0,
            'cache_hits': 0,
            'pool_reuses': 0,
            'total_time': 0.0,
            'cache_stats': {
                'state_value_hits': 0,
                'state_value_misses': 0,
                'parent_action_hits': 0,
                'parent_action_misses': 0
            }
        }
        # Don't clear caches - keep them across turns for better reuse!
        # self.cache.clear()
        # self.state_value_cache.clear()
        self.sim_pool.reset_metrics()  # Reset pool reuse counter


# Global optimizer instance
_minimax_optimizer = MinimaxOptimizer()


def get_minimax_optimizer() -> MinimaxOptimizer:
    """Get the global minimax optimizer instance."""
    return _minimax_optimizer


def initialize_minimax_optimization(battle: Battle, **localsim_kwargs):
    """Initialize minimax optimizations for a battle."""
    _minimax_optimizer.initialize(battle, **localsim_kwargs)


@lru_cache(maxsize=500)
def fast_battle_evaluation(
    active_hp_player: int,
    active_hp_opp: int, 
    team_count_player: int,
    team_count_opp: int,
    turn: int
) -> float:
    """
    Fast heuristic evaluation function that avoids LLM calls.
    
    This provides a quick approximation of battle state value based on:
    - HP advantage
    - Team size advantage  
    - Turn progression penalty
    """
    # HP advantage (0-100 scale)
    hp_advantage = (active_hp_player - active_hp_opp) * 20
    
    # Team advantage (each Pokemon worth ~15 points)
    team_advantage = (team_count_player - team_count_opp) * 15
    
    # Turn penalty (encourages quicker wins)
    turn_penalty = min(turn * 0.5, 10)
    
    # Base score starts at 50 (neutral)
    score = 50 + hp_advantage + team_advantage - turn_penalty
    
    # Clamp to valid range
    return max(0, min(100, score))