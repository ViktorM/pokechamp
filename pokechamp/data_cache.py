"""
Global data cache for Pokemon simulation data.

This module provides a singleton cache for all static game data to avoid 
repeated file loading during battles and simulations.
"""

import json
import orjson
from typing import Dict, Any
from functools import lru_cache
from poke_env.data.gen_data import GenData


class GameDataCache:
    """Singleton cache for static game data."""
    
    _instance = None
    _data = {}
    _elo_tier = 1825  # Default to highest Elo tier for sharper priors
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Initialize data only once
        if not self._data:
            self._load_all_data()
    
    @classmethod
    def set_elo_tier(cls, elo: int):
        """
        Set the Elo tier for loading move sets.
        Available tiers: 0, 1000, 1500, 1825
        Higher Elo = sharper priors from top ladder play.
        """
        available_tiers = [0, 1000, 1500, 1825]
        if elo not in available_tiers:
            print(f"⚠️  Invalid Elo tier {elo}. Available: {available_tiers}. Using closest tier.")
            # Find closest tier
            elo = min(available_tiers, key=lambda x: abs(x - elo))
        cls._elo_tier = elo
        # Clear cached move sets to force reload with new tier
        instance = cls._instance
        if instance:
            keys_to_clear = [k for k in instance._data.keys() if k.startswith('moves_set_')]
            for key in keys_to_clear:
                del instance._data[key]
        print(f"✅ Set Elo tier to {elo} (sharper priors for top ladder)")
    
    def _load_all_data(self):
        """Load all static game data into memory."""
        print("🔄 Loading static game data into cache...")
        
        # Move effects and Pokemon move mappings
        try:
            with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
                self._data['move_effect'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  moves_effect.json not found, using empty dict")
            self._data['move_effect'] = {}
            
        try:
            with open("./poke_env/data/static/moves/gen8pokemon_move_dict.json", "r") as f:
                self._data['pokemon_move_dict'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  gen8pokemon_move_dict.json not found, using empty dict")
            self._data['pokemon_move_dict'] = {}
        
        # Ability effects and Pokemon ability mappings
        try:
            with open("./poke_env/data/static/abilities/ability_effect.json", "r") as f:
                self._data['ability_effect'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  ability_effect.json not found, using empty dict")
            self._data['ability_effect'] = {}
            
        try:
            with open("./poke_env/data/static/abilities/gen8pokemon_ability_dict.json", "r") as f:
                self._data['pokemon_ability_dict'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  gen8pokemon_ability_dict.json not found, using empty dict")
            self._data['pokemon_ability_dict'] = {}
        
        # Item effects
        try:
            with open("./poke_env/data/static/items/item_effect.json", "r") as f:
                self._data['item_effect'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  item_effect.json not found, using empty dict")
            self._data['item_effect'] = {}
        
        # Pokemon item mappings (if needed)
        self._data['pokemon_item_dict'] = {}  # Currently unused
        
        print("✅ Static game data loaded into cache")
    
    def get_move_effect(self) -> Dict[str, Any]:
        """Get cached move effects data."""
        return self._data['move_effect']
    
    def get_pokemon_move_dict(self) -> Dict[str, Any]:
        """Get cached Pokemon move mappings."""
        return self._data['pokemon_move_dict']
    
    def get_ability_effect(self) -> Dict[str, Any]:
        """Get cached ability effects data."""
        return self._data['ability_effect']
    
    def get_pokemon_ability_dict(self) -> Dict[str, Any]:
        """Get cached Pokemon ability mappings."""
        return self._data['pokemon_ability_dict']
    
    def get_item_effect(self) -> Dict[str, Any]:
        """Get cached item effects data."""
        return self._data['item_effect']
    
    def get_pokemon_item_dict(self) -> Dict[str, Any]:
        """Get cached Pokemon item mappings."""
        return self._data['pokemon_item_dict']
    
    @lru_cache(maxsize=5)  # Cache up to 5 different generations
    def get_pokedex(self, gen: int) -> Dict[str, Any]:
        """Get cached Pokedex data for a specific generation."""
        cache_key = f'pokedex_gen{gen}'
        
        if cache_key not in self._data:
            try:
                with open(f"./poke_env/data/static/pokedex/gen{gen}pokedex.json", "r") as f:
                    self._data[cache_key] = json.load(f)
                print(f"✅ Loaded gen{gen} Pokedex data")
            except FileNotFoundError:
                print(f"⚠️  gen{gen}pokedex.json not found, using empty dict")
                self._data[cache_key] = {}
        
        return self._data[cache_key]
    
    @lru_cache(maxsize=10)  # Cache up to 10 different formats
    def get_moves_set(self, format: str, elo_tier: int = None) -> Dict[str, Any]:
        """
        Get cached moves set data for a specific format and Elo tier.
        
        Args:
            format: Battle format (e.g., 'gen9ou')
            elo_tier: Optional Elo tier override. If None, uses class default.
        
        Returns:
            Dict containing move sets for the format.
        """
        if elo_tier is None:
            elo_tier = self._elo_tier
        
        cache_key = f'moves_set_{format}_elo{elo_tier}'
        
        if cache_key not in self._data:
            try:
                if format == 'gen9ou':
                    file_path = f'poke_env/data/static/gen9/ou/sets_{elo_tier}.json'
                else:
                    # Add more formats as needed
                    file_path = f'poke_env/data/static/{format}/sets_{elo_tier}.json'
                
                with open(file_path, 'r') as f:
                    self._data[cache_key] = orjson.loads(f.read())
                print(f"✅ Loaded {format} moves set data (Elo {elo_tier})")
            except FileNotFoundError:
                print(f"⚠️  {format} sets_{elo_tier}.json not found, trying fallback...")
                # Try fallback to 1000 tier
                if elo_tier != 1000:
                    return self.get_moves_set(format, elo_tier=1000)
                else:
                    print(f"⚠️  No sets found for {format}, using empty dict")
                    self._data[cache_key] = {}
        
        return self._data[cache_key]
    
    def clear_cache(self):
        """Clear all cached data (useful for testing)."""
        self._data.clear()
        print("🧹 Cache cleared")


# Global cache instance
_cache = GameDataCache()


def get_cached_move_effect() -> Dict[str, Any]:
    """Get cached move effects data."""
    return _cache.get_move_effect()


def get_cached_pokemon_move_dict() -> Dict[str, Any]:
    """Get cached Pokemon move mappings."""
    return _cache.get_pokemon_move_dict()


def get_cached_ability_effect() -> Dict[str, Any]:
    """Get cached ability effects data."""
    return _cache.get_ability_effect()


def get_cached_pokemon_ability_dict() -> Dict[str, Any]:
    """Get cached Pokemon ability mappings."""
    return _cache.get_pokemon_ability_dict()


def get_cached_item_effect() -> Dict[str, Any]:
    """Get cached item effects data."""
    return _cache.get_item_effect()


def get_cached_pokemon_item_dict() -> Dict[str, Any]:
    """Get cached Pokemon item mappings."""
    return _cache.get_pokemon_item_dict()


def get_cached_pokedex(gen: int) -> Dict[str, Any]:
    """Get cached Pokedex data for a specific generation."""
    return _cache.get_pokedex(gen)


def get_cached_moves_set(format: str, elo_tier: int = None) -> Dict[str, Any]:
    """
    Get cached moves set data for a specific format.
    
    Args:
        format: Battle format (e.g., 'gen9ou')
        elo_tier: Optional Elo tier (0, 1000, 1500, 1825). 
                  If None, uses default (1825).
    
    Returns:
        Dict containing move sets for the format at the specified Elo tier.
    """
    return _cache.get_moves_set(format, elo_tier)


def set_elo_tier(elo: int):
    """
    Set the default Elo tier for move set loading.
    
    Args:
        elo: Elo tier (0, 1000, 1500, 1825)
             Higher = sharper priors from top ladder play
    
    Example:
        # Use 1825 Elo sets for top ladder
        set_elo_tier(1825)
        
        # Use 1500 Elo sets for mid ladder
        set_elo_tier(1500)
    """
    GameDataCache.set_elo_tier(elo)


def clear_data_cache():
    """Clear all cached data."""
    _cache.clear_cache()