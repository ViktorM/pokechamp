"""
Gen1 OU Common Sets - Quick hack for better predictions without full Bayesian model.
Based on typical Gen1 OU competitive sets from high-level play.
"""

# Gen1 doesn't have natures, so we use 'Hardy' (neutral) for all
# EVs in Gen1 are typically simple 252/252/4 spreads
# Moves are the most commonly seen competitive sets

GEN1_COMMON_SETS = {
    'tauros': {
        'moves': ['bodyslam', 'hyperbeam', 'earthquake', 'blizzard'],
        'evs': {'hp': 252, 'atk': 252, 'def': 4, 'spa': 0, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.85  # 85% usage in Gen1 OU
    },
    'snorlax': {
        'moves': ['bodyslam', 'selfdestruct', 'earthquake', 'hyperbeam'],
        'evs': {'hp': 252, 'atk': 252, 'def': 4, 'spa': 0, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.80
    },
    'chansey': {
        'moves': ['softboiled', 'icebeam', 'thunderwave', 'thunderbolt'],
        'evs': {'hp': 252, 'atk': 0, 'def': 252, 'spa': 4, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.75
    },
    'exeggutor': {
        'moves': ['psychic', 'sleeppowder', 'explosion', 'stunspore'],
        'evs': {'hp': 252, 'atk': 0, 'def': 0, 'spa': 252, 'spd': 4, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.60
    },
    'alakazam': {
        'moves': ['psychic', 'thunderwave', 'recover', 'seismictoss'],
        'evs': {'hp': 252, 'atk': 0, 'def': 4, 'spa': 252, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.55
    },
    'starmie': {
        'moves': ['blizzard', 'thunderbolt', 'thunderwave', 'recover'],
        'evs': {'hp': 252, 'atk': 0, 'def': 4, 'spa': 252, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.50
    },
    'zapdos': {
        'moves': ['thunderbolt', 'drillpeck', 'thunderwave', 'agility'],
        'evs': {'hp': 252, 'atk': 0, 'def': 4, 'spa': 252, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.45
    },
    'rhydon': {
        'moves': ['earthquake', 'rockslide', 'bodyslam', 'substitute'],
        'evs': {'hp': 252, 'atk': 252, 'def': 4, 'spa': 0, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.40
    },
    'gengar': {
        'moves': ['hypnosis', 'explosion', 'thunderbolt', 'nightshade'],
        'evs': {'hp': 252, 'atk': 0, 'def': 4, 'spa': 252, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.35
    },
    'jynx': {
        'moves': ['lovelykiss', 'blizzard', 'psychic', 'rest'],
        'evs': {'hp': 252, 'atk': 0, 'def': 4, 'spa': 252, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.30
    },
    'lapras': {
        'moves': ['blizzard', 'thunderbolt', 'bodyslam', 'sing'],
        'evs': {'hp': 252, 'atk': 0, 'def': 128, 'spa': 128, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.25
    },
    'golem': {
        'moves': ['earthquake', 'rockslide', 'bodyslam', 'explosion'],
        'evs': {'hp': 252, 'atk': 252, 'def': 4, 'spa': 0, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.20
    },
    'slowbro': {
        'moves': ['psychic', 'thunderwave', 'amnesia', 'rest'],
        'evs': {'hp': 252, 'atk': 0, 'def': 252, 'spa': 4, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.20
    },
    'cloyster': {
        'moves': ['blizzard', 'explosion', 'clamp', 'hyperbeam'],
        'evs': {'hp': 252, 'atk': 128, 'def': 0, 'spa': 128, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.15
    },
    'persian': {
        'moves': ['slash', 'hyperbeam', 'thunderbolt', 'bubblebeam'],
        'evs': {'hp': 252, 'atk': 252, 'def': 4, 'spa': 0, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.10
    },
    'jolteon': {
        'moves': ['thunderbolt', 'thunderwave', 'pinmissile', 'doublekick'],
        'evs': {'hp': 252, 'atk': 0, 'def': 4, 'spa': 252, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.10
    },
    'victreebel': {
        'moves': ['razorleaf', 'sleeppowder', 'stunspore', 'wrap'],
        'evs': {'hp': 252, 'atk': 0, 'def': 4, 'spa': 252, 'spd': 0, 'spe': 0},
        'nature': 'Hardy', 
        'usage_rate': 0.08
    },
    'dragonite': {
        'moves': ['wrap', 'hyperbeam', 'blizzard', 'thunderwave'],
        'evs': {'hp': 252, 'atk': 128, 'def': 0, 'spa': 128, 'spd': 0, 'spe': 0},
        'nature': 'Hardy',
        'usage_rate': 0.08
    }
}

# Alternative sets for some Pokemon (less common but viable)
GEN1_ALTERNATIVE_SETS = {
    'snorlax': [
        {
            'moves': ['bodyslam', 'reflect', 'rest', 'earthquake'],
            'evs': {'hp': 252, 'atk': 252, 'def': 4, 'spa': 0, 'spd': 0, 'spe': 0}
        },
        {
            'moves': ['bodyslam', 'hyperbeam', 'counter', 'selfdestruct'],
            'evs': {'hp': 252, 'atk': 252, 'def': 4, 'spa': 0, 'spd': 0, 'spe': 0}
        }
    ],
    'chansey': [
        {
            'moves': ['softboiled', 'seismictoss', 'thunderwave', 'icebeam'],
            'evs': {'hp': 252, 'atk': 0, 'def': 252, 'spa': 4, 'spd': 0, 'spe': 0}
        }
    ],
    'exeggutor': [
        {
            'moves': ['psychic', 'sleeppowder', 'explosion', 'doubleedge'],
            'evs': {'hp': 252, 'atk': 128, 'def': 0, 'spa': 128, 'spd': 0, 'spe': 0}
        }
    ]
}

def get_gen1_set(pokemon_name: str) -> dict:
    """Get the most common Gen1 set for a Pokemon."""
    pokemon_name = pokemon_name.lower()
    return GEN1_COMMON_SETS.get(pokemon_name, None)

def get_gen1_moves(pokemon_name: str, revealed_moves: list = None) -> list:
    """Get predicted moves for a Gen1 Pokemon, considering revealed moves."""
    pokemon_name = pokemon_name.lower()
    
    if pokemon_name not in GEN1_COMMON_SETS:
        return []
    
    common_moves = GEN1_COMMON_SETS[pokemon_name]['moves']
    
    if not revealed_moves:
        return common_moves
    
    # If we've seen some moves, adjust predictions
    revealed_moves = [m.lower().replace(' ', '').replace('-', '') for m in revealed_moves]
    remaining_moves = [m for m in common_moves if m not in revealed_moves]
    
    # Combine revealed and predicted moves
    return revealed_moves + remaining_moves[:4-len(revealed_moves)]

def get_gen1_stats(pokemon_name: str) -> tuple:
    """Get predicted EVs and nature for a Gen1 Pokemon."""
    pokemon_name = pokemon_name.lower()
    
    if pokemon_name in GEN1_COMMON_SETS:
        set_data = GEN1_COMMON_SETS[pokemon_name]
        evs = set_data['evs']
        # Convert dict to list format expected by the game
        ev_list = [evs['hp'], evs['atk'], evs['def'], evs['spa'], evs['spd'], evs['spe']]
        return ev_list, set_data['nature']
    
    # Default fallback
    return [85, 85, 85, 85, 85, 85], 'Hardy'
