#!/usr/bin/env python3
"""Quick verification that Gen1 hack is working."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle
from poke_env.data.static.gen1_common_sets import GEN1_COMMON_SETS, get_gen1_stats, get_gen1_moves

# Create a mock battle with gen1ou format
class MockBattle:
    def __init__(self):
        self._format = 'gen1ou'

# Test stat guessing
print("Testing Gen1 Stat Prediction:")
print("="*50)

battle = MockBattle()

# Test a few common Gen1 Pokemon
test_pokemon = ['Tauros', 'Snorlax', 'Chansey', 'Alakazam', 'Pikachu']

for poke_name in test_pokemon:
    # Create mock Pokemon
    pokemon = Pokemon(species=poke_name, gen=1)
    
    # Get stats using guess_stats with battle context
    evs, nature = pokemon.guess_stats(battle=battle)
    
    # Get expected stats from our Gen1 sets
    expected_evs, expected_nature = get_gen1_stats(poke_name)
    
    print(f"\n{poke_name}:")
    print(f"  Predicted EVs: {evs}")
    print(f"  Expected EVs:  {expected_evs}")
    print(f"  Match: {'✅' if evs == expected_evs else '❌'}")
    
    if poke_name.lower() in GEN1_COMMON_SETS:
        print(f"  Usage Rate: {GEN1_COMMON_SETS[poke_name.lower()]['usage_rate']*100:.0f}%")

# Test move prediction
print("\n\nTesting Gen1 Move Prediction:")
print("="*50)

# Test with no revealed moves
print("\nTauros (no moves revealed):")
predicted = get_gen1_moves('Tauros', [])
print(f"  Predicted: {predicted}")
print(f"  Expected: {GEN1_COMMON_SETS['tauros']['moves']}")

# Test with some revealed moves
print("\nSnorlax (Body Slam revealed):")
predicted = get_gen1_moves('Snorlax', ['bodyslam'])
print(f"  Predicted: {predicted}")
print(f"  Should start with bodyslam and add 3 more")

print("\n✅ Verification complete!")
