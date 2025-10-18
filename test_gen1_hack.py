#!/usr/bin/env python3
"""
DIRECT comparison: WITH hack vs WITHOUT hack
This is what you probably wanted to see!
"""
import asyncio
import sys
import os
from argparse import ArgumentParser, Namespace

sys.path.insert(0, os.path.abspath('.'))

# Valid Gen1 team
GEN1_TEAM = """Tauros
- Body Slam
- Hyper Beam
- Earthquake
- Blizzard

Snorlax
- Body Slam
- Earthquake
- Rest
- Hyper Beam

Chansey
- Soft-Boiled
- Thunder Wave
- Ice Beam
- Thunderbolt

Starmie
- Psychic
- Blizzard
- Thunder Wave
- Recover

Alakazam
- Psychic
- Thunder Wave
- Recover
- Seismic Toss

Exeggutor
- Psychic
- Sleep Powder
- Explosion
- Stun Spore"""

async def test_direct(num_battles: int = 10, backend: str = "openai/gpt-4o"):
    """Run direct comparison: WITH hack vs WITHOUT hack"""
    from poke_env import AccountConfiguration
    from poke_env.player.team_util import get_llm_player
    from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
    from poke_env.player import Player
    from poke_env.environment.pokemon import Pokemon
    import time
    
    # Disable team rejection fallback
    Player._max_team_rejections = 1
    
    # Create args
    args = Namespace(
        player_backend=backend,
        player_name="pokechamp",
        opponent_backend=backend,
        opponent_name="pokechamp",
        temperature=0.7,
        opponent_temperature=0.7,
        log_dir=None,
        reasoning_effort='low',
        verbose=False
    )
    
    # Save original guess_stats
    original_guess = Pokemon.guess_stats
    
    # Create a wrapper that tracks which player uses hack
    hack_enabled = {}
    
    def guess_with_tracking(self, *args, **kwargs):
        # Check if this Pokemon belongs to the hack-enabled player
        if hasattr(self, '_battle') and self._battle:
            player_name = self._battle.player_username
            if player_name in hack_enabled and hack_enabled[player_name]:
                # Use Gen1 hack
                if 'gen1' in self._battle._format.lower():
                    from poke_env.data.static.gen1_common_sets import get_gen1_stats
                    return get_gen1_stats(self.species)
        # Otherwise use default
        return (
            {'hp': 85, 'atk': 85, 'def': 85, 'spa': 85, 'spd': 85, 'spe': 85},
            'Hardy'
        )
    
    Pokemon.guess_stats = guess_with_tracking
    
    try:
        print("\n" + "="*60)
        print("DIRECT COMPARISON: WITH hack vs WITHOUT hack")
        print(f"Running {num_battles} battles...")
        print("="*60 + "\n")
        
        # Create players
        id1 = f"PHack{int(time.time()) % 1000}"
        id2 = f"PNoHack{int(time.time()) % 1000 + 1}"
        
        player_with_hack = get_llm_player(
            args,
            backend,
            "pokechamp",
            "pokechamp",
            KEY='',
            battle_format="gen1ou",
            USERNAME=id1
        )
        
        player_no_hack = get_llm_player(
            args,
            backend,
            "pokechamp",
            "pokechamp", 
            KEY='',
            battle_format="gen1ou",
            USERNAME=id2
        )
        
        # Mark which player has hack
        hack_enabled[id1] = True
        hack_enabled[id2] = False
        
        # Set teams
        player_with_hack._team = ConstantTeambuilder(GEN1_TEAM)
        player_no_hack._team = ConstantTeambuilder(GEN1_TEAM)
        
        print(f"Player 1 ({id1}): WITH Gen1 hack (competitive EVs)")
        print(f"Player 2 ({id2}): WITHOUT hack (generic 85 EVs)")
        print()
        
        # Run battles
        await player_with_hack.battle_against(player_no_hack, n_battles=num_battles)
        
        # Results
        total = player_with_hack.n_won_battles + player_no_hack.n_won_battles
        if total > 0:
            hack_win_rate = player_with_hack.n_won_battles / total * 100
            no_hack_win_rate = player_no_hack.n_won_battles / total * 100
            
            print("\n" + "="*60)
            print("RESULTS:")
            print(f"WITH hack:    {player_with_hack.n_won_battles}/{total} ({hack_win_rate:.1f}%)")
            print(f"WITHOUT hack: {player_no_hack.n_won_battles}/{total} ({no_hack_win_rate:.1f}%)")
            print()
            
            if hack_win_rate > no_hack_win_rate:
                improvement = hack_win_rate - 50
                print(f"✅ The hack WORKS! +{improvement:.1f}% win rate improvement")
            elif hack_win_rate < no_hack_win_rate:
                print("❌ Unexpected: hack performed worse")
            else:
                print("➖ No difference detected")
            print("="*60)
            
    finally:
        # Restore original
        Pokemon.guess_stats = original_guess

async def main():
    parser = ArgumentParser()
    parser.add_argument("--N", type=int, default=10, help="Number of battles")
    parser.add_argument("--backend", default="openai/gpt-4o", help="LLM backend")
    args = parser.parse_args()
    
    await test_direct(args.N, args.backend)

if __name__ == "__main__":
    asyncio.run(main())
