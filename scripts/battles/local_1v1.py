import asyncio
from tqdm import tqdm
import os
import sys
import argparse

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from common import *
from poke_env.player.team_util import get_llm_player, get_metamon_teams, load_random_team

parser = argparse.ArgumentParser()

# Player arguments
parser.add_argument("--player_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--player_backend", type=str, default="openai/gpt-4o", choices=[
    # OpenAI models (direct API)
    "gpt-5-pro", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3-mini", "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-11-20", "gpt-4-turbo", "gpt-4",
    # OpenAI models (via OpenRouter)
    "openai/gpt-5-pro", "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/o4-mini", "openai/o3-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4o-2024-11-20", "openai/gpt-4-turbo", "openai/gpt-4",
    # Anthropic models
    "anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5", "anthropic/claude-opus-4.1", "anthropic/claude-3.5-sonnet",
    # Google models
    "gemini-2.5-flash", "gemini-2.5-pro", "google/gemini-2.5-flash-preview-09-2025", "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro",
    # xAI models - Grok-4 only for now
    "x-ai/grok-4", "x-ai/grok-4-fast",
    # Meta models
    "meta-llama/llama-4-maverick", "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.1-70b-instruct",
    # Local models (via OpenRouter)
    "llama", 'None'
])
parser.add_argument("--player_name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--player_device", type=int, default=0)

# Opponent arguments
parser.add_argument("--opponent_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--opponent_backend", type=str, default="openai/gpt-4o", choices=[
    # OpenAI models (direct API)
    "gpt-5-pro", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3-mini", "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-11-20", "gpt-4-turbo", "gpt-4",
    # OpenAI models (via OpenRouter)
    "openai/gpt-5-pro", "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/o4-mini", "openai/o3-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4o-2024-11-20", "openai/gpt-4-turbo", "openai/gpt-4",
    # Anthropic models
    "anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5", "anthropic/claude-opus-4.1", "anthropic/claude-3.5-sonnet",
    # Google models
    "gemini-2.5-flash", "gemini-2.5-pro", "google/gemini-2.5-flash-preview-09-2025", "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro",
    # xAI models - Grok-4 only for now
    "x-ai/grok-4", "x-ai/grok-4-fast",
    # Meta models
    "meta-llama/llama-4-maverick", "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.1-70b-instruct",
    # Local models (via OpenRouter)
    "llama", 'None'
])
parser.add_argument("--opponent_name", type=str, default='pokellmon', choices=bot_choices)
parser.add_argument("--opponent_device", type=int, default=0)

# Shared arguments
parser.add_argument("--temperature", type=float, default=0.3, help="Default temperature for both players")
parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"],
                    help="Reasoning effort for the supported models (low=faster, high=better quality)")
parser.add_argument("--battle_format", default="gen9ou", choices=[
    # PokéAgent Challenge formats
    "gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen9ou",
    # VGC formats
    "gen9vgc2024regg",
    # Random battle formats
    "gen8randombattle", "gen9randombattle",
    # Other OU formats
    "gen8ou"
])
parser.add_argument("--log_dir", type=str, default="./battle_log/one_vs_one")
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--verbose", action="store_true", help="Show detailed turn-by-turn battle information")
parser.add_argument("--profile", action="store_true", help="Show timing breakdown for minimax (without verbose debug output)")
parser.add_argument("--max_tokens", type=int, default=300, help="Default max tokens for both players")
parser.add_argument("--move_time_limit", type=float, default=8.0, help="Time limit per move in seconds (default: 8.0)")
parser.add_argument("--player_K", type=int, default=None, help="For sc: samples; for minimax/TOT: search breadth/depth (player)")
parser.add_argument("--opponent_K", type=int, default=None, help="For sc: samples; for minimax/TOT: search breadth/depth (opponent)")
parser.add_argument("--elo_tier", type=int, default=1825, choices=[0, 1000, 1500, 1825],
                    help="Elo tier for move sets (default: 1825 = top ladder, sharper priors)")

# Two-tier temperature/token configuration (shared, optional overrides)
parser.add_argument("--temp_action", type=float, default=None,
                    help="Temperature for structured JSON decisions (default: 0.0)")
parser.add_argument("--mt_action", type=int, default=None,
                    help="Max tokens for JSON decisions (default: 16)")
parser.add_argument("--temp_expand", type=float, default=None,
                    help="Temperature for reasoning/expansion (default: uses --temperature)")
parser.add_argument("--mt_expand", type=int, default=None,
                    help="Max tokens for expansion (default: uses --max_tokens)")

args = parser.parse_args()

async def main():
    import logging
    from pokechamp.data_cache import set_elo_tier
    
    # Set Elo tier for move sets
    set_elo_tier(args.elo_tier)
    
    # Set logging level based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    
    # Create unique player IDs to avoid conflicts
    import time
    timestamp = str(int(time.time() * 100) % 10000)  # 4 digits max
    
    player = get_llm_player(args, 
                            args.player_backend, 
                            args.player_prompt_algo, 
                            args.player_name, 
                            device=args.player_device,
                            PNUMBER1=PNUMBER1 + timestamp,  # for name uniqueness locally
                            battle_format=args.battle_format,
                            log_level=log_level)
    # Apply local tuning knobs
    try:
        player.max_tokens = int(max(1, args.max_tokens))
        if args.player_K is not None:
            player.K = int(max(1, args.player_K))
        player.move_time_limit_s = float(max(1.0, args.move_time_limit))
    except Exception:
        pass
    
    opponent = get_llm_player(args, 
                            args.opponent_backend, 
                            args.opponent_prompt_algo, 
                            args.opponent_name, 
                            device=args.opponent_device,
                            PNUMBER1=PNUMBER1 + str(int(timestamp) + 1),  # for name uniqueness locally
                            battle_format=args.battle_format,
                            log_level=log_level)
    try:
        opponent.max_tokens = int(max(1, args.max_tokens))
        if args.opponent_K is not None:
            opponent.K = int(max(1, args.opponent_K))
        opponent.move_time_limit_s = float(max(1.0, args.move_time_limit))
    except Exception:
        pass

    # Use old teamloader for player, modern for opponent
    player_teamloader = get_metamon_teams(args.battle_format, "competitive")
    opponent_teamloader = get_metamon_teams(args.battle_format, "modern_replays")
    
    if not 'random' in args.battle_format:
        # Set teamloader on players for rejection recovery
        player.set_teamloader(player_teamloader)
        opponent.set_teamloader(opponent_teamloader)
        
        player.update_team(player_teamloader.yield_team())
        opponent.update_team(opponent_teamloader.yield_team())

    # play against bot for battles
    N = args.N
    pbar = tqdm(total=N)
    for i in range(N):
        # Store win counts before battle
        player_wins_before = player.n_won_battles
        opponent_wins_before = opponent.n_won_battles
        
        x = np.random.randint(0, 100)
        if x > 50:
            await player.battle_against(opponent, n_battles=1)
        else:
            await opponent.battle_against(player, n_battles=1)
            
        # Determine who won this battle
        if player.n_won_battles > player_wins_before:
            winner = f"🏆 PLAYER ({args.player_prompt_algo}_{args.player_name}_{args.player_backend}) WON"
        elif opponent.n_won_battles > opponent_wins_before:
            winner = f"🏆 OPPONENT ({args.opponent_prompt_algo}_{args.opponent_name}_{args.opponent_backend}) WON"
        else:
            winner = "⚔️  Draw or error"
        
        print(f"\nBattle {i+1}: {winner}")
        print(f"Player ({args.player_prompt_algo}_{args.player_name}_{args.player_backend}): {player.n_won_battles}W-{player.n_lost_battles}L | Opponent ({args.opponent_prompt_algo}_{args.opponent_name}_{args.opponent_backend}): {opponent.n_won_battles}W-{opponent.n_lost_battles}L")
        
        if not 'random' in args.battle_format:
            player.update_team(player_teamloader.yield_team())
            opponent.update_team(opponent_teamloader.yield_team())
        pbar.set_description(f"{player.win_rate*100:.2f}%")
        pbar.update(1)
    
    print(f'\n{"="*80}')
    print(f'FINAL RESULTS - Algorithm + Model Comparison:')
    print(f'{"="*80}')
    print(f'PLAYER:   {args.player_prompt_algo:8}_{args.player_name:12}_{args.player_backend:30} = {player.n_won_battles}W-{player.n_lost_battles}L ({player.win_rate*100:.1f}%)')
    print(f'OPPONENT: {args.opponent_prompt_algo:8}_{args.opponent_name:12}_{args.opponent_backend:30} = {opponent.n_won_battles}W-{opponent.n_lost_battles}L ({opponent.win_rate*100:.1f}%)')
    print(f'{"="*80}')
    
    # Show winner
    if player.win_rate > opponent.win_rate:
        print(f'\n🏆 Winner: {args.player_prompt_algo}_{args.player_name}_{args.player_backend}')
    elif opponent.win_rate > player.win_rate:
        print(f'\n🏆 Winner: {args.opponent_prompt_algo}_{args.opponent_name}_{args.opponent_backend}')
    else:
        print(f'\n🤝 Tied matchup!')
    
    # Helpful reminder for pure model testing
    if args.player_name != args.opponent_name:
        print(f'\n💡 Note: Different algorithms used ({args.player_name} vs {args.opponent_name}).')
        print(f'   For pure model comparison, use: --player_name {args.player_name} --opponent_name {args.player_name}')


if __name__ == "__main__":
    asyncio.run(main())
