import asyncio
from time import sleep
from tqdm import tqdm
import argparse
import os, sys


# Add the current directory to Python path (since we're in project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from common import *
from poke_env.player.team_util import get_llm_player, get_metamon_teams, load_random_team

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--reasoning_effort", type=str, default="medium", choices=["low", "medium", "high"],
                    help="Reasoning effort for gpt-5/o4-mini model (low=faster, high=better quality)")
parser.add_argument("--prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--online", action="store_true", help="Use online Showdown server (ladder)")
parser.add_argument("--server", type=str, default="showdown", choices=["showdown", "pokeagent"],
                    help="Select server: official showdown or PokéAgent competition server")
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
parser.add_argument("--backend", type=str, default="openai/gpt-4o", choices=[
    # OpenAI models (direct API) - Latest
    "gpt-5-pro", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3-mini", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4-turbo", "gpt-4",
    # OpenAI models (via OpenRouter) - Latest
    "openai/gpt-5-pro", "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/o4-mini", "openai/o3-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4o", "openai/gpt-4o-2024-11-20", "openai/gpt-4-turbo", "openai/gpt-4",
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
parser.add_argument("--log_dir", type=str, default="./battle_log/ladder")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--USERNAME", type=str, default='')
parser.add_argument("--PASSWORD", type=str, default='')
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--max_tokens", type=int, default=300)
parser.add_argument("--K", type=int, default=None, help="For sc: samples; for minimax/TOT: search breadth/depth")
parser.add_argument("--team_pool", type=str, default="competitive", choices=["competitive","modern_replays","pokeagent_modern_replays"],
                    help="Choose team source pool for the ladder")
parser.add_argument("--min_elo", type=int, default=None, help="Filter teams by minimum Elo (if available in filenames)")
parser.add_argument("--max_elo", type=int, default=None, help="Filter teams by maximum Elo (if available in filenames)")
parser.add_argument("--move_time_limit", type=float, default=8.0, help="Time limit per move in seconds (default: 8.0)")
parser.add_argument("--elo_tier", type=int, default=1825, choices=[0, 1000, 1500, 1825],
                    help="Elo tier for move sets (default: 1825 = top ladder, sharper priors)")
args = parser.parse_args()
    
async def main():
    from pokechamp.data_cache import set_elo_tier
    
    # Set Elo tier for move sets (use sharper priors for top ladder)
    set_elo_tier(args.elo_tier)
    
    player = get_llm_player(args, 
                            args.backend, 
                            args.prompt_algo, 
                            args.name, 
                            device=args.device,
                            battle_format=args.battle_format, 
                            online=args.online, 
                            server=args.server,
                            USERNAME=args.USERNAME, 
                            PASSWORD=args.PASSWORD)
    # Propagate max_tokens to player
    try:
        player.max_tokens = int(max(1, args.max_tokens))
    except Exception:
        pass
    # Optionally set K (samples or depth)
    try:
        if args.K is not None:
            player.K = int(max(1, args.K))
    except Exception:
        pass
    # Set move time limit
    try:
        player.move_time_limit_s = float(max(1.0, args.move_time_limit))
    except Exception:
        pass
    
    teamloader = get_metamon_teams(args.battle_format, args.team_pool, min_elo=args.min_elo, max_elo=args.max_elo)
    
    if not 'random' in args.battle_format:
        # Set teamloader on player for rejection recovery
        player.set_teamloader(teamloader)
        player.update_team(teamloader.yield_team())

    # Playing n_challenges games on the ladder
    n_challenges = args.N
    pbar = tqdm(total=n_challenges)
    wins = 0
    for i in range(n_challenges):
        print('starting ladder')
        await player.ladder(1)
        winner = 'opponent'
        if player.win_rate > 0: 
            winner = args.name
            wins += 1
        if not 'random' in args.battle_format:
            player.update_team(teamloader.yield_team())
        sleep(30)
        pbar.set_description(f"{wins/(i+1)*100:.2f}%")
        pbar.update(1)
        print(winner)
        player.reset_battles()
    print(f'player 2 winrate: {wins/n_challenges*100}')

if __name__ == "__main__":
    asyncio.run(main())