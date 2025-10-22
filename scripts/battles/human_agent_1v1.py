import asyncio
from tqdm import tqdm
import argparse

from common import *
from poke_env.player.team_util import get_llm_player, load_random_team

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--prompt_algo", default="minimax", choices=prompt_algos)
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
    "gpt-5-pro", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3-mini", "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-11-20", "gpt-4-turbo", "gpt-4",
    # OpenAI models (via OpenRouter) - Latest
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
parser.add_argument("--log_dir", type=str, default="./battle_log/ladder")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random'])
parser.add_argument("--elo_tier", type=int, default=1825, choices=[0, 1000, 1500, 1825],
                    help="Elo tier for move sets (default: 1825 = top ladder, sharper priors)")

# Two-tier temperature/token configuration (optional overrides)
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
    from pokechamp.data_cache import set_elo_tier
    
    # Set Elo tier for move sets
    set_elo_tier(args.elo_tier)

    opponent = get_llm_player(args, 
                            args.backend, 
                            args.prompt_algo, 
                            args.name, 
                            PNUMBER1=PNUMBER1,
                            battle_format=args.battle_format)
    if not 'random' in args.battle_format:
        opponent.update_team(load_random_team())                      
    
    # Playing 5 games on local
    for i in tqdm(range(5)):
        await opponent.ladder(1)
        if not 'random' in args.battle_format:
            opponent.update_team(load_random_team())

if __name__ == "__main__":
    asyncio.run(main())
