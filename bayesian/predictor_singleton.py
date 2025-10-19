# Singleton module for PokemonPredictor to avoid circular imports
# and ensure only one instance is created

import os
import traceback
import sys

_predictor_instances = {}  # Format -> predictor mapping

def _log_message(msg):
    """Log to both stdout and a file for debugging."""
    print(msg, file=sys.stderr)  # Use stderr to ensure it shows
    try:
        with open('/tmp/gen1_predictor_log.txt', 'a') as f:
            f.write(msg + '\n')
    except:
        pass

def get_pokemon_predictor(battle_format=None):
    """Get the shared PokemonPredictor instance for the given format."""
    global _predictor_instances
    
    # Try to detect format from call stack if not provided
    if battle_format is None:
        try:
            # Look for battle format in the call stack
            for frame_info in traceback.extract_stack():
                if 'gen1' in frame_info.line.lower():
                    battle_format = 'gen1ou'
                    break
        except:
            pass
    
    # Default to gen9ou if still not determined
    if battle_format is None:
        battle_format = 'gen9ou'
    
    # Normalize format
    format_key = battle_format.lower()
    
    # Load appropriate predictor
    if format_key not in _predictor_instances:
        if 'gen1' in format_key:
            # Try to load Gen1 predictor
            # Check both possible locations
            cache_path = os.path.join(os.path.dirname(__file__), 'gen1ou_team_predictor.pkl')
            metamon_cache_path = os.path.expanduser('~/metamon_cache/gen1ou_team_predictor.pkl')
            
            # Use metamon cache if it exists
            if os.path.exists(metamon_cache_path):
                cache_path = metamon_cache_path
            if os.path.exists(cache_path):
                # Only log once when first loading
                _log_message(f"Loading Gen1 Bayesian predictor from {cache_path}")
                try:
                    # Load using the proper class method
                    from bayesian.team_predictor import BayesianTeamPredictor
                    predictor = BayesianTeamPredictor(cache_file="gen1ou_team_predictor.pkl")
                    
                    # Override the cache path to use the metamon cache location
                    predictor.cache_path = cache_path
                    
                    # Load the cached model
                    predictor._load_cache()
                    
                    _predictor_instances[format_key] = predictor
                    team_count = getattr(predictor, 'total_teams', 'unknown')
                    _log_message(f"Successfully loaded Gen1 predictor with {team_count} teams")
                except Exception as e:
                    print(f"Failed to load Gen1 predictor: {e}")
                    # Fall back to default
                    try:
                        from bayesian.pokemon_predictor import PokemonPredictor
                        _predictor_instances[format_key] = PokemonPredictor()
                    except:
                        # If even that fails, use a dummy
                        print("Using dummy predictor due to import errors")
                        _predictor_instances[format_key] = None
            else:
                print(f"Gen1 predictor not found at {cache_path}. Using default predictor.")
                print("Train it with: python bayesian/train_gen1_predictor.py")
                # Use default predictor
                from bayesian.pokemon_predictor import PokemonPredictor
                _predictor_instances[format_key] = PokemonPredictor()
        else:
            # Use default Gen9 predictor
            from bayesian.pokemon_predictor import PokemonPredictor
            _predictor_instances[format_key] = PokemonPredictor()
    
    return _predictor_instances[format_key]