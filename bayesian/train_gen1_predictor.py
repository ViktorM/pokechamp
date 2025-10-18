#!/usr/bin/env python3
"""
Train the Bayesian Team Predictor on Gen1OU modern_replays dataset.
This creates a Gen1-specific predictor for better battle predictions.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bayesian.team_predictor import BayesianTeamPredictor, get_metamon_teams
from tqdm import tqdm


class Gen1BayesianPredictor(BayesianTeamPredictor):
    """Gen1-specific Bayesian predictor."""
    
    def __init__(self):
        super().__init__(cache_file="gen1ou_team_predictor.pkl")
        
    def _train_from_data(self):
        """Override to train on Gen1OU data instead of Gen9OU."""
        # Get Gen1OU team data
        team_set = get_metamon_teams("gen1ou", "modern_replays")
        team_files = team_set.team_files
        
        print(f"Training on {len(team_files)} Gen1OU teams...")
        print("This is faster than Gen9 training due to fewer teams (13k vs 1M)")
        
        for file_path in tqdm(team_files, desc="Processing Gen1 teams"):
            try:
                team_data = self.parser.parse_team_file(file_path)
                self._update_counts(team_data)
                self.total_teams += 1
                
                # Progress reporting
                if self.total_teams % 1000 == 0:
                    print(f"Processed {self.total_teams} teams...")
                    
            except Exception as e:
                # Gen1 has different format, some parsing errors expected
                continue
        
        print(f"Trained on {self.total_teams} Gen1OU teams")
        print(f"Found {len(self.species_counts)} unique Gen1 species")


def main():
    """Train the Gen1 predictor."""
    print("=" * 60)
    print("TRAINING BAYESIAN TEAM PREDICTOR FOR GEN1 OU")
    print("=" * 60)
    
    # Set up environment
    cache_dir = os.getenv('METAMON_CACHE_DIR', '/tmp/metamon_cache')
    print(f"Using cache directory: {cache_dir}")
    
    # Check if Gen1 data exists
    gen1_path = os.path.join(cache_dir, "teams", "modern_replays", "gen1ou")
    if not os.path.exists(gen1_path):
        print(f"ERROR: Gen1OU data not found at {gen1_path}")
        print("Please download it first with:")
        print("  python -m poke_env.data.download teams --formats gen1ou")
        return
    
    # Initialize predictor
    predictor = Gen1BayesianPredictor()
    
    # Start training
    start_time = time.time()
    
    try:
        # Force retrain to create Gen1-specific model
        predictor.load_and_train(force_retrain=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print("GEN1 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Training time: {duration:.1f} seconds")
        print(f"Total teams processed: {predictor.total_teams:,}")
        print(f"Unique species found: {len(predictor.species_counts):,}")
        print(f"Model cached at: {predictor.cache_path}")
        
        # Show Gen1-specific statistics
        print(f"\nTop 10 most common Gen1 Pokemon:")
        for species, count in predictor.species_counts.most_common(10):
            percentage = (count / predictor.total_teams) * 100
            print(f"  {species}: {count:,} ({percentage:.1f}%)")
        
        # Test Gen1 predictions
        print(f"\n{'='*40}")
        print("TESTING GEN1 PREDICTIONS")
        print(f"{'='*40}")
        
        # Test case: Classic Gen1 core
        test_revealed = ["Tauros", "Snorlax"]
        print(f"\nGiven revealed Pokemon: {test_revealed}")
        
        predictions = predictor.predict_unrevealed_pokemon(test_revealed, max_predictions=5)
        if predictions:
            print("Most likely unrevealed teammates:")
            for species, prob in predictions:
                print(f"  {species}: {prob:.4f}")
            
            # Test move prediction for Tauros
            tauros_moves = predictor.predict_component_probabilities(
                "Tauros", 
                teammates=test_revealed,
                observed_moves=["Body Slam"]
            )
            
            if 'moves' in tauros_moves:
                print(f"\nPredicted moves for Tauros (given Body Slam):")
                for move, prob in tauros_moves['moves'][:5]:
                    print(f"  {move}: {prob:.2%}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Processed {predictor.total_teams} teams before interruption.")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
