#!/usr/bin/env python3
"""
Quick viewer for profiling CSVs with formatted output.
Usage: python scripts/view_profile.py ./battle_log/test_profile
"""
import sys
import os
import pandas as pd


def view_profile(log_dir):
    """View profiling CSVs with nice formatting."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.precision', 2)
    
    # Minimax profile
    minimax_path = os.path.join(log_dir, 'minimax_profile.csv')
    if os.path.exists(minimax_path):
        print("="*80)
        print("DETAILED PROFILING DATA")
        print("="*80)
        df = pd.read_csv(minimax_path)
        
        # Show per algorithm if multiple
        if 'algorithm' in df.columns:
            algos = df['algorithm'].unique()
            print(f"\n📊 Algorithms: {', '.join(algos)}")
            print(f"📊 Total: {len(df)} turns profiled\n")
            
            for algo in algos:
                df_algo = df[df['algorithm'] == algo]
                print(f"\n{'='*80}")
                print(f"{algo.upper()} - {len(df_algo)} turns")
                print('='*80)

        # Summary statistics
        print("Summary Statistics:")
        print(df[['elapsed_s', 'init_ms', 'expansion_ms', 'llm_calls_ms', 'batch_eval_ms']].describe())

        # Cache performance over time
        print("\n\nCache Performance (by turn):")
        df['tt_hit_rate'] = df['tt_hits'] / (df['tt_hits'] + df['tt_misses'] + 0.001)
        df['q_hit_rate'] = df['q_hits'] / (df['q_hits'] + df['q_misses'] + 0.001)
        print(df[['turn', 'elapsed_s', 'tt_hits', 'tt_misses', 'tt_hit_rate', 'q_hits', 'q_misses', 'q_hit_rate']].to_string(index=False))

        # Sim performance
        print("\n\nSimulator Performance:")
        print(df[['turn', 'sim_steps', 'sim_step_ms', 'nodes_created']].to_string(index=False))

        # Phase breakdown
        print("\n\nPhase Time Breakdown (ms):")
        print(df[['turn', 'init_ms', 'expansion_ms', 'cache_ops_ms', 'llm_calls_ms', 'batch_eval_ms', 'action_selection_ms', 'other_ms']].to_string(index=False))

    # General metrics
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        print("\n" + "="*80)
        print("GENERAL METRICS (All Algorithms)")
        print("="*80)
        df = pd.read_csv(metrics_path)
        print(f"\n📊 {len(df)} moves logged\n")

        # Group by algorithm
        print("By Algorithm:")
        summary = df.groupby('algorithm').agg({
            'latency_ms': ['count', 'mean', 'std', 'min', 'max'],
            'json_ok': 'mean',
            'near_timeout': 'sum'
        })
        print(summary)

if __name__ == '__main__':
    log_dir = sys.argv[1] if len(sys.argv) > 1 else './battle_log/test_profile'
    view_profile(log_dir)

