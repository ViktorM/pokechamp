#!/usr/bin/env python3
"""
Battle Log Analysis Script
Analyzes Pokemon battle performance from CSV metrics
"""

import pandas as pd
import os
import sys
from datetime import datetime

def load_data(log_dir='battle_log/ladder'):
    """Load battle metrics and summaries."""
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    summary_path = os.path.join(log_dir, 'metrics_summary.csv')
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found!")
        return None, None
        
    # Load per-move metrics
    df = pd.read_csv(metrics_path)
    
    # Require proper logging format
    required_columns = ['battle_tag', 'turn', 'backend', 'algorithm', 'player_role', 'player_username']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Old log format detected. Missing columns: {missing_columns}")
        print("Please regenerate logs with the updated logging system.")
        return None, None
    
    # Create combined model+algo identifier
    df['model_algo'] = df['backend'] + ' (' + df['algorithm'] + ')'
    
    # Load battle summaries if exists
    battles = None
    if os.path.exists(summary_path):
        battles = pd.read_csv(summary_path, header=None)
        
        # Require new format with player info (17 columns)
        if len(battles.columns) != 17:
            print(f"Error: Old summary format detected. Expected 17 columns, got {len(battles.columns)}")
            print("Please regenerate logs with the updated logging system.")
            return df, None
        
        battles.columns = ['battle_id', 'player_role', 'backend', 'algorithm', 
                          'player_username', 'won', 'turns', 'total_moves', 'avg_latency',
                          'llm_calls', 'json_errors', 'switches', 'timeouts', 'avoided', 
                          'osc', 'prog_trend', 'belief_entropy_trend']
            
        # 'won' column is already 1 for win, 0 for loss (from battle.won)
        # Add player_index based on row order within each battle
        battles['player_index'] = battles.groupby('battle_id').cumcount()
    
    return df, battles

def analyze_model_performance(df):
    """Analyze performance by LLM backend + algorithm."""
    print("\n" + "="*60)
    print("MODEL + ALGORITHM PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Group by model+algo combination
    model_stats = df.groupby('model_algo').agg({
        'json_ok': ['count', 'mean'],
        'latency_ms': ['mean', 'std', 'min', 'max', 'median'],
        'near_timeout': ['sum', 'mean'],
        'tokens_prompt': 'mean',
        'tokens_completion': 'mean',
        'fallback_reason': lambda x: (x != '').sum()  # Count of fallbacks
    }).round(2)
    
    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    
    print("\nDETAILED MODEL PERFORMANCE METRICS:")
    print("-" * 80)
    
    for model_algo in model_stats.index:
        print(f"\n{model_algo}:")
        print(f"{'Metric':<35} {'Value':>15}")
        print("-" * 52)
        
        stats = model_stats.loc[model_algo]
        
        # Extract and format metrics
        metrics = [
            ('Total Moves', int(stats['json_ok_count'])),
            ('JSON Success Rate', f"{stats['json_ok_mean']:.1%}"),
            ('Fallback Count', int(stats.get('fallback_reason_<lambda>', 0))),
            ('', ''),  # Blank line
            ('LATENCY STATISTICS:', ''),
            ('  Average (ms)', f"{stats['latency_ms_mean']:.1f}"),
            ('  Median (ms)', f"{stats['latency_ms_median']:.1f}"),
            ('  Std Dev (ms)', f"{stats['latency_ms_std']:.1f}"),
            ('  Min (ms)', f"{stats['latency_ms_min']:.1f}"),
            ('  Max (ms)', f"{stats['latency_ms_max']:.1f}"),
            ('', ''),  # Blank line
            ('TIMEOUT RISK:', ''),
            ('  Near Timeouts', int(stats['near_timeout_sum'])),
            ('  Near Timeout Rate', f"{stats['near_timeout_mean']:.1%}"),
            ('', ''),  # Blank line
            ('TOKEN USAGE:', ''),
            ('  Avg Prompt Tokens', f"{stats['tokens_prompt_mean']:.1f}"),
            ('  Avg Completion Tokens', f"{stats['tokens_completion_mean']:.1f}")
        ]
        
        for label, value in metrics:
            if label:
                print(f"{label:<35} {str(value):>15}")
            else:
                print()  # Blank line
    
    # Add a comparison section
    if len(model_stats) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY:")
        print("="*60)
        
        # Calculate relative performance
        backends = list(model_stats.index)
        if len(backends) == 2:
            backend1, backend2 = backends
            lat1 = model_stats.loc[backend1, 'latency_ms_mean']
            lat2 = model_stats.loc[backend2, 'latency_ms_mean']
            
            faster_backend = backend1 if lat1 < lat2 else backend2
            slower_backend = backend2 if lat1 < lat2 else backend1
            speed_ratio = max(lat1, lat2) / min(lat1, lat2)
            
            print(f"\n{faster_backend} is {speed_ratio:.1f}x faster than {slower_backend}")
            print(f"Average latencies: {faster_backend} = {min(lat1, lat2):.0f}ms, {slower_backend} = {max(lat1, lat2):.0f}ms")
    
    # JSON error analysis
    json_errors = df[df['json_ok'] == 0]
    if len(json_errors) > 0:
        print("\n\nJSON Error Breakdown:")
        error_breakdown = json_errors.groupby(['model_algo', 'fallback_reason']).size()
        print(error_breakdown.sort_values(ascending=False))

def analyze_battle_outcomes(battles, df):
    """Analyze win/loss patterns using actual backend data."""
    if battles is None:
        return
        
    print("\n" + "="*60)
    print("BATTLE OUTCOME ANALYSIS")
    print("="*60)
    
    # Each battle has 2 rows (one per player)
    total_battles = len(battles['battle_id'].unique())
    
    print(f"\nTotal Battles: {total_battles}")
    print(f"Total Rows in CSV: {len(battles)} (2 per battle)")
    
    # Now analyze by backend+algorithm combination
    print("\n" + "="*60)
    print("WIN RATE BY BACKEND+ALGORITHM")
    print("="*60)
    
    backend_algo_performance = {}
    
    # Use explicit backend+algorithm from battle summary
    for idx, row in battles.iterrows():
        if pd.notna(row['backend']) and pd.notna(row['algorithm']):
            model_algo = f"{row['backend']} ({row['algorithm']})"
            
            if model_algo not in backend_algo_performance:
                backend_algo_performance[model_algo] = {'battles': 0, 'wins': 0}
            
            backend_algo_performance[model_algo]['battles'] += 1
            if row['won'] == 1:
                backend_algo_performance[model_algo]['wins'] += 1
    
    # Print results
    for algo, stats in sorted(backend_algo_performance.items()):
        win_rate = stats['wins'] / stats['battles'] if stats['battles'] > 0 else 0
        print(f"\n{algo}:")
        print(f"  Battles played: {stats['battles']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Win Rate: {win_rate:.1%}")
        
        # Also show average latency from moves data
        algo_moves = df[df['model_algo'] == algo]
        if len(algo_moves) > 0:
            avg_latency = algo_moves['latency_ms'].mean()
            print(f"  Avg Latency: {avg_latency:.1f}ms")
    
    # Show all head-to-head matchups
    print("\n" + "="*60)
    print("HEAD-TO-HEAD MATCHUPS")
    print("="*60)
    
    # Build matchup statistics from battle data
    matchup_stats = {}
    
    for battle_id in battles['battle_id'].unique():
        battle_rows = battles[battles['battle_id'] == battle_id]
        
        if len(battle_rows) == 2:
            algos = []
            for idx, row in battle_rows.iterrows():
                if pd.notna(row['backend']) and pd.notna(row['algorithm']):
                    model_algo = f"{row['backend']} ({row['algorithm']})"
                    algos.append((model_algo, row['won']))
            
            if len(algos) == 2:
                # Sort for consistent key
                sorted_algos = sorted([algos[0][0], algos[1][0]])
                matchup_key = f"{sorted_algos[0]} vs {sorted_algos[1]}"
                
                if matchup_key not in matchup_stats:
                    matchup_stats[matchup_key] = {
                        'battles': 0,
                        'algo1_wins': 0,
                        'algo2_wins': 0,
                        'algo1': sorted_algos[0],
                        'algo2': sorted_algos[1]
                    }
                
                matchup_stats[matchup_key]['battles'] += 1
                
                # Determine winner
                for algo, won in algos:
                    if won == 1:
                        if algo == sorted_algos[0]:
                            matchup_stats[matchup_key]['algo1_wins'] += 1
                        else:
                            matchup_stats[matchup_key]['algo2_wins'] += 1
    
    # Print all matchups
    for matchup, stats in sorted(matchup_stats.items()):
        print(f"\n{matchup}:")
        print(f"  Total battles: {stats['battles']}")
        print(f"  {stats['algo1']} wins: {stats['algo1_wins']}")
        print(f"  {stats['algo2']} wins: {stats['algo2_wins']}")
        algo1_winrate = stats['algo1_wins'] / stats['battles'] * 100 if stats['battles'] > 0 else 0
        print(f"  {stats['algo1']} win rate: {algo1_winrate:.1f}%")


def analyze_move_patterns(df):
    """Analyze move and switch patterns."""
    print("\n" + "="*60)
    print("MOVE PATTERN ANALYSIS")
    print("="*60)
    
    # Action type breakdown
    action_counts = df['action_kind'].value_counts()
    print("\nAction Type Distribution:")
    for action, count in action_counts.items():
        pct = count / len(df) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Most common moves
    moves_df = df[df['action_kind'] == 'move']
    if len(moves_df) > 0:
        print("\n\nTop 15 Most Used Moves:")
        top_moves = moves_df['action_label'].value_counts().head(15)
        for move, count in top_moves.items():
            pct = count / len(moves_df) * 100
            print(f"  {move}: {count} ({pct:.1f}%)")
    
    # Switch patterns
    switches_df = df[df['action_kind'] == 'switch']
    if len(switches_df) > 0:
        print("\n\nTop 10 Switch Targets:")
        top_switches = switches_df['action_label'].value_counts().head(10)
        for mon, count in top_switches.items():
            pct = count / len(switches_df) * 100
            print(f"  {mon}: {count} ({pct:.1f}%)")

def analyze_progress_trends(df):
    """Analyze progress score trends."""
    print("\n" + "="*60)
    print("PROGRESS SCORE ANALYSIS")
    print("="*60)
    
    # Progress score statistics
    progress_stats = df['progress_score'].describe()
    print("\nProgress Score Distribution:")
    print(progress_stats)
    
    # Correlation with actions
    print("\n\nAverage Progress Score by Action Type:")
    action_progress = df.groupby('action_kind')['progress_score'].mean().sort_values(ascending=False)
    for action, score in action_progress.items():
        print(f"  {action}: {score:.1f}")

def analyze_fallback_patterns(df):
    """Analyze fallback usage patterns."""
    print("\n" + "="*60)
    print("FALLBACK ANALYSIS")
    print("="*60)
    
    # Fallback reasons
    fallbacks = df[df['fallback_reason'] != '']
    if len(fallbacks) > 0:
        print("\nFallback Reason Distribution:")
        fallback_counts = fallbacks['fallback_reason'].value_counts()
        for reason, count in fallback_counts.items():
            pct = count / len(df) * 100
            print(f"  {reason}: {count} ({pct:.1f}% of all moves)")
        
        # Fallbacks by model+algo
        print("\n\nFallbacks by Model+Algorithm:")
        fallback_by_model = fallbacks.groupby('model_algo')['fallback_reason'].count().sort_values(ascending=False)
        for model_algo, count in fallback_by_model.items():
            model_total = len(df[df['model_algo'] == model_algo])
            pct = count / model_total * 100 if model_total > 0 else 0
            print(f"  {model_algo}: {count} ({pct:.1f}% of moves)")

def main():
    """Run all analyses."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Pokemon battle logs")
    parser.add_argument("--log_dir", type=str, default="battle_log/ladder",
                       help="Directory containing metrics.csv and metrics_summary.csv")
    args = parser.parse_args()
    
    print("Pokemon Battle Log Analysis")
    print(f"Analyzing: {args.log_dir}")
    print("="*60)
    
    # Load data
    df, battles = load_data(args.log_dir)
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} move records")
    if battles is not None:
        num_battles = len(battles['battle_id'].unique())
        print(f"Loaded {num_battles} battles ({len(battles)} summary rows, 2 per battle)")
    
    # Run analyses
    analyze_model_performance(df)
    analyze_battle_outcomes(battles, df)
    analyze_move_patterns(df)
    analyze_progress_trends(df)
    analyze_fallback_patterns(df)
    
    # Summary recommendations
    print("\n" + "="*60)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    # Check for high-error model+algo combinations
    model_errors = df.groupby('model_algo')['json_ok'].mean()
    problematic_models = model_errors[model_errors < 0.95]
    if len(problematic_models) > 0:
        print("\n⚠️  Model+Algorithm combos with high JSON error rates (>5%):")
        for model_algo, success_rate in problematic_models.items():
            print(f"   - {model_algo}: {(1-success_rate)*100:.1f}% error rate")
    
    # Check for timeout issues PER MODEL+ALGO
    timeout_by_model = df.groupby('model_algo')['near_timeout'].agg(['sum', 'mean'])
    print("\n⏱️  TIMEOUT RISK ANALYSIS PER MODEL+ALGORITHM:")
    
    for model_algo in timeout_by_model.index:
        total_near_timeouts = timeout_by_model.loc[model_algo, 'sum']
        near_timeout_rate = timeout_by_model.loc[model_algo, 'mean'] * 100
        total_moves = len(df[df['model_algo'] == model_algo])
        
        # Determine risk level
        risk_level = "🟢 LOW" if near_timeout_rate < 10 else "🟡 MEDIUM" if near_timeout_rate < 30 else "🔴 HIGH"
        
        print(f"\n{model_algo}:")
        print(f"  Near-timeout rate: {near_timeout_rate:.1f}% ({total_near_timeouts}/{total_moves} moves)")
        print(f"  Risk level: {risk_level}")
        
        if near_timeout_rate > 50:
            print(f"  ⚠️  CRITICAL: This model is too slow for reliable competition use!")
        elif near_timeout_rate > 30:
            print(f"  ⚠️  WARNING: Consider reducing max_tokens or switching algorithms")
    
    # Model+Algorithm-specific recommendations
    print("\n📊 MODEL+ALGORITHM-SPECIFIC RECOMMENDATIONS:")
    
    for model_algo in df['model_algo'].unique():
        model_algo_df = df[df['model_algo'] == model_algo]
        avg_latency = model_algo_df['latency_ms'].mean()
        near_timeout_rate = model_algo_df['near_timeout'].mean() * 100
        
        print(f"\n{model_algo}:")
        
        # Token recommendations
        if near_timeout_rate > 30:
            current_tokens = model_algo_df['max_tokens'].iloc[0] if 'max_tokens' in model_algo_df else 400
            recommended_tokens = int(current_tokens * 0.75)
            print(f"  💡 Reduce max_tokens from {current_tokens} to {recommended_tokens}")
    
    # Battle length statistics
    if battles is not None and 'turns' in battles.columns:
        print("\n📊 BATTLE LENGTH STATISTICS:")
        print(f"  Average turns: {battles['turns'].mean():.1f}")
        print(f"  Min turns: {int(battles['turns'].min())}")
        print(f"  Max turns: {int(battles['turns'].max())}")
        print(f"  Median turns: {battles['turns'].median():.1f}")
        
        if battles['turns'].mean() > 50:
            print(f"\n💡 Long battles detected (avg {battles['turns'].mean():.0f} turns) - consider more aggressive strategies")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
