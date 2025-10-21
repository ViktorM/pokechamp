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
    
    # Check if algorithm column exists (for backward compatibility)
    if 'algorithm' not in df.columns:
        # If no algorithm column, assume 'io' for all entries
        df['algorithm'] = 'io'
        # Reorder columns to put algorithm after backend
        cols = df.columns.tolist()
        backend_idx = cols.index('backend')
        cols.insert(backend_idx + 1, cols.pop(cols.index('algorithm')))
        df = df[cols]
    
    # Create combined model+algo identifier
    df['model_algo'] = df['backend'] + ' (' + df['algorithm'] + ')'
    
    # Load battle summaries if exists
    battles = None
    if os.path.exists(summary_path):
        battles = pd.read_csv(summary_path, header=None)
        battles.columns = ['battle_id', 'won', 'turns', 'avg_latency', 'total_moves', 
                          'json_errors', 'switches', 'timeouts', 'final_score', 'extra']
    
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
    # Group by battle_id to get unique battles
    unique_battles = battles.groupby('battle_id').agg({
        'won': 'sum',  # Should always be 1 (one winner per battle)
        'turns': 'mean',
        'avg_latency': 'mean',
        'total_moves': 'sum',
        'json_errors': 'sum',
        'switches': 'sum',
        'timeouts': 'sum',
        'final_score': 'mean'
    }).reset_index()
    
    total_battles = len(unique_battles)
    
    print(f"\nTotal Battles: {total_battles}")
    print(f"Total Rows in CSV: {len(battles)} (2 per battle)")
    
    # Now analyze by backend
    print("\n" + "="*60)
    print("WIN RATE BY BACKEND")
    print("="*60)
    
    # Create a mapping of battle_tag to model_algo from metrics
    battle_to_model = df.groupby('battle_tag')['model_algo'].first().to_dict()
    
    # Add model_algo to battles dataframe
    battles['model_algo'] = battles['battle_id'].map(battle_to_model)
    
    # Group by actual model+algo
    model_stats = []
    for model_algo in sorted(battles['model_algo'].unique()):
        if pd.isna(model_algo):
            continue
            
        model_battles = battles[battles['model_algo'] == model_algo]
        if len(model_battles) > 0:
            # Count UNIQUE battles (each battle has 2 rows)
            unique_model_battles = model_battles.groupby('battle_id').first()
            total = len(unique_model_battles)
            wins = unique_model_battles['won'].sum()
            win_rate = wins / total if total > 0 else 0
            avg_latency = model_battles['avg_latency'].mean()  # Use all rows for latency avg
            
            model_stats.append({
                'model_algo': model_algo,
                'battles': total,
                'wins': wins,
                'win_rate': win_rate,
                'avg_latency': avg_latency
            })
            
            print(f"\n{model_algo}:")
            print(f"  Battles: {total}")
            print(f"  Wins: {wins}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Avg Latency: {avg_latency:.1f}ms")
    
    # Analyze statistics PER MODEL
    print("\n\nPER-MODEL DETAILED STATISTICS:")
    print("="*60)
    
    for model_algo in sorted(battles['model_algo'].unique()):
        if pd.isna(model_algo):
            continue
            
        group_battles = battles[battles['model_algo'] == model_algo]
        if len(group_battles) > 0:
            # Count UNIQUE battles (each battle has 2 rows)
            unique_group_battles = group_battles.groupby('battle_id').first()
            
            print(f"\n{model_algo}:")
            print(f"{'Metric':<25} {'Value':>15}")
            print("-" * 42)
            
            metrics = [
                ('Total Battles', len(unique_group_battles)),
                ('Wins', unique_group_battles['won'].sum()),
                ('Win Rate', f"{unique_group_battles['won'].mean():.1%}"),
                ('Avg Latency (ms)', f"{group_battles['avg_latency'].mean():.1f}"),
                ('Avg Turns per Game', f"{group_battles['turns'].mean():.1f}"),
                ('Avg Switches per Game', f"{group_battles['switches'].mean():.1f}"),
                ('Total JSON Errors', group_battles['json_errors'].sum()),
                ('Total Timeouts', group_battles['timeouts'].sum()),
                ('Games with Timeouts', len(unique_group_battles[unique_group_battles['timeouts'] > 0])),
                ('Min Latency (ms)', f"{group_battles['avg_latency'].min():.1f}"),
                ('Max Latency (ms)', f"{group_battles['avg_latency'].max():.1f}"),
                ('Latency Std Dev', f"{group_battles['avg_latency'].std():.1f}")
            ]
            
            for label, value in metrics:
                if isinstance(value, (int, float)):
                    print(f"{label:<25} {value:>15}")
                else:
                    print(f"{label:<25} {value:>15}")
            
            # Win/Loss breakdown for this model
            wins = group_battles[group_battles['won'] == 1]
            losses = group_battles[group_battles['won'] == 0]
            
            if len(wins) > 0 and len(losses) > 0:
                print(f"\n  Win vs Loss Comparison for {model_algo}:")
                print(f"  {'Metric':<20} {'When Win':>12} {'When Lose':>12}")
                print("  " + "-" * 46)
                
                comparison_metrics = [
                    ('Avg Turns', 'turns'),
                    ('Avg Latency (ms)', 'avg_latency'),
                    ('Switches', 'switches')
                ]
                
                for label, col in comparison_metrics:
                    win_val = wins[col].mean() if len(wins) > 0 else 0
                    loss_val = losses[col].mean() if len(losses) > 0 else 0
                    print(f"  {label:<20} {win_val:>12.1f} {loss_val:>12.1f}")
    
    # Timeout analysis
    timeout_battles = battles[battles['timeouts'] > 0]
    if len(timeout_battles) > 0:
        num_timeout_battles = len(timeout_battles.groupby('battle_id'))
        print(f"\n\nTimeout Analysis:")
        print(f"Battles with timeouts: {num_timeout_battles} ({num_timeout_battles/total_battles*100:.1f}%)")
        print(f"Win rate when you timeout: {timeout_battles['won'].mean():.1%}")

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
        print(f"Loaded {len(battles)} battle summaries")
    
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
    
    # Battle length insight
    if battles is not None:
        avg_turns = battles['turns'].mean()
        if avg_turns > 50:
            print(f"\n💡 Long battles detected (avg {avg_turns:.0f} turns) - consider more aggressive strategies")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
