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
    
    # Load battle summaries if exists
    battles = None
    if os.path.exists(summary_path):
        battles = pd.read_csv(summary_path, header=None)
        battles.columns = ['battle_id', 'won', 'turns', 'avg_latency', 'total_moves', 
                          'json_errors', 'switches', 'timeouts', 'final_score', 'extra']
    
    return df, battles

def analyze_model_performance(df):
    """Analyze performance by LLM backend."""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Group by backend
    model_stats = df.groupby('backend').agg({
        'json_ok': ['count', 'mean'],
        'latency_ms': ['mean', 'std', 'max'],
        'near_timeout': 'sum',
        'tokens_prompt': 'mean',
        'tokens_completion': 'mean'
    }).round(2)
    
    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    model_stats.columns = ['total_moves', 'json_success_rate', 'avg_latency_ms', 
                          'latency_std', 'max_latency_ms', 'near_timeouts',
                          'avg_prompt_tokens', 'avg_completion_tokens']
    
    # Sort by usage
    model_stats = model_stats.sort_values('total_moves', ascending=False)
    
    print("\nModel Usage and Performance:")
    print(model_stats)
    
    # JSON error analysis
    json_errors = df[df['json_ok'] == 0]
    if len(json_errors) > 0:
        print("\n\nJSON Error Breakdown:")
        error_breakdown = json_errors.groupby(['backend', 'fallback_reason']).size()
        print(error_breakdown.sort_values(ascending=False))

def analyze_battle_outcomes(battles):
    """Analyze win/loss patterns."""
    if battles is None:
        return
        
    print("\n" + "="*60)
    print("BATTLE OUTCOME ANALYSIS")
    print("="*60)
    
    total_battles = len(battles)
    wins = battles['won'].sum()
    win_rate = wins / total_battles if total_battles > 0 else 0
    
    print(f"\nTotal Battles: {total_battles}")
    print(f"Wins: {wins}")
    print(f"Losses: {total_battles - wins}")
    print(f"Win Rate: {win_rate:.1%}")
    
    # Separate wins and losses
    wins_df = battles[battles['won'] == 1]
    losses_df = battles[battles['won'] == 0]
    
    print("\n\nWin vs Loss Statistics:")
    print(f"{'Metric':<20} {'Wins':>15} {'Losses':>15}")
    print("-" * 52)
    
    metrics = [
        ('Avg Turns', 'turns'),
        ('Avg Latency (ms)', 'avg_latency'),
        ('JSON Errors', 'json_errors'),
        ('Switches', 'switches'),
        ('Timeouts', 'timeouts')
    ]
    
    for label, col in metrics:
        win_avg = wins_df[col].mean() if len(wins_df) > 0 else 0
        loss_avg = losses_df[col].mean() if len(losses_df) > 0 else 0
        print(f"{label:<20} {win_avg:>15.1f} {loss_avg:>15.1f}")
    
    # Timeout analysis
    timeout_battles = battles[battles['timeouts'] > 0]
    if len(timeout_battles) > 0:
        print(f"\n\nTimeout Analysis:")
        print(f"Battles with timeouts: {len(timeout_battles)} ({len(timeout_battles)/total_battles*100:.1f}%)")
        print(f"Win rate in timeout battles: {timeout_battles['won'].mean():.1%}")

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
        
        # Fallbacks by backend
        print("\n\nFallbacks by Model:")
        fallback_by_model = fallbacks.groupby('backend')['fallback_reason'].count().sort_values(ascending=False)
        for model, count in fallback_by_model.items():
            model_total = len(df[df['backend'] == model])
            pct = count / model_total * 100 if model_total > 0 else 0
            print(f"  {model}: {count} ({pct:.1f}% of model's moves)")

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
    analyze_battle_outcomes(battles)
    analyze_move_patterns(df)
    analyze_progress_trends(df)
    analyze_fallback_patterns(df)
    
    # Summary recommendations
    print("\n" + "="*60)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    # Check for high-error models
    model_errors = df.groupby('backend')['json_ok'].mean()
    problematic_models = model_errors[model_errors < 0.95]
    if len(problematic_models) > 0:
        print("\n⚠️  Models with high JSON error rates (>5%):")
        for model, success_rate in problematic_models.items():
            print(f"   - {model}: {(1-success_rate)*100:.1f}% error rate")
    
    # Check for timeout issues
    timeout_pct = df['near_timeout'].sum() / len(df) * 100
    if timeout_pct > 5:
        print(f"\n⚠️  High near-timeout rate: {timeout_pct:.1f}% of moves")
    
    # Battle length insight
    if battles is not None:
        avg_turns = battles['turns'].mean()
        if avg_turns > 50:
            print(f"\n💡 Long battles detected (avg {avg_turns:.0f} turns) - consider more aggressive strategies")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
