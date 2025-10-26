#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate profiling logs (metrics + timing events) and render summary tables and charts.
General across algorithms (io, io2, ranker, sc, tot, minimax), *not* minimax-specific.

Usage (CLI):
  python analyze_profiles.py --log-dirs /path/to/runA /path/to/runB --out-dir /path/to/report

Programmatic:
  from analyze_profiles import generate_profile_report
  generate_profile_report(log_dirs=['./logs/runA'], out_dir='./logs/runA/report')

Expected files in each log_dir (any subset is fine):
  - metrics.csv                      (per-move KPIs written by LLMPlayer)
  - metrics_summary.csv              (per-battle KPIs written by LLMPlayer)
  - minimax_profile.csv              (minimax timing CSV)
  - *trace*.json                     (Chrome trace JSON with traceEvents)

Outputs (written to out_dir):
  - aggregate_moves.csv              (per-move table joining metrics + timing)
  - aggregate_battles.csv            (per-battle summary across algorithms)
  - aggregate_algorithms.csv         (per-algorithm summary KPIs)
  - timeline_top_turn_<k>.png        (Gantt-like timeline per slowest turns)
  - stacked_phase_means.png          (stacked bar: mean phase durations per algorithm)
  - profile_trace.json               (Chrome trace JSON constructed from timing CSVs)

Charts: matplotlib only, 1 chart per figure, no explicit colors/styles set.
"""
import os
import glob
import json
import argparse
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# Optional matplotlib for charts
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _read_csv_safe(path: str, **kwargs) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def _load_metrics_csv(log_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(log_dir, "metrics.csv")
    df = _read_csv_safe(path)
    if df is None:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    df["source_dir"] = os.path.abspath(log_dir)
    return df


def _load_metrics_summary_csv(log_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(log_dir, "metrics_summary.csv")
    df = _read_csv_safe(path)
    if df is None:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    df["source_dir"] = os.path.abspath(log_dir)
    return df


def _load_minimax_profile_csv(log_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(log_dir, "minimax_profile.csv")
    df = _read_csv_safe(path)
    if df is None:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    df["source_dir"] = os.path.abspath(log_dir)
    return df


def _collect_trace_jsons(log_dir: str):
    return sorted(glob.glob(os.path.join(log_dir, "*trace*.json")))


def _parse_trace_json(path: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    events = data.get("traceEvents") or data.get("events") or []
    if not events:
        return None
    rows = []
    for e in events:
        if e.get("ph") not in ["B", "E", "X"]:
            continue
        name = e.get("name", "")
        ts_us = e.get("ts")
        dur_us = e.get("dur")
        args = e.get("args", {}) or {}
        rows.append({
            "battle_tag": args.get("battle_tag", ""),
            "turn": args.get("turn", np.nan),
            "algorithm": args.get("algorithm", "minimax"),
            "backend": args.get("backend", ""),
            "phase": name,
            "start_ms": (float(ts_us) / 1000.0) if isinstance(ts_us, (int, float)) else np.nan,
            "duration_ms": (float(dur_us) / 1000.0) if isinstance(dur_us, (int, float)) else np.nan,
            "source_file": os.path.abspath(path)
        })
    df = pd.DataFrame(rows)
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="coerce")
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
    return df


def _load_all_events(log_dirs: List[str]) -> pd.DataFrame:
    frames = []
    for d in log_dirs:
        for p in _collect_trace_jsons(d):
            df = _parse_trace_json(p)
            if df is not None and not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["battle_tag", "turn", "algorithm", "backend", "phase", "start_ms", "duration_ms", "source_file"])
    events = pd.concat(frames, ignore_index=True)
    events["phase"] = events["phase"].fillna("").astype(str)
    events = events.dropna(subset=["duration_ms"])
    return events


def _load_all_metrics(log_dirs: List[str]):
    frames_moves = []
    frames_battles = []
    frames_profile = []
    for d in log_dirs:
        m = _load_metrics_csv(d)
        if m is not None:
            frames_moves.append(m)
        s = _load_metrics_summary_csv(d)
        if s is not None:
            frames_battles.append(s)
        p = _load_minimax_profile_csv(d)
        if p is not None:
            frames_profile.append(p)
    df_moves = pd.concat(frames_moves, ignore_index=True) if frames_moves else pd.DataFrame()
    df_battles = pd.concat(frames_battles, ignore_index=True) if frames_battles else pd.DataFrame()
    df_profile = pd.concat(frames_profile, ignore_index=True) if frames_profile else pd.DataFrame()
    return df_moves, df_battles, df_profile


def _aggregate_algorithms(df_profile: pd.DataFrame, df_battles: pd.DataFrame) -> pd.DataFrame:
    if df_profile.empty and df_battles.empty:
        return pd.DataFrame()

    result = pd.DataFrame()

    if not df_profile.empty:
        result = df_profile.groupby(["battle_tag"], dropna=False).agg({
            "elapsed_s": "mean",
            "tt_hits": "sum",
            "tt_misses": "sum",
            "sim_steps": "sum",
            "sim_step_ms": "sum",
        }).reset_index()

    if not df_battles.empty and "won" in df_battles.columns:
        wins = df_battles.groupby(["algorithm", "backend"], dropna=False).agg({
            "won": lambda s: float(np.mean(pd.to_numeric(s, errors="coerce").fillna(0)))
        }).reset_index()
        wins = wins.rename(columns={"won": "win_rate"})
        if not result.empty and "algorithm" in result.columns:
            result = pd.merge(result, wins, how="outer", on=["algorithm", "backend"])

    return result


def _plot_stacked_phase_means(df_profile: pd.DataFrame, out_path: str):
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping chart")
        return
    if df_profile.empty:
        return

    # Calculate mean for each phase across all turns
    phase_cols = ["init_ms", "expansion_ms", "cache_ops_ms", "llm_calls_ms", "batch_eval_ms", "action_selection_ms", "other_ms"]
    available_cols = [c for c in phase_cols if c in df_profile.columns]
    if not available_cols:
        return

    means = df_profile[available_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    means.plot(kind="bar", ax=ax)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mean duration (ms)")
    ax.set_title("Mean phase durations in minimax")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved stacked phase chart to {out_path}")


def _select_top_slowest_turns(df_profile: pd.DataFrame, k: int = 3):
    if df_profile.empty or "elapsed_s" not in df_profile.columns:
        return []
    tmp = df_profile.dropna(subset=["elapsed_s"]).copy()
    tmp = tmp.sort_values(by="elapsed_s", ascending=False)
    rows = tmp.head(k)[["battle_tag", "turn"]].dropna().values.tolist()
    return [(str(a), int(b)) for a, b in rows]


def generate_profile_report(log_dirs: List[str], out_dir: str, top_k_timeline: int = 3) -> Dict[str, str]:
    """Generate profiling report from log directories."""
    os.makedirs(out_dir, exist_ok=True)
    df_moves, df_bsum, df_profile = _load_all_metrics(log_dirs)
    df_events = _load_all_events(log_dirs)

    out_paths = {}

    # Save aggregated CSVs
    if not df_profile.empty:
        p_profile = os.path.join(out_dir, "aggregate_minimax_profile.csv")
        df_profile.to_csv(p_profile, index=False)
        out_paths["aggregate_profile"] = p_profile

    if not df_moves.empty:
        p_moves = os.path.join(out_dir, "aggregate_moves.csv")
        df_moves.to_csv(p_moves, index=False)
        out_paths["aggregate_moves"] = p_moves

    if not df_bsum.empty:
        p_battles = os.path.join(out_dir, "aggregate_battles.csv")
        df_bsum.to_csv(p_battles, index=False)
        out_paths["aggregate_battles"] = p_battles

    # Generate charts if matplotlib is available
    if HAS_MATPLOTLIB and not df_profile.empty:
        p_stack = os.path.join(out_dir, "stacked_phase_means.png")
        _plot_stacked_phase_means(df_profile, p_stack)
        out_paths["stacked_phase_means"] = p_stack

    # Merge all trace JSONs into one
    if not df_events.empty:
        p_trace = os.path.join(out_dir, "profile_trace.json")
        trace_data = {"traceEvents": []}
        for d in log_dirs:
            for trace_path in _collect_trace_jsons(d):
                try:
                    with open(trace_path, "r") as f:
                        data = json.load(f)
                        trace_data["traceEvents"].extend(data.get("traceEvents", []))
                except Exception:
                    pass
        with open(p_trace, "w") as f:
            json.dump(trace_data, f)
        out_paths["profile_trace"] = p_trace
        print(f"Merged Chrome trace saved to {p_trace}")
        print(f"Open in: chrome://tracing or https://ui.perfetto.dev/")
    
    return out_paths


def main():
    ap = argparse.ArgumentParser(description="Aggregate and visualize profiling logs (general across algorithms).")
    ap.add_argument("--log-dirs", nargs="+", required=True, help="Directories with metrics.csv / minimax_profile.csv / *trace*.json")
    ap.add_argument("--out-dir", required=True, help="Directory to write aggregated tables and charts")
    ap.add_argument("--top-k-timeline", type=int, default=5, help="Number of slowest moves to render as timeline charts")
    args = ap.parse_args()
    out = generate_profile_report(args.log_dirs, args.out_dir, top_k_timeline=args.top_k_timeline)
    print("\n✅ Profiling Report Generated:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

