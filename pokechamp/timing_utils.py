"""
Timing utilities for profiling performance.
"""

import time
import functools
import json
import os
from typing import Dict, List, Optional
from collections import defaultdict


class ChromeTracer:
    """
    Lightweight tracer that writes Chrome Trace Event JSON.
    Open the resulting file in chrome://tracing or https://ui.perfetto.dev/.
    """
    def __init__(self, enabled: bool, log_dir: Optional[str], battle_tag: str, turn: int):
        self.enabled = bool(enabled)
        self.t0 = time.perf_counter()
        self.events = []
        self.stack = []  # [(name, start_ts, cat, args)]
        self.log_dir = log_dir
        self.battle_tag = battle_tag or "battle"
        self.turn = int(turn or 0)

    def _ts(self) -> float:
        return (time.perf_counter() - self.t0) * 1e6  # microseconds

    def begin(self, name: str, cat: str = "algorithm", args: Optional[dict] = None):
        if not self.enabled:
            return
        ts = self._ts()
        self.stack.append((name, ts, cat, args or {}))

    def end(self, name: Optional[str] = None, cat: str = "algorithm"):
        if not self.enabled or not self.stack:
            return
        last_name, ts_start, last_cat, last_args = self.stack.pop()
        ts_end = self._ts()
        dur = ts_end - ts_start
        # Use "X" (complete) format with explicit duration for better Perfetto compatibility
        self.events.append({
            "name": last_name,
            "cat": last_cat,
            "ph": "X",  # Complete event with duration
            "ts": ts_start,
            "dur": dur,
            "pid": 1,
            "tid": 1,
            "args": last_args
        })

    def instant(self, name: str, cat: str = "minimax", args: Optional[dict] = None):
        if not self.enabled:
            return
        self.events.append({"name": name, "cat": cat, "ph": "i", "s": "t", "ts": self._ts(), "pid": 1, "tid": 1, "args": args or {}})

    def summarize(self) -> Dict[str, float]:
        """Return duration summary by span name (milliseconds)."""
        if not self.enabled:
            return {}
        totals = {}
        # Handle both "X" (complete) and "B"/"E" (begin/end) formats
        for ev in self.events:
            if ev.get("ph") == "X":
                # Complete event with explicit duration
                name = ev.get("name", "unknown")
                dur_us = ev.get("dur", 0)
                dt_ms = float(dur_us) / 1000.0
                totals[name] = totals.get(name, 0.0) + dt_ms
        
        # Also handle B/E pairs for backward compatibility
        open_stack = []
        for ev in self.events:
            if ev.get("ph") == "B":
                open_stack.append((ev["name"], ev["ts"]))
            elif ev.get("ph") == "E" and open_stack:
                name, ts0 = open_stack.pop()
                dt_ms = max(0.0, (ev["ts"] - ts0) / 1000.0)
                totals[name] = totals.get(name, 0.0) + dt_ms
        return totals

    def dump(self, algorithm_suffix: str = "") -> Optional[str]:
        if not self.enabled or not self.log_dir:
            return None
        if not self.events:
            print(f"⚠️  [Tracer] No events to dump for turn {self.turn}")
            return None
        os.makedirs(self.log_dir, exist_ok=True)
        # Include algorithm in filename for clarity
        suffix = f"_{algorithm_suffix}" if algorithm_suffix else ""
        path = os.path.join(self.log_dir, f"trace_{self.battle_tag}_t{self.turn:03d}{suffix}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"traceEvents": self.events}, f)
            print(f"[Tracer] Wrote {len(self.events)} events to {path}")
            return path
        except Exception as e:
            print(f"⚠️  [Tracer] Failed to write trace: {e}")
            return None


class TimingStats:
    """Collect and report timing statistics."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
    
    def record(self, name: str, duration: float):
        """Record a timing."""
        self.timings[name].append(duration)
        self.call_counts[name] += 1
    
    def report(self):
        """Print timing report."""
        print("\n=== TIMING REPORT ===")
        print(f"{'Function':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        print("-" * 90)
        
        # Sort by total time
        sorted_funcs = sorted(self.timings.items(), 
                             key=lambda x: sum(x[1]), 
                             reverse=True)
        
        for func_name, times in sorted_funcs:
            total = sum(times)
            avg = total / len(times) * 1000  # Convert to ms
            min_time = min(times) * 1000
            max_time = max(times) * 1000
            
            print(f"{func_name:<40} {self.call_counts[func_name]:>8} "
                  f"{total:>10.3f} {avg:>10.1f} {min_time:>10.1f} {max_time:>10.1f}")
    
    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.call_counts.clear()

# Global timing stats
_timing_stats = TimingStats()

def timed(name=None):
    """Decorator to time function calls."""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                _timing_stats.record(func_name, duration)
        
        return wrapper
    return decorator

def report_timings():
    """Print timing report."""
    _timing_stats.report()

def reset_timings():
    """Reset timing statistics."""
    _timing_stats.reset()


class time_block:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start
        _timing_stats.record(self.name, duration)


class MinimaxTimer:
    """
    Hierarchical timer for minimax that ensures mutual exclusivity.
    
    Uses a stack-based approach where child timers automatically
    subtract from parent timers to avoid double-counting.
    """
    
    def __init__(self):
        self.categories = {
            'total': 0.0,
            'initialization': 0.0,
            'expansion_loop': 0.0,
            'llm_calls': 0.0,
            'cache_ops': 0.0,
            'batch_eval': 0.0,
            'action_selection': 0.0,
            'other': 0.0
        }
        self.stack = []  # Stack of (category, start_time, child_time)
        self.start_time = None
    
    def start(self):
        """Start the overall timer."""
        self.start_time = time.time()
        self.push('total')
    
    def push(self, category: str):
        """Start timing a category."""
        self.stack.append([category, time.time(), 0.0])
    
    def pop(self):
        """End timing the current category."""
        if not self.stack:
            return
        
        category, start, child_time = self.stack.pop()
        elapsed = time.time() - start
        
        # Subtract child time to avoid double-counting
        net_time = elapsed - child_time
        self.categories[category] += net_time
        
        # Add this elapsed time to parent's child_time
        if self.stack:
            self.stack[-1][2] += elapsed
    
    def finish(self):
        """Finish timing and close all open categories."""
        # Close all open spans
        while self.stack:
            self.pop()
        
        # Override 'total' with actual wall-clock time (not hierarchical accounting)
        if self.start_time is not None:
            self.categories['total'] = time.time() - self.start_time
        
        # Calculate 'other' time
        accounted = sum(v for k, v in self.categories.items() 
                       if k not in ('total', 'other'))
        self.categories['other'] = max(0, self.categories['total'] - accounted)
    
    def report(self):
        """Print a formatted timing report with visualization."""
        total = self.categories['total']
        if total == 0:
            print("No timing data collected")
            return
        
        print(f"\n⏱️  Minimax Timing Breakdown (Total: {total:.2f}s)")
        print("=" * 60)
        
        # Sort by time descending, but keep total first
        items = [(k, v) for k, v in self.categories.items() 
                if k != 'total' and v > 0.001]  # Filter negligible times
        items.sort(key=lambda x: x[1], reverse=True)
        
        for name, duration in items:
            pct = (duration / total) * 100
            bar_width = min(int(pct / 2), 50)  # Scale to 50 chars max
            bar = '█' * bar_width
            
            print(f"  {name:<20} {duration:>6.2f}s {pct:>5.1f}% {bar}")
        
        print("=" * 60)
        
        # Sanity check
        accounted = sum(v for k, v in self.categories.items() if k != 'total')
        if abs(accounted - total) > 0.01:
            print(f"⚠️  Warning: Timing mismatch ({accounted:.2f}s vs {total:.2f}s)")
    
    def get_dict(self):
        """Get timing data as a dictionary."""
        return dict(self.categories)
