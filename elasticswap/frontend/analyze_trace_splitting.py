#!/usr/bin/env python3
"""
Analyze the effect of splitting traces at different gap thresholds.
Show how many traces would result and their duration distributions.
"""

from dateutil import parser as date_parser
import json
import glob
from collections import defaultdict
import numpy as np

def split_trace_at_gaps(session_events, gap_threshold_seconds):
    """Split a session into multiple sub-traces at large gaps between LLM calls"""
    sub_traces = []
    current_trace = []
    last_llm_time = None
    
    for event in session_events:
        if event['event_type'] == 'llm_call':
            msg_id = event['message_id']
            current_time = date_parser.parse(event['timestamp']).timestamp()
            
            # Check if this is a duplicate (skip in gap calculation)
            if last_llm_time is not None:
                gap = current_time - last_llm_time
                
                if gap > gap_threshold_seconds:
                    # Large gap detected - save current trace and start new one
                    if current_trace:
                        sub_traces.append(current_trace)
                    current_trace = []
            
            last_llm_time = current_time
        
        current_trace.append(event)
    
    # Don't forget the last trace
    if current_trace:
        sub_traces.append(current_trace)
    
    return sub_traces


def analyze_splitting(claude_dataset_dir="/vllm/vllm/elasticswap/toolcall_dataset_claude",
                     gap_thresholds=[300, 600, 900, 1800]):
    """Analyze trace splitting at different gap thresholds"""
    
    json_files = glob.glob(f"{claude_dataset_dir}/*_events.json")
    
    print(f"Analyzing splitting strategies on {len(json_files)} trace files...\n")
    
    # Collect original traces
    original_traces = []
    for json_file in json_files:
        with open(json_file) as f:
            events = json.load(f)
        events.sort(key=lambda e: e['timestamp'])
        
        sessions = defaultdict(list)
        for event in events:
            sessions[event['session_id']].append(event)
        
        for session_events in sessions.values():
            if session_events:
                original_traces.append(session_events)
    
    print("="*70)
    print("ORIGINAL TRACES (No splitting)")
    print("="*70)
    
    original_durations = []
    for trace in original_traces:
        timestamps = [date_parser.parse(e['timestamp']).timestamp() for e in trace]
        duration = (timestamps[-1] - timestamps[0]) / 60  # minutes
        original_durations.append(duration)
    
    print(f"Total traces: {len(original_durations)}")
    print(f"Duration stats (minutes):")
    print(f"  Median: {np.median(original_durations):.2f}")
    print(f"  Mean:   {np.mean(original_durations):.2f}")
    print(f"  P90:    {np.percentile(original_durations, 90):.2f}")
    print(f"  P95:    {np.percentile(original_durations, 95):.2f}")
    print(f"Over 20 min: {sum(1 for d in original_durations if d > 20)} ({sum(1 for d in original_durations if d > 20)/len(original_durations)*100:.1f}%)")
    
    # Try different splitting thresholds
    for threshold in gap_thresholds:
        print(f"\n{'='*70}")
        print(f"SPLITTING AT {threshold}s GAPS ({threshold/60:.0f} minutes)")
        print("="*70)
        
        all_sub_traces = []
        num_splits = 0
        
        for trace in original_traces:
            sub_traces = split_trace_at_gaps(trace, threshold)
            if len(sub_traces) > 1:
                num_splits += 1
            all_sub_traces.extend(sub_traces)
        
        # Calculate durations of sub-traces
        sub_durations = []
        for sub_trace in all_sub_traces:
            timestamps = [date_parser.parse(e['timestamp']).timestamp() for e in sub_trace]
            duration = (timestamps[-1] - timestamps[0]) / 60  # minutes
            sub_durations.append(duration)
        
        print(f"Total traces after splitting: {len(sub_durations)} "
              f"(+{len(sub_durations) - len(original_durations)} new traces)")
        print(f"Original traces that were split: {num_splits} ({num_splits/len(original_traces)*100:.1f}%)")
        print(f"\nDuration stats (minutes):")
        print(f"  Median: {np.median(sub_durations):.2f}")
        print(f"  Mean:   {np.mean(sub_durations):.2f}")
        print(f"  P90:    {np.percentile(sub_durations, 90):.2f}")
        print(f"  P95:    {np.percentile(sub_durations, 95):.2f}")
        print(f"  Max:    {max(sub_durations):.2f}")
        over_20 = sum(1 for d in sub_durations if d > 20)
        print(f"Over 20 min: {over_20} ({over_20/len(sub_durations)*100:.1f}%)")
        
        # Show trace length distribution
        under_5 = sum(1 for d in sub_durations if d <= 5)
        under_10 = sum(1 for d in sub_durations if d <= 10)
        under_15 = sum(1 for d in sub_durations if d <= 15)
        under_20 = sum(1 for d in sub_durations if d <= 20)
        
        print(f"\nDuration distribution:")
        print(f"  0-5 min:   {under_5:3d} ({under_5/len(sub_durations)*100:.1f}%)")
        print(f"  5-10 min:  {under_10-under_5:3d} ({(under_10-under_5)/len(sub_durations)*100:.1f}%)")
        print(f"  10-15 min: {under_15-under_10:3d} ({(under_15-under_10)/len(sub_durations)*100:.1f}%)")
        print(f"  15-20 min: {under_20-under_15:3d} ({(under_20-under_15)/len(sub_durations)*100:.1f}%)")
        print(f"  >20 min:   {len(sub_durations)-under_20:3d} ({(len(sub_durations)-under_20)/len(sub_durations)*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print("="*70)
    print("Based on the analysis above, choose a gap threshold that:")
    print("  1. Minimizes traces over 20 minutes")
    print("  2. Doesn't create too many very short traces (< 5 min)")
    print("  3. Preserves natural workflow patterns")
    print("\nSuggested threshold: 300s (5 minutes)")
    print("  - Splits at clear human pause boundaries")
    print("  - Keeps natural thinking/tool execution time")

if __name__ == "__main__":
    analyze_splitting()


