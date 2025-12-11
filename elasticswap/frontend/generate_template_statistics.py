#!/usr/bin/env python3
"""
Generate comprehensive statistics for trace templates with t=300s threshold.
Includes end-to-end timestamp differences, number of turns, tools, tokens, etc.
"""

import os
import sys
import json
import numpy as np
from collections import Counter
from dateutil import parser as date_parser

# Add the frontend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_trace_templates_humanpause import (
    extract_trace_templates_with_evolution,
    TraceTemplate
)


def calculate_template_duration(template: TraceTemplate) -> float:
    """
    Calculate end-to-end duration of a template in seconds.
    
    Returns the time difference between the first event and the last event.
    """
    all_timestamps = []
    
    for turn in template.turns:
        for tool_exec in turn.tool_executions:
            try:
                # Parse timestamp and add start time
                start_epoch = date_parser.parse(tool_exec.start_timestamp).timestamp()
                all_timestamps.append(start_epoch)
                
                # Add end time (start + duration)
                end_epoch = start_epoch + tool_exec.duration
                all_timestamps.append(end_epoch)
            except (ValueError, TypeError) as e:
                # Skip invalid timestamps
                continue
    
    if len(all_timestamps) < 2:
        return 0.0
    
    # Return duration from first to last event
    return max(all_timestamps) - min(all_timestamps)


def generate_statistics(templates: list, threshold_seconds: float = 300.0):
    """Generate comprehensive statistics for templates."""
    if not templates:
        print("No templates found!")
        return
    
    print("\n" + "="*80)
    print(f"TEMPLATE STATISTICS (t={threshold_seconds}s threshold)")
    print("="*80)
    
    # Calculate metrics for each template
    durations = []
    num_turns_list = []
    num_tools_list = []
    total_tokens_list = []
    context_tokens_list = []
    output_tokens_list = []
    
    for template in templates:
        # End-to-end duration
        duration = calculate_template_duration(template)
        durations.append(duration)
        
        # Number of turns
        num_turns_list.append(template.num_turns)
        
        # Number of tools
        num_tools_list.append(template.total_tools)
        
        # Total input tokens (accumulated context at the end)
        context_tokens_list.append(template.total_context_accumulated)
        
        # Total output tokens
        total_output = sum(turn.output for turn in template.turns)
        output_tokens_list.append(total_output)
        
        # Total tokens (input + output)
        total_tokens_list.append(template.total_context_accumulated + total_output)
    
    # Basic counts
    print(f"\n📊 BASIC COUNTS")
    print(f"  Total templates: {len(templates)}")
    
    # End-to-end duration statistics
    print(f"\n⏱️  END-TO-END DURATION (seconds)")
    print(f"  Min: {min(durations):.2f}s ({min(durations)/60:.2f} minutes)")
    print(f"  Max: {max(durations):.2f}s ({max(durations)/60:.2f} minutes)")
    print(f"  Mean: {np.mean(durations):.2f}s ({np.mean(durations)/60:.2f} minutes)")
    print(f"  Median: {np.median(durations):.2f}s ({np.median(durations)/60:.2f} minutes)")
    print(f"  Std: {np.std(durations):.2f}s ({np.std(durations)/60:.2f} minutes)")
    print(f"  P25: {np.percentile(durations, 25):.2f}s ({np.percentile(durations, 25)/60:.2f} minutes)")
    print(f"  P75: {np.percentile(durations, 75):.2f}s ({np.percentile(durations, 75)/60:.2f} minutes)")
    print(f"  P90: {np.percentile(durations, 90):.2f}s ({np.percentile(durations, 90)/60:.2f} minutes)")
    print(f"  P95: {np.percentile(durations, 95):.2f}s ({np.percentile(durations, 95)/60:.2f} minutes)")
    print(f"  P99: {np.percentile(durations, 99):.2f}s ({np.percentile(durations, 99)/60:.2f} minutes)")
    
    # Duration distribution
    duration_dist = Counter()
    for d in durations:
        if d < 60:
            duration_dist["<1min"] += 1
        elif d < 300:
            duration_dist["1-5min"] += 1
        elif d < 600:
            duration_dist["5-10min"] += 1
        elif d < 1800:
            duration_dist["10-30min"] += 1
        elif d < 3600:
            duration_dist["30-60min"] += 1
        else:
            duration_dist[">60min"] += 1
    
    print(f"\n  Duration distribution:")
    for bucket, count in sorted(duration_dist.items()):
        pct = (count / len(templates)) * 100
        print(f"    {bucket}: {count} templates ({pct:.1f}%)")
    
    # Number of turns
    print(f"\n🔄 NUMBER OF TURNS")
    print(f"  Min: {min(num_turns_list)}")
    print(f"  Max: {max(num_turns_list)}")
    print(f"  Mean: {np.mean(num_turns_list):.2f}")
    print(f"  Median: {np.median(num_turns_list):.2f}")
    print(f"  Std: {np.std(num_turns_list):.2f}")
    
    # Number of tools
    print(f"\n🔧 NUMBER OF TOOLS")
    print(f"  Min: {min(num_tools_list)}")
    print(f"  Max: {max(num_tools_list)}")
    print(f"  Mean: {np.mean(num_tools_list):.2f}")
    print(f"  Median: {np.median(num_tools_list):.2f}")
    print(f"  Std: {np.std(num_tools_list):.2f}")
    
    # Total tokens
    print(f"\n📝 TOTAL TOKENS (input + output)")
    print(f"  Min: {min(total_tokens_list):,}")
    print(f"  Max: {max(total_tokens_list):,}")
    print(f"  Mean: {np.mean(total_tokens_list):,.0f}")
    print(f"  Median: {np.median(total_tokens_list):,.0f}")
    print(f"  Std: {np.std(total_tokens_list):,.0f}")
    
    # Context tokens (input)
    print(f"\n📖 CONTEXT TOKENS (input, accumulated)")
    print(f"  Min: {min(context_tokens_list):,}")
    print(f"  Max: {max(context_tokens_list):,}")
    print(f"  Mean: {np.mean(context_tokens_list):,.0f}")
    print(f"  Median: {np.median(context_tokens_list):,.0f}")
    print(f"  Std: {np.std(context_tokens_list):,.0f}")
    
    # Output tokens
    print(f"\n✍️  OUTPUT TOKENS")
    print(f"  Min: {min(output_tokens_list):,}")
    print(f"  Max: {max(output_tokens_list):,}")
    print(f"  Mean: {np.mean(output_tokens_list):,.0f}")
    print(f"  Median: {np.median(output_tokens_list):,.0f}")
    print(f"  Std: {np.std(output_tokens_list):,.0f}")
    
    # Tools per turn
    all_tools_per_turn = []
    for template in templates:
        for turn in template.turns:
            all_tools_per_turn.append(turn.num_tools)
    
    if all_tools_per_turn:
        print(f"\n🔧 TOOLS PER TURN (across all templates)")
        print(f"  Min: {min(all_tools_per_turn)}")
        print(f"  Max: {max(all_tools_per_turn)}")
        print(f"  Mean: {np.mean(all_tools_per_turn):.2f}")
        print(f"  Median: {np.median(all_tools_per_turn):.2f}")
    
    # Duration per turn
    duration_per_turn = [d / n if n > 0 else 0 for d, n in zip(durations, num_turns_list)]
    print(f"\n⏱️  DURATION PER TURN (seconds)")
    print(f"  Min: {min(duration_per_turn):.2f}s")
    print(f"  Max: {max(duration_per_turn):.2f}s")
    print(f"  Mean: {np.mean(duration_per_turn):.2f}s")
    print(f"  Median: {np.median(duration_per_turn):.2f}s")
    
    # Tokens per turn
    tokens_per_turn = [t / n if n > 0 else 0 for t, n in zip(total_tokens_list, num_turns_list)]
    print(f"\n📝 TOKENS PER TURN")
    print(f"  Min: {min(tokens_per_turn):.0f}")
    print(f"  Max: {max(tokens_per_turn):.0f}")
    print(f"  Mean: {np.mean(tokens_per_turn):.0f}")
    print(f"  Median: {np.median(tokens_per_turn):.0f}")
    
    # Create summary dictionary
    summary = {
        "threshold_seconds": threshold_seconds,
        "total_templates": len(templates),
        "duration": {
            "min": float(min(durations)),
            "max": float(max(durations)),
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "std": float(np.std(durations)),
            "p25": float(np.percentile(durations, 25)),
            "p75": float(np.percentile(durations, 75)),
            "p90": float(np.percentile(durations, 90)),
            "p95": float(np.percentile(durations, 95)),
            "p99": float(np.percentile(durations, 99)),
        },
        "turns": {
            "min": int(min(num_turns_list)),
            "max": int(max(num_turns_list)),
            "mean": float(np.mean(num_turns_list)),
            "median": float(np.median(num_turns_list)),
            "std": float(np.std(num_turns_list)),
        },
        "tools": {
            "min": int(min(num_tools_list)),
            "max": int(max(num_tools_list)),
            "mean": float(np.mean(num_tools_list)),
            "median": float(np.median(num_tools_list)),
            "std": float(np.std(num_tools_list)),
        },
        "total_tokens": {
            "min": int(min(total_tokens_list)),
            "max": int(max(total_tokens_list)),
            "mean": float(np.mean(total_tokens_list)),
            "median": float(np.median(total_tokens_list)),
            "std": float(np.std(total_tokens_list)),
        },
        "context_tokens": {
            "min": int(min(context_tokens_list)),
            "max": int(max(context_tokens_list)),
            "mean": float(np.mean(context_tokens_list)),
            "median": float(np.median(context_tokens_list)),
            "std": float(np.std(context_tokens_list)),
        },
        "output_tokens": {
            "min": int(min(output_tokens_list)),
            "max": int(max(output_tokens_list)),
            "mean": float(np.mean(output_tokens_list)),
            "median": float(np.median(output_tokens_list)),
            "std": float(np.std(output_tokens_list)),
        },
    }
    
    return summary


def main():
    # Dataset path
    dataset_dir = "/vllm/vllm/elasticswap/toolcall_dataset_claude_annotated_60s_sessionwise_v3_clustered/toolcall_dataset_claude_annotated_60s_sessionwise_v3_bfcl_benchmark"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please update the dataset_dir path in the script.")
        return
    
    threshold_seconds = 300.0
    
    print("="*80)
    print(f"GENERATING TEMPLATE STATISTICS (t={threshold_seconds}s)")
    print("="*80)
    print(f"Dataset: {dataset_dir}")
    print(f"Threshold: {threshold_seconds} seconds")
    
    # Extract templates with t=300s
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir=dataset_dir,
        window_duration_minutes=None,
        training_tool_calls_path=None,  # Skip training filter
        min_humanpause_duration_seconds=threshold_seconds
    )
    
    print(f"\n✓ Extracted {len(templates)} templates")
    
    # Generate statistics
    summary = generate_statistics(templates, threshold_seconds)
    
    # Save summary to JSON
    output_file = f"/vllm/vllm/elasticswap/frontend/example_traces/template_statistics_t{int(threshold_seconds)}s.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {output_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

