#!/usr/bin/env python3
"""
Test the trace splitting functionality by loading traces and showing statistics.
"""

from extract_trace_templates import extract_trace_templates_with_evolution
import numpy as np

def test_splitting():
    """Test the trace splitting with different parameters"""
    
    print("="*70)
    print("Testing Trace Splitting Implementation")
    print("="*70)
    
    # Test 1: Original (no splitting, no duration filter)
    print("\n\n1. BASELINE: No splitting, no duration filter")
    print("-" * 70)
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir="/vllm/vllm/elasticswap/toolcall_dataset_claude",
        trace_filter_percentile=90.0,
        split_at_gap_seconds=999999,  # Effectively no splitting
        max_trace_duration_minutes=999999  # No duration filter
    )
    print(f"\nResult: {len(templates)} traces extracted")
    
    # Test 2: With 5-minute gap splitting + 20-minute duration filter
    print("\n\n2. WITH SPLITTING: 5-minute gaps + 20-minute duration filter")
    print("-" * 70)
    templates = extract_trace_templates_with_evolution(
        claude_dataset_dir="/vllm/vllm/elasticswap/toolcall_dataset_claude",
        trace_filter_percentile=90.0,
        split_at_gap_seconds=300,  # 5 minutes
        max_trace_duration_minutes=20  # 20 minutes
    )
    print(f"\nResult: {len(templates)} traces extracted")
    
    # Show some trace statistics
    if templates:
        print("\n" + "="*70)
        print("TRACE STATISTICS")
        print("="*70)
        
        num_turns = [t.num_turns for t in templates]
        num_tools = [t.total_tools for t in templates]
        context_sizes = [t.total_context_accumulated for t in templates]
        
        print(f"\nNumber of turns per trace:")
        print(f"  Min: {min(num_turns)}, Max: {max(num_turns)}, Mean: {np.mean(num_turns):.1f}, Median: {np.median(num_turns):.1f}")
        
        print(f"\nTools per trace:")
        print(f"  Min: {min(num_tools)}, Max: {max(num_tools)}, Mean: {np.mean(num_tools):.1f}, Median: {np.median(num_tools):.1f}")
        
        print(f"\nContext size (tokens):")
        print(f"  Min: {min(context_sizes)}, Max: {max(context_sizes)}, Mean: {np.mean(context_sizes):.0f}, Median: {np.median(context_sizes):.0f}")
        
        print(f"\n✓ Successfully extracted {len(templates)} traces suitable for 20-minute experiments!")
        
        # Show a few example traces
        print(f"\n{'='*70}")
        print("EXAMPLE TRACES (first 3)")
        print("="*70)
        for i, template in enumerate(templates[:3]):
            print(f"\n{i+1}. {template}")

if __name__ == "__main__":
    test_splitting()


