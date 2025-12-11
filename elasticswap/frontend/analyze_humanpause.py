#!/usr/bin/env python3
"""
Analyze HumanPause durations in the extracted templates.
"""

import json
import os

def analyze_humanpause_durations():
    """Count HumanPause toolcalls by duration threshold."""
    templates_file = "/vllm/vllm/elasticswap/frontend/example_traces/all_templates.json"
    
    if not os.path.exists(templates_file):
        print(f"Error: Templates file not found: {templates_file}")
        return
    
    print(f"Loading templates from {templates_file}...")
    with open(templates_file, 'r') as f:
        templates = json.load(f)
    
    print(f"Loaded {len(templates)} templates\n")
    
    # Collect all HumanPause durations
    humanpause_durations = []
    humanpause_details = []
    
    for template in templates:
        for turn in template.get('turns', []):
            for tool in turn.get('tools', []):
                if tool.get('name') == 'HumanPause':
                    duration = tool.get('duration_seconds', 0)
                    humanpause_durations.append(duration)
                    humanpause_details.append({
                        'template_id': template.get('template_id'),
                        'turn_idx': turn.get('turn_idx'),
                        'duration': duration,
                        'timestamp': tool.get('start_timestamp')
                    })
    
    if not humanpause_durations:
        print("No HumanPause toolcalls found!")
        return
    
    print(f"Total HumanPause toolcalls: {len(humanpause_durations)}\n")
    
    # Count by thresholds
    thresholds = [60, 120, 180, 240, 300, 600, 900, 1800, 3600]
    
    print("HumanPause duration distribution:")
    print("-" * 60)
    print(f"{'Threshold (seconds)':<25} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    
    total = len(humanpause_durations)
    for threshold in thresholds:
        count = sum(1 for d in humanpause_durations if d > threshold)
        pct = (count / total * 100) if total > 0 else 0
        print(f"> {threshold:>6} seconds{'':<10} {count:<10} {pct:>6.2f}%")
    
    # Specific answer for > 300 seconds
    count_above_300 = sum(1 for d in humanpause_durations if d > 300)
    print("\n" + "=" * 60)
    print(f"HumanPause toolcalls with duration > 300 seconds: {count_above_300}")
    print(f"Percentage: {count_above_300 / total * 100:.2f}%")
    print("=" * 60)
    
    # Show statistics
    print(f"\nHumanPause Duration Statistics:")
    print(f"  Min: {min(humanpause_durations):.2f}s")
    print(f"  Max: {max(humanpause_durations):.2f}s")
    print(f"  Mean: {sum(humanpause_durations) / len(humanpause_durations):.2f}s")
    print(f"  Median: {sorted(humanpause_durations)[len(humanpause_durations) // 2]:.2f}s")
    
    # Show examples of long pauses
    long_pauses = sorted(humanpause_details, key=lambda x: x['duration'], reverse=True)[:10]
    print(f"\nTop 10 longest HumanPause durations:")
    for i, pause in enumerate(long_pauses, 1):
        print(f"  {i}. Template {pause['template_id']}, Turn {pause['turn_idx']}: "
              f"{pause['duration']:.2f}s ({pause['duration']/60:.1f} minutes)")
    
    # Count exactly at 300 seconds
    count_exactly_300 = sum(1 for d in humanpause_durations if d == 300)
    count_300_or_above = sum(1 for d in humanpause_durations if d >= 300)
    print(f"\nAdditional info:")
    print(f"  Exactly 300 seconds: {count_exactly_300}")
    print(f"  >= 300 seconds: {count_300_or_above}")

if __name__ == "__main__":
    analyze_humanpause_durations()


