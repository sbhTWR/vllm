#!/usr/bin/env python3
"""
Analyze template diversity from saved experiment results.

Usage:
    python analyze_template_diversity.py <path_to_requests_meta.json>

Example:
    python analyze_template_diversity.py ../results/oracle-test-159-claude-num-rate-1-batch-64/requests_meta.json
"""

import sys
import json
from collections import Counter


def analyze_template_diversity(meta_file_path):
    """Analyze and display template diversity statistics."""
    
    with open(meta_file_path, 'r') as f:
        data = json.load(f)
    
    requests_meta = data['requests_meta']
    total_requests = len(requests_meta)
    
    # Count template usage
    template_counts = Counter()
    for req in requests_meta:
        template_counts[req['template_id']] += 1
    
    unique_templates = len(template_counts)
    
    print(f"\n{'='*70}")
    print(f"TEMPLATE DIVERSITY ANALYSIS")
    print(f"{'='*70}")
    print(f"File: {meta_file_path}")
    print(f"\nTotal requests: {total_requests}")
    print(f"Unique templates used: {unique_templates}")
    
    # Check if we have the new template_diversity field
    if 'template_diversity' in data:
        div = data['template_diversity']
        print(f"Available templates: {div.get('total_templates_available', 'N/A')}")
        coverage = (unique_templates / div.get('total_templates_available', 1)) * 100
        print(f"Template coverage: {coverage:.1f}%")
    
    # Statistics
    usage_values = list(template_counts.values())
    print(f"\nUsage statistics:")
    print(f"  Min usage per template: {min(usage_values)}")
    print(f"  Max usage per template: {max(usage_values)}")
    print(f"  Avg usage per template: {sum(usage_values)/len(usage_values):.1f}")
    
    # Show distribution
    print(f"\nTemplate usage distribution:")
    print(f"  {'Template ID':<20} {'Count':>8} {'Percentage':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12}")
    
    for template_id, count in template_counts.most_common():
        percentage = (count / total_requests) * 100
        print(f"  {template_id:<20} {count:>8} {percentage:>11.1f}%")
    
    # Histogram of usage
    print(f"\nUsage histogram:")
    max_bar_width = 50
    max_count = max(usage_values)
    for template_id, count in template_counts.most_common(20):
        bar_width = int((count / max_count) * max_bar_width)
        bar = '█' * bar_width
        print(f"  {template_id:<15} {bar} {count}")
    
    if len(template_counts) > 20:
        print(f"  ... ({len(template_counts) - 20} more templates)")
    
    # Diversity quality assessment
    print(f"\n{'='*70}")
    print(f"DIVERSITY ASSESSMENT")
    print(f"{'='*70}")
    
    # Calculate entropy as a measure of diversity
    import math
    probabilities = [count / total_requests for count in usage_values]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    max_entropy = math.log2(unique_templates)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    print(f"\nShannon Entropy: {entropy:.2f} bits")
    print(f"Max possible entropy: {max_entropy:.2f} bits")
    print(f"Normalized entropy: {normalized_entropy:.2%}")
    print(f"  (100% = perfectly uniform distribution)")
    
    if normalized_entropy > 0.95:
        print(f"\n✅ Excellent diversity: Templates are very evenly distributed")
    elif normalized_entropy > 0.85:
        print(f"\n✓ Good diversity: Templates are reasonably well distributed")
    elif normalized_entropy > 0.70:
        print(f"\n⚠ Moderate diversity: Some templates are overrepresented")
    else:
        print(f"\n⚠ Poor diversity: Distribution is heavily skewed")
    
    # Check for any dominant templates
    max_usage_pct = (max(usage_values) / total_requests) * 100
    if max_usage_pct > 20:
        print(f"\n⚠ Note: The most common template accounts for {max_usage_pct:.1f}% of requests")
        print(f"  Consider using more requests to achieve better template diversity")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    meta_file = sys.argv[1]
    
    try:
        analyze_template_diversity(meta_file)
    except FileNotFoundError:
        print(f"Error: File not found: {meta_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {meta_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)














