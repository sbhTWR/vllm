#!/usr/bin/env python3
"""
Test script to verify that execute_workload_claude_traces generates
identical requests across multiple runs with the same seed.

Usage:
    python test_determinism.py [num_runs] [num_requests]

Example:
    python test_determinism.py 3 10
    # Runs the workload generator 3 times with 10 requests each
"""

import sys
import os

# Set DEBUG mode to skip actual execution
os.environ['DEBUG'] = '1'

# Import after setting DEBUG
from extract_trace_templates import generate_claude_trace_workload
import numpy as np

def test_determinism(num_runs=3, num_requests=10, seed=42, arrival_rate=0.5):
    """
    Run workload generation multiple times and verify identical outputs.
    """
    print(f"\n{'='*70}")
    print(f"DETERMINISM TEST")
    print(f"{'='*70}")
    print(f"Testing {num_runs} runs with:")
    print(f"  - num_requests: {num_requests}")
    print(f"  - seed: {seed}")
    print(f"  - arrival_rate: {arrival_rate}")
    print(f"{'='*70}\n")
    
    results = []
    hashes = []
    
    for run_idx in range(num_runs):
        print(f"\n{'─'*70}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'─'*70}")
        
        # Generate workload
        dags, dag_names, arrival_times, requests_meta = generate_claude_trace_workload(
            num_requests=num_requests,
            arrival_rate=arrival_rate,
            seed=seed
        )
        
        # Extract hash from the print output (it's printed by the function)
        # For verification, we'll compute our own hash
        import hashlib
        hash_data = []
        
        # Include request IDs
        for name in dag_names:
            hash_data.append(name)
        
        # Include arrival times (rounded to microseconds for float comparison)
        for t in arrival_times:
            hash_data.append(f"{t:.6f}")
        
        # Include request metadata
        for meta in requests_meta:
            hash_data.append(str(meta['template_id']))
            hash_data.append(str(meta['num_turns']))
            hash_data.append(str(meta['total_input_tokens']))
        
        hash_str = '|'.join(hash_data)
        run_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:16]
        hashes.append(run_hash)
        
        results.append({
            'dag_names': dag_names,
            'arrival_times': arrival_times,
            'requests_meta': requests_meta,
            'hash': run_hash
        })
        
        print(f"\n  Run {run_idx + 1} verification hash: {run_hash}")
    
    # Verify all hashes are identical
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    
    all_identical = len(set(hashes)) == 1
    
    print(f"\nHashes from all runs:")
    for i, h in enumerate(hashes):
        print(f"  Run {i+1}: {h}")
    
    print(f"\nAll hashes identical: {all_identical}")
    
    if all_identical:
        print(f"\n✅ SUCCESS: All {num_runs} runs generated identical requests!")
        print(f"   The workload is fully deterministic with seed={seed}")
    else:
        print(f"\n❌ FAILURE: Runs generated different requests!")
        print(f"   This indicates non-determinism in the workload generation")
        
        # Show differences
        print(f"\n   Checking detailed differences:")
        base = results[0]
        for i in range(1, len(results)):
            print(f"\n   Run 1 vs Run {i+1}:")
            if results[i]['dag_names'] != base['dag_names']:
                print(f"     - Different request IDs")
            if not np.allclose(results[i]['arrival_times'], base['arrival_times']):
                print(f"     - Different arrival times")
            if results[i]['requests_meta'] != base['requests_meta']:
                print(f"     - Different request metadata")
        
        return False
    
    print(f"\n{'='*70}\n")
    return True


if __name__ == "__main__":
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    success = test_determinism(num_runs=num_runs, num_requests=num_requests)
    
    sys.exit(0 if success else 1)














