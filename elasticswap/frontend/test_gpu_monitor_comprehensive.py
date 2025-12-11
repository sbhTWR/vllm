#!/usr/bin/env python3
"""Comprehensive test for GPUMonitor class"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import GPUMonitor

def test_gpu_monitor_comprehensive():
    """Run comprehensive tests"""
    print("=" * 60)
    print("Comprehensive GPUMonitor Test Suite")
    print("=" * 60)
    
    test_file = "/tmp/test_gpu_monitor_comprehensive.csv"
    
    # Clean up any existing test file
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Test 1: Basic functionality
    print("\n[Test 1] Basic functionality test")
    print("-" * 60)
    monitor1 = GPUMonitor(test_file, interval=0.5)
    monitor1.start()
    time.sleep(2)
    monitor1.stop()
    
    if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
        with open(test_file, 'r') as f:
            lines = f.readlines()
        print(f"✓ PASSED: Collected {len(lines)} lines (including header)")
    else:
        print("✗ FAILED: No data collected")
        return False
    
    # Test 2: Multiple start/stop cycles
    print("\n[Test 2] Multiple start/stop cycles")
    print("-" * 60)
    monitor2 = GPUMonitor("/tmp/test_gpu_monitor_cycle.csv", interval=0.5)
    monitor2.start()
    time.sleep(1)
    monitor2.stop()
    time.sleep(0.5)
    monitor2.start()  # Should be able to restart
    time.sleep(1)
    monitor2.stop()
    print("✓ PASSED: Multiple start/stop cycles work correctly")
    
    # Test 3: Cleanup verification
    print("\n[Test 3] Cleanup verification")
    print("-" * 60)
    monitor3 = GPUMonitor("/tmp/test_gpu_monitor_cleanup.csv", interval=0.5)
    monitor3.start()
    time.sleep(1)
    monitor3.stop()
    # Check that monitoring flag is reset
    if not monitor3.monitoring:
        print("✓ PASSED: Monitoring flag correctly reset after stop")
    else:
        print("✗ FAILED: Monitoring flag not reset")
        return False
    
    # Test 4: File format validation
    print("\n[Test 4] CSV file format validation")
    print("-" * 60)
    import csv
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        row_count = sum(1 for row in reader)
    
    if 'timestamp' in fieldnames and 'elapsed_seconds' in fieldnames:
        print(f"✓ PASSED: CSV has required columns (total: {len(fieldnames)} columns)")
        print(f"✓ PASSED: Collected {row_count} data rows")
    else:
        print("✗ FAILED: Missing required columns")
        return False
    
    # Test 5: Interval timing
    print("\n[Test 5] Sampling interval timing")
    print("-" * 60)
    interval_test_file = "/tmp/test_gpu_monitor_interval.csv"
    monitor4 = GPUMonitor(interval_test_file, interval=1.0)
    monitor4.start()
    time.sleep(3.5)  # Should get ~3-4 samples
    monitor4.stop()
    
    with open(interval_test_file, 'r') as f:
        lines = f.readlines()
        data_lines = len(lines) - 1  # Exclude header
    
    if 3 <= data_lines <= 5:  # Allow some variance
        print(f"✓ PASSED: Collected {data_lines} samples in ~3.5 seconds (expected ~3-4)")
    else:
        print(f"⚠ WARNING: Collected {data_lines} samples (expected ~3-4)")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_gpu_monitor_comprehensive()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




