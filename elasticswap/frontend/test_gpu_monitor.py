#!/usr/bin/env python3
"""Test script for GPUMonitor class"""

import os
import sys
import time

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import GPUMonitor

def test_gpu_monitor():
    """Test the GPU monitor for a short duration"""
    print("Testing GPUMonitor class...")
    
    # Create a test output file
    test_output_file = "/tmp/test_gpu_utilization.csv"
    
    # Remove existing test file if it exists
    if os.path.exists(test_output_file):
        os.remove(test_output_file)
        print(f"Removed existing test file: {test_output_file}")
    
    # Create monitor with 0.5 second interval for faster testing
    monitor = GPUMonitor(test_output_file, interval=0.5)
    
    print(f"Starting GPU monitor (will run for 5 seconds)...")
    print(f"Output file: {test_output_file}")
    
    # Start monitoring
    monitor.start()
    
    # Run for 5 seconds
    try:
        time.sleep(5)
        print("Test duration complete")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Stop monitoring
    monitor.stop()
    
    # Check if file was created
    if os.path.exists(test_output_file):
        print(f"\n✓ Test file created: {test_output_file}")
        
        # Read and display first few lines
        with open(test_output_file, 'r') as f:
            lines = f.readlines()
            print(f"\nFile contains {len(lines)} lines (including header)")
            print("\nFirst 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"  {i+1}: {line.strip()}")
            
            if len(lines) > 5:
                print(f"\n... (showing first 5 of {len(lines)} lines)")
        
        # Check file size
        file_size = os.path.getsize(test_output_file)
        print(f"\nFile size: {file_size} bytes")
        
        if len(lines) > 1:
            print("\n✓ Test PASSED: GPU monitor successfully collected data")
            return True
        else:
            print("\n✗ Test FAILED: File contains only header, no data collected")
            return False
    else:
        print(f"\n✗ Test FAILED: Output file not created")
        return False

if __name__ == "__main__":
    success = test_gpu_monitor()
    sys.exit(0 if success else 1)




