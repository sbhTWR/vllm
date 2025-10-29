# Determinism Improvements for Claude Trace Workload Generation

## Summary

Made changes to ensure that `execute_workload_claude_traces()` generates **identical requests** across multiple executions with the same seed.

## Problem

The original implementation had two sources of non-determinism:

1. **`glob.glob()` file ordering**: The function used `glob.glob()` to find JSON files, which doesn't guarantee consistent ordering across runs or systems
2. **Dictionary iteration order**: When iterating over `sessions.items()`, the order could vary (though in Python 3.7+ dicts are insertion-ordered, the insertion order depended on the glob order)

## Changes Made

### 1. **extract_trace_templates.py** (Line 213)
```python
# Before:
json_files = glob.glob(f"{claude_dataset_dir}/*_events.json")

# After:
json_files = sorted(glob.glob(f"{claude_dataset_dir}/*_events.json"))
```
**Impact**: Files are now processed in a consistent alphabetical order.

### 2. **extract_trace_templates.py** (Line 268)
```python
# Before:
for session_id, session_events in sessions.items():

# After:
for session_id, session_events in sorted(sessions.items()):
```
**Impact**: Sessions are now processed in a consistent order sorted by session ID.

### 3. **extract_trace_templates.py** (Lines 709-729)
Added a determinism verification hash that:
- Includes all token IDs from all requests
- Includes all arrival times
- Prints the hash to stdout for easy verification

**Output example:**
```
✓ Generated Claude trace workload:
  Requests: 10
  Arrival rate: 0.5 req/s
  Seed: 42
  Total duration: 24.3s
  Avg turns per request: 3.2
  Avg tokens per request: 1234
  Total DAG nodes: 64
  Determinism hash: a1b2c3d4e5f6g7h8
    (This hash should be identical across runs with same seed)
```

### 4. **pipeline.py** (Lines 543-546)
Updated docstring to document the determinism guarantee:
```python
"""
DETERMINISM GUARANTEE:
Multiple executions with the same seed will generate identical requests
(same token IDs, arrival times, and execution order). A verification hash
is printed to confirm determinism across runs.
"""
```

## Testing

### Quick Test
A test script is provided: `test_determinism.py`

```bash
cd /vllm/vllm/elasticswap/frontend
python test_determinism.py 3 10
```

This will:
- Run workload generation 3 times
- Generate 10 requests each time
- Verify all runs produce identical results
- Print success/failure with detailed diagnostics

### Manual Verification
You can also manually verify by looking for the "Determinism hash" in the output:

```python
from extract_trace_templates import generate_claude_trace_workload

# Run 1
dags1, names1, times1, meta1 = generate_claude_trace_workload(
    num_requests=10, seed=42, arrival_rate=0.5
)
# Check the printed "Determinism hash"

# Run 2
dags2, names2, times2, meta2 = generate_claude_trace_workload(
    num_requests=10, seed=42, arrival_rate=0.5
)
# Verify the hash matches Run 1
```

## Verification

The determinism is guaranteed by:

1. ✅ **Sorted file iteration**: Files are always processed in the same order
2. ✅ **Sorted session iteration**: Sessions are always processed in the same order  
3. ✅ **Seeded RNG**: All random operations use `np.random.RandomState(seed)`
4. ✅ **Deterministic hash**: A verification hash is computed from all token IDs and arrival times

## Impact

- **No breaking changes**: All existing functionality remains the same
- **Reproducible experiments**: Running the same experiment with the same seed will now produce identical results
- **Easy verification**: The printed hash makes it trivial to verify determinism
- **Better debugging**: Deterministic behavior makes debugging much easier

## What's Already Deterministic

The following were already deterministic:
- ✅ Event timestamp sorting (line 257)
- ✅ Random number generation in `generate_request_trajectories()`
- ✅ Random number generation in `generate_claude_trace_workload()`
- ✅ Token ID generation in `generate_unique_token_sequence()`

## Usage Example

```python
# In your experiments:
execute_workload_claude_traces(
    num_requests=100,
    arrival_rate=0.5,
    seed=42,  # Same seed = identical workload
    port=8000,
    model="Qwen/Qwen2.5-Coder-32B-Instruct"
)

# Running this twice will:
# 1. Generate the exact same 100 requests
# 2. With the same token IDs
# 3. With the same arrival times
# 4. Print the same determinism hash both times
```

## Notes

- The global seeds (`random.seed(42)` and `np.random.seed(42)` at the top of pipeline.py) are still set, but the workload generation uses its own `RandomState` instances to avoid interference
- The determinism hash is a SHA256 hash of all token IDs and arrival times, truncated to 16 characters for readability














