# Trace Splitting Quick Reference

## Overview

Traces from the Claude dataset have been processed to:
1. ✅ **Split at human pauses** (gaps > 5 minutes between LLM calls)
2. ✅ **Filter by duration** (remove traces > 20 minutes)
3. ✅ **Filter by tool time** (remove traces with abnormally long tool executions)

## Results

- **267 traces** ready for 20-minute experiments (was 151 original sessions)
- **0% exceed 20 minutes** (all filtered out)
- **76% of data preserved** after quality filtering

## Usage

### Default (20-minute experiments)

```python
from extract_trace_templates import extract_trace_templates_with_evolution

templates = extract_trace_templates_with_evolution(
    claude_dataset_dir="/vllm/vllm/elasticswap/toolcall_dataset_claude"
)
# Returns 267 traces suitable for 20-min experiments
```

### Custom Configuration

```python
# For different experiment durations
templates = extract_trace_templates_with_evolution(
    claude_dataset_dir="/vllm/vllm/elasticswap/toolcall_dataset_claude",
    trace_filter_percentile=90.0,        # Tool time P90 threshold
    split_at_gap_seconds=300,             # Split at 5-min gaps (adjustable)
    max_trace_duration_minutes=20         # Max trace length (adjustable)
)
```

## Testing

Run the test script to verify:

```bash
cd /vllm/vllm/elasticswap/frontend
python test_trace_splitting.py
```

## Documentation

- **TRACE_ANALYSIS_SUMMARY.md** - Initial problem analysis
- **TRACE_SPLITTING_RESULTS.md** - Implementation results and statistics
- **analyze_trace_splitting.py** - Simulation tool for different thresholds
- **test_trace_splitting.py** - Verification test

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_at_gap_seconds` | 300 | Split traces when LLM-to-LLM gap exceeds this (seconds) |
| `max_trace_duration_minutes` | 20 | Filter traces longer than this (minutes) |
| `trace_filter_percentile` | 90.0 | Tool time percentile for filtering |

## Example Output

```
===== Trace Processing Summary =====
Original sessions split: 79
Total sub-traces after splitting: 351
Filtered (tool time too high): 84 (23.9%)
Filtered (duration > 20 min): 93 (26.5%)
Kept traces: 267 (76.1%)
Gap threshold used: 300s (5 minutes)
```

## Trace Statistics

- **Turns per trace**: Median 17, Mean 29
- **Tools per trace**: Median 17, Mean 23
- **Context size**: Median 73K tokens, Mean 77K tokens

All traces now fit comfortably within 20-minute experiment windows! 🎉


