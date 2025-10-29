# Trace Duration Analysis Summary

## Problem Statement
We're running 20-minute experiments, but need to understand:
1. How long are the Claude traces?
2. Do they contain human pauses (long gaps between LLM calls)?
3. Can we fit them within 20-minute experiments?

## Key Findings

### Original Traces (No Processing)
- **Total traces:** 151
- **Median duration:** 35.82 minutes
- **Mean duration:** 745.13 minutes (~12 hours!)
- **Traces over 20 min:** 96 (63.6%)
- **Longest trace:** 11,619 minutes (~8 days!)

### Human Pauses Analysis
- **88.6%** of traces have at least one gap > 1 minute between LLM calls
- **53.0%** of traces have at least one gap > 5 minutes
- **36.2%** of traces have at least one gap > 10 minutes

However, most individual gaps are reasonable:
- **Median gap:** 7.05 seconds
- **P95 gap:** 90.56 seconds
- **Only 1.3%** of all gaps exceed 5 minutes

**Conclusion:** Traces contain normal interaction patterns punctuated by occasional long human pauses (user stepped away, thinking, etc.)

## Splitting Strategy Results

### 5-Minute Gap Threshold (RECOMMENDED)
Splitting traces when there's a >5 minute gap between LLM calls:
- **Total traces:** 434 (+283 new sub-traces)
- **Median duration:** 8.37 minutes ✓
- **Mean duration:** 14.83 minutes ✓
- **Traces over 20 min:** 95 (21.9%) - much better!
- **Distribution:**
  - 0-5 min: 38.9%
  - 5-10 min: 17.7%
  - 10-15 min: 13.8%
  - 15-20 min: 7.6%
  - >20 min: 21.9%

### 10-Minute Gap Threshold
- **Traces over 20 min:** 33.5% (worse)
- Less aggressive splitting, still many long traces

### 15-Minute Gap Threshold
- **Traces over 20 min:** 41.8% (much worse)

## Recommendation

### Two-Step Filtering Approach:

1. **Split at 5-minute gaps** (300 seconds)
   - Splits traces at clear human pause boundaries
   - Preserves natural workflow patterns
   - Reduces long traces significantly

2. **Filter out remaining traces > 20 minutes**
   - After splitting, 21.9% (95 traces) still exceed 20 minutes
   - These are likely edge cases with complex, sustained interactions
   - Filtering them leaves us with 339 traces (78.1% of split traces)

### Final Result:
- **339 usable traces** (all ≤ 20 minutes)
- **Median duration:** ~8 minutes
- **Good distribution** of trace lengths
- **Natural interaction patterns preserved**

## Implementation

The splitting logic should be added to `extract_trace_templates.py`:

```python
HUMAN_PAUSE_THRESHOLD = 300  # 5 minutes in seconds
MAX_TRACE_DURATION = 20 * 60  # 20 minutes in seconds

def split_trace_at_gaps(session_events, gap_threshold_seconds):
    """Split session into sub-traces at large LLM-to-LLM gaps"""
    # Implementation in analyze_trace_splitting.py
    ...

def filter_trace_duration(trace_events, max_duration_seconds):
    """Check if trace duration is within acceptable limits"""
    timestamps = [parse(e['timestamp']).timestamp() for e in trace_events]
    return (timestamps[-1] - timestamps[0]) <= max_duration_seconds
```

## Next Steps

1. Update `extract_trace_templates_with_evolution()` to:
   - Split traces at 5-minute gaps
   - Filter out traces > 20 minutes after splitting
   
2. Re-run trace extraction with new filtering

3. Verify that remaining traces work well in 20-minute experiments

4. Update experiment configurations if needed


