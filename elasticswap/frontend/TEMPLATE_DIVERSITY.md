# Template Diversity Tracking

## Overview

The Claude trace workload generator now provides **comprehensive visibility** into template diversity, ensuring that your experiments use a **diverse mix of trace types** rather than repeatedly sampling the same patterns.

## What Changed

### 1. Real-time Diversity Statistics

When generating workloads, you now see detailed template diversity information:

```
✓ Generated 100 trajectories
  Avg turns per request: 3.2
  Avg tokens per request: 1234

  Template diversity:
    Available templates: 45
    Unique templates used: 38 (84.4%)
    Usage per template: min=1, max=5, avg=2.6

    Top templates used:
      a1b2c3d4: 5 times (5.0%)
      e5f6g7h8: 4 times (4.0%)
      i9j0k1l2: 4 times (4.0%)
      m3n4o5p6: 3 times (3.0%)
      q7r8s9t0: 3 times (3.0%)
      ...
```

### 2. Saved Template Diversity Metadata

The `requests_meta.json` file now includes template diversity statistics:

```json
{
  "arrival_times": [...],
  "requests_meta": [...],
  "template_diversity": {
    "total_templates_available": 45,
    "unique_templates_used": 38,
    "template_usage": {
      "a1b2c3d4": 5,
      "e5f6g7h8": 4,
      ...
    },
    "most_common": [
      ["a1b2c3d4", 5],
      ["e5f6g7h8", 4],
      ...
    ]
  }
}
```

### 3. Analysis Tool

A new script `analyze_template_diversity.py` provides detailed post-analysis:

```bash
python analyze_template_diversity.py ../results/experiment-name/requests_meta.json
```

**Output includes:**
- Template usage distribution (table and histogram)
- Shannon entropy analysis (diversity quality metric)
- Diversity assessment (excellent/good/moderate/poor)
- Warnings if any templates dominate

## How Template Sampling Works

### Random Sampling with Replacement

Templates are sampled **uniformly at random with replacement**:

```python
template = rng.choice(templates)  # Each template has equal probability
```

This means:
- ✅ Every template has an **equal chance** of being selected
- ✅ **Diversity increases** with more requests
- ✅ **Deterministic** with the same seed
- ⚠️ Small request counts may not cover all templates

### Expected Coverage

With **uniform random sampling**, the expected number of unique templates used follows the **coupon collector problem**:

| Requests | Expected Unique Templates (out of 45) | Coverage |
|----------|--------------------------------------|----------|
| 10       | ~9                                   | 20%      |
| 50       | ~31                                  | 69%      |
| 100      | ~38                                  | 84%      |
| 200      | ~42                                  | 93%      |
| 500      | ~44                                  | 98%      |

**Rule of thumb:** To use ≥90% of templates, generate at least **5× the number of templates** in requests.

## Usage Examples

### Example 1: Check Diversity During Generation

```python
from extract_trace_templates import generate_claude_trace_workload

dags, names, times, meta = generate_claude_trace_workload(
    num_requests=100,
    seed=42
)

# Output will show:
#   Template diversity:
#     Available templates: 45
#     Unique templates used: 38 (84.4%)
#     ...
```

### Example 2: Analyze Diversity After Experiment

```bash
# After running your experiment
cd /vllm/vllm/elasticswap/frontend
python analyze_template_diversity.py \
    ../results/oracle-test-159-claude-num-rate-1-batch-64/requests_meta.json
```

Output:
```
==================================================================
TEMPLATE DIVERSITY ANALYSIS
==================================================================
File: ../results/.../requests_meta.json

Total requests: 100
Unique templates used: 38
Available templates: 45
Template coverage: 84.4%

Usage statistics:
  Min usage per template: 1
  Max usage per template: 5
  Avg usage per template: 2.6

Shannon Entropy: 5.12 bits
Max possible entropy: 5.25 bits
Normalized entropy: 97.5%
  (100% = perfectly uniform distribution)

✅ Excellent diversity: Templates are very evenly distributed
```

### Example 3: Verify Same Templates Across Runs

```python
# Run 1
dags1, _, _, meta1 = generate_claude_trace_workload(num_requests=100, seed=42)

# Run 2
dags2, _, _, meta2 = generate_claude_trace_workload(num_requests=100, seed=42)

# The "Top templates used" section should be IDENTICAL
# This verifies both determinism AND template diversity
```

## Interpreting Diversity Metrics

### Shannon Entropy

**Shannon entropy** measures the **uniformity** of the distribution:

- **0 bits**: All requests use the same template (no diversity)
- **log₂(N) bits**: Perfectly uniform distribution (maximum diversity)
- **Normalized entropy**: Percentage of maximum possible diversity

**Quality thresholds:**
- **>95%**: Excellent - nearly perfect uniform distribution
- **85-95%**: Good - reasonably uniform
- **70-85%**: Moderate - some skew present
- **<70%**: Poor - heavily skewed distribution

### Usage Distribution

Check the histogram to identify:
- **Even bars**: Good diversity
- **One tall bar**: Dominant template (poor diversity)
- **Long tail**: Many templates used once (expected with random sampling)

## Tips for Better Diversity

### 1. Use Enough Requests

**Problem:** Only 10 requests from 45 templates
```
⚠ Note: 35 templates were not sampled (increase num_requests for more diversity)
```

**Solution:** Use at least 5× the number of templates:
```python
num_requests = len(templates) * 5  # For 45 templates → 225 requests
```

### 2. Check for Dominant Templates

**Problem:**
```
⚠ Note: The most common template accounts for 25.0% of requests
```

**Solution:** This is expected with small sample sizes. Increase `num_requests`.

### 3. Verify Diversity in Production

Always check the diversity output when running experiments:

```python
# Good diversity output:
#   Unique templates used: 42 (93.3%)
#   Usage per template: min=3, max=8, avg=5.2
#   Normalized entropy: 96.2%

# Poor diversity output:
#   Unique templates used: 12 (26.7%)
#   Usage per template: min=1, max=35, avg=8.3
#   Normalized entropy: 62.4%
```

## FAQs

### Q: Why don't all templates get used?

**A:** With random sampling, some templates may not be selected (especially with small request counts). This is expected behavior called the "coupon collector problem."

To use all templates, generate ~5× more requests than templates:
- 45 templates → use 225+ requests
- 100 templates → use 500+ requests

### Q: Is the template selection deterministic?

**A:** Yes! With the same seed, you get:
1. Same templates selected
2. Same order of selection
3. Same usage distribution

Verify by checking the "Determinism hash" output.

### Q: What if I want guaranteed coverage of all templates?

**A:** Modify `generate_request_trajectories()` to use round-robin sampling:

```python
# Option 1: Cycle through all templates first, then random
templates_cycle = templates * (num_requests // len(templates) + 1)
for i in range(num_requests):
    template = templates_cycle[i]
    ...

# Option 2: Stratified sampling (uniform coverage)
template_indices = np.tile(np.arange(len(templates)), 
                          num_requests // len(templates) + 1)[:num_requests]
rng.shuffle(template_indices)
```

However, **random sampling is preferred** because it:
- Better represents realistic workloads
- Provides statistical properties useful for analysis
- Maintains full reproducibility with seeds

### Q: How do I know if my diversity is good enough?

**A:** Use these guidelines:

1. **Normalized entropy > 90%**: Excellent for most use cases
2. **Template coverage > 80%**: Good for comparative studies
3. **Max usage < 10% of requests**: No single template dominates

If any metric is below these thresholds, increase `num_requests`.

## Implementation Details

### Where Tracking Happens

1. **`generate_request_trajectories()`** (lines 534-582)
   - Tracks `template_usage` with Counter
   - Prints diversity statistics
   - Shows top 10 templates

2. **`generate_claude_trace_workload()`** (lines 692-696)
   - Re-computes stats from trajectories
   - Includes in metadata

3. **`execute_workload_claude_traces()`** (lines 557-576)
   - Adds diversity to `requests_meta.json`
   - Saves for post-analysis

### Key Functions

```python
# Track template usage
from collections import Counter
template_usage = Counter()
for traj in trajectories:
    template_usage[traj.template_id] += 1

# Analyze diversity
unique_templates_used = len(template_usage)
coverage = unique_templates_used / total_templates_available

# Top templates
top_10 = template_usage.most_common(10)
```

## Summary

The template diversity features ensure that:

✅ **Visibility**: You can see exactly which templates are being used  
✅ **Verification**: Easy to verify diverse sampling is happening  
✅ **Analysis**: Post-experiment analysis of diversity quality  
✅ **Determinism**: Same seed = same template distribution  
✅ **Quality**: Entropy metrics help assess distribution uniformity  

This gives you confidence that your experiments are testing a **diverse mix of workload patterns**, not just variations of the same trace.














