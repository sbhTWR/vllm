# Quick Reference: Template Diversity & Determinism

## TL;DR

✅ **Workloads are now fully deterministic** - same seed = identical requests  
✅ **Template diversity is tracked** - you can see which trace types are used  
✅ **Easy verification** - automated tools confirm both properties  

---

## At a Glance

### What You See in the Output

```
✓ Generated Claude trace workload:
  Requests: 50
  Seed: 42                                    ← Determinism guarantee
  
  Template diversity:                         ← Diversity tracking
    Available templates: 267
    Unique templates used: 39 (14.6%)         ← Coverage percentage
    
    Top templates used:                       ← Distribution visibility
      2b8f1410-0: 3 times (6.0%)
      21a42386-8: 3 times (6.0%)
      ...
  
  Determinism hash: 0efc3062b14751d3          ← Verification hash
```

### What the Numbers Mean

| Metric | Good | Needs Work | Action |
|--------|------|------------|--------|
| **Template coverage** | >80% | <50% | Increase num_requests |
| **Max usage %** | <10% | >20% | Increase num_requests |
| **Normalized entropy** | >90% | <70% | Increase num_requests |
| **Determinism hash** | Same across runs | Different | Check seed |

---

## Quick Commands

### Verify Determinism
```bash
python test_determinism.py 3 10
# Runs 3 times, should show identical hashes
```

### Analyze Diversity
```bash
python analyze_template_diversity.py path/to/requests_meta.json
# Shows detailed diversity analysis with entropy metrics
```

### Generate Good Workload
```python
# Rule: Use 5× the number of templates in requests
num_templates = 267
num_requests = 267 * 5  # = 1335 requests

generate_claude_trace_workload(
    num_requests=1335,   # Good diversity
    seed=42,             # Determinism
    arrival_rate=0.5
)
```

---

## Common Scenarios

### ❓ "Are my experiments using diverse trace types?"

**Check the output:**
```
Template diversity:
  Unique templates used: 38 (14.6%)  ← Only 14.6% coverage
  
⚠ Note: 228 templates were not sampled
```

**Action:** Increase `num_requests` (need ~1335 for 267 templates)

---

### ❓ "Are my experiments reproducible?"

**Check the hash:**
```bash
# Run 1
Determinism hash: 0efc3062b14751d3

# Run 2 (same seed)
Determinism hash: 0efc3062b14751d3  ← Should match!
```

**If different:** Check that seed is the same

---

### ❓ "Is one template dominating my workload?"

**Check the distribution:**
```
Top templates used:
  abc123: 25 times (50.0%)  ← One template is 50% of requests!
```

**If max > 20%:** Increase `num_requests` for better distribution

---

### ❓ "How many requests do I need?"

**Formula:** `num_requests = num_templates × 5`

**Example:**
- 45 templates → 225 requests (≥90% coverage)
- 267 templates → 1335 requests (≥90% coverage)

---

## Output Interpretation

### Template Diversity Section

```
Template diversity:
  Available templates: 267           ← Total unique traces extracted
  Unique templates used: 39 (14.6%)  ← How many you're actually using
  Usage per template: min=1, max=3, avg=1.3
                      ↑     ↑         ↑
                      │     │         └─ Average uses per template
                      │     └─ Most used template count
                      └─ Least used template count (always 1)
```

### Top Templates List

```
Top templates used:
  2b8f1410-0: 3 times (6.0%)  ← Template ID: count (percentage)
  21a42386-8: 3 times (6.0%)
  ...
```
- Shows which templates are most common
- None should be >10% for good diversity

### Warning Messages

```
⚠ Note: 228 templates were not sampled
```
- **Meaning:** Only some templates were used (expected with small samples)
- **Action:** Increase `num_requests` if you want better coverage

---

## Best Practices

### ✅ DO

1. **Set a seed** for reproducibility
   ```python
   seed=42  # Always use the same seed for comparable experiments
   ```

2. **Use enough requests** for diversity
   ```python
   num_requests = num_templates * 5  # Good rule of thumb
   ```

3. **Save metadata** for analysis
   ```python
   results_dir="/path/to/results"  # Enables post-experiment analysis
   ```

4. **Check the output** every time
   - Look at template coverage %
   - Verify determinism hash matches across runs
   - Ensure no template dominates (>20%)

### ❌ DON'T

1. **Don't ignore warnings**
   ```
   ⚠ Note: 228 templates were not sampled
   ```
   This means poor diversity - increase requests!

2. **Don't forget the seed**
   ```python
   generate_claude_trace_workload(num_requests=100)  # ❌ Non-deterministic
   generate_claude_trace_workload(num_requests=100, seed=42)  # ✅ Deterministic
   ```

3. **Don't use too few requests**
   ```python
   # 267 templates available
   num_requests=10  # ❌ Only uses ~4% of templates
   num_requests=1335  # ✅ Uses ~95% of templates
   ```

---

## Example Workflow

```python
# 1. Generate workload
from extract_trace_templates import generate_claude_trace_workload

dags, names, times, meta = generate_claude_trace_workload(
    num_requests=1335,      # 5× num templates
    arrival_rate=0.5,
    seed=42,                # For reproducibility
    results_dir="./results" # Save metadata
)

# 2. Check output (look for):
#    ✓ Unique templates used: >80%
#    ✓ Determinism hash: <printed>
#    ✓ No warnings about templates not sampled

# 3. Run experiment
# ... your experiment code ...

# 4. Verify diversity after
# python analyze_template_diversity.py ./results/requests_meta.json
#    ✓ Normalized entropy: >90%
#    ✓ Template coverage: >80%
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `extract_trace_templates.py` | Core workload generation (modified for diversity tracking) |
| `pipeline.py` | Experiment runner (saves diversity metadata) |
| `test_determinism.py` | Automated determinism testing |
| `analyze_template_diversity.py` | Post-experiment diversity analysis |
| `DETERMINISM_CHANGES.md` | Detailed determinism documentation |
| `TEMPLATE_DIVERSITY.md` | Detailed diversity documentation |
| `WORKLOAD_IMPROVEMENTS_SUMMARY.md` | Complete summary of all changes |
| `QUICK_REFERENCE.md` | This file |

---

## Need Help?

- **Determinism not working?** → See `DETERMINISM_CHANGES.md`
- **Poor diversity?** → See `TEMPLATE_DIVERSITY.md`
- **Want full details?** → See `WORKLOAD_IMPROVEMENTS_SUMMARY.md`
- **Quick test?** → Run `python test_determinism.py`














