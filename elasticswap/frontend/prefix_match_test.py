import random
from lorem_text import lorem
import numpy as np


random.seed(42)
np.random.seed(42)

def generate_variable_llm_toolcall_workload(num_requests: int, 
                                            seed: int = None, 
                                            vary_interrupts: bool = True):
    """
    Generate a list of LLM request parameter dicts:
    - request_size: Number of tokens (capped at 30,000)
    - num_interrupts: Number of tool calls
    - interrupt_len: (if vary_interrupts=False) Constant duration for all tool calls
    - interrupt_lens: (if vary_interrupts=True) List of durations for each tool call
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    
    workloads = []
    
    for _ in range(num_requests):
        # 1. Request size: log-normal distribution, capped at 30k
        # request_size = int(np.clip(
        #     np.random.lognormal(mean=np.log(15000), sigma=0.5),
        #     8000, 30000
        # ))

        request_size = 30000
        
        # 2. Number of tool call interrupts
        num_interrupts = int(np.random.choice(
            [1, 2, 3, 4, 5, 6, 8, 10],
            p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.15, 0.07, 0.03]
        ))
        
        # 3. Interrupt durations
        max_total_interrupt_time = 1 * request_size / 1000  # e.g., 15s for 30k

        if vary_interrupts:
            # Generate individual tool call durations
            durations = []
            for _ in range(num_interrupts):
                dur = float(np.clip(
                    rng.lognormal(mean=np.log(1.0), sigma=0.7),
                    0.1, 25.0
                ))
                durations.append(dur)
            
            # Normalize total time if necessary
            total = sum(durations)
            if total > max_total_interrupt_time:
                scale = max_total_interrupt_time / total
                durations = [round(d * scale, 3) for d in durations]
            else:
                durations = [round(d, 3) for d in durations]
            
            workload = {
                "request_size": request_size,
                "num_interrupts": num_interrupts,
                "interrupt_lens": durations
            }
        else:
            # Generate one constant duration
            # dur = float(np.clip(
            #     rng.lognormal(mean=np.log(1.0), sigma=0.7),
            #     0.1, 250.0
            # ))
            # dur = min(dur, max_total_interrupt_time / num_interrupts)
            num = rng.integers(1, 20)
            dur = int(num)
            workload = {
                "context": lorem.words(request_size),
                "num_interrupts": num_interrupts,
                "interrupt_len": dur
            }
        
        workloads.append(workload)
    
    return workloads

def longest_common_prefix(s1: str, s2: str) -> int:
    """Return length of the longest common prefix between s1 and s2."""
    match_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            match_len += 1
        else:
            break
    return match_len

def compute_prefix_match_matrix(strings: list[str]) -> list[list[int]]:
    """Return a 2D matrix where entry [i][j] is the prefix match length between strings[i] and strings[j]."""
    n = len(strings)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = longest_common_prefix(strings[i], strings[j])
    
    return matrix


requests = generate_variable_llm_toolcall_workload(50, 42, False)

contexts = []

for request in requests:
    context = request['context']
    contexts.append(context)

matrix = compute_prefix_match_matrix(contexts)

# Print matrix
print("Prefix Match Matrix:")
for row in matrix:
    print(row)