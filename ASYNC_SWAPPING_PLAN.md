# Async Swapping Model Implementation Plan

## Overview
This branch implements asynchronous block swapping using a separate CUDA stream to overlap computation with memory transfers, improving overall system throughput.

## Current State
- ✅ Fixed pinned blocks threshold issue in evictor and block allocator
- 🔄 Ready to implement async swapping infrastructure

## Implementation Phases

### Phase 1: Basic Async Infrastructure
- [ ] Create separate CUDA stream for swapping operations
- [ ] Implement AsyncSwapManager class
- [ ] Add block state tracking (RESIDENT_GPU, RESIDENT_CPU, SWAPPING_IN, SWAPPING_OUT, etc.)
- [ ] Modify block allocator to handle pending states
- [ ] Implement basic async swap operations

### Phase 2: Scheduler Integration
- [ ] Update scheduler to work with async swaps
- [ ] Implement conservative allocation strategy
- [ ] Add swap completion polling
- [ ] Integrate with existing block manager

### Phase 3: Advanced Features
- [ ] Implement predictive swapping
- [ ] Add swap prioritization
- [ ] Implement memory defragmentation
- [ ] Add performance monitoring

### Phase 4: Optimization
- [ ] Fine-tune swap thresholds
- [ ] Implement adaptive strategies
- [ ] Add comprehensive metrics
- [ ] Performance testing and optimization

## Key Components

### AsyncSwapManager
```python
class AsyncSwapManager:
    def __init__(self):
        self.swap_stream = cuda.Stream()
        self.pending_swaps = {}  # block_id -> SwapOperation
        self.swap_events = {}    # block_id -> cudaEvent
        self.swap_queue = deque()
    
    def launch_swap(self, block_id, src_device, dst_device):
        # Launch async swap operation
        # Return immediately, don't wait for completion
    
    def poll_completed_swaps(self):
        # Check which swaps have completed
        # Update block states accordingly
```

### Block States
- **RESIDENT_GPU**: Block is on GPU and available
- **RESIDENT_CPU**: Block is on CPU and available  
- **SWAPPING_IN**: Block is being transferred from CPU to GPU
- **SWAPPING_OUT**: Block is being transferred from GPU to CPU
- **PENDING_SWAP_IN**: Block is queued for swap-in but not yet started
- **PENDING_SWAP_OUT**: Block is queued for swap-out but not yet started

## Design Decisions

### Swap Granularity
- **Block-level**: Swap individual blocks (current approach)
- **Sequence-level**: Swap entire sequences together
- **Hybrid**: Swap blocks but group related blocks

### Swap Timing
- **Immediate**: Swap as soon as decision is made
- **Batched**: Collect swaps and execute in batches
- **Predictive**: Swap based on predicted future needs

### Request Handling Strategies
1. **Immediate Allocation**: Try to allocate from resident blocks first
2. **Wait for Swap**: If needed blocks are swapping, wait for completion
3. **Preemption**: Preempt less important sequences to free up blocks
4. **Recomputation**: As last resort, recompute instead of waiting for swap

## Performance Considerations

### Metrics to Track
- Swap completion time
- Block residency time
- Memory bandwidth utilization
- Scheduler stall time
- Request latency impact

### Optimization Opportunities
- Overlap multiple swaps
- Use pinned memory for faster transfers
- Implement swap prefetching
- Use compression for swap data

## Error Handling
- **Retry**: Retry failed swaps
- **Fallback**: Use recomputation as fallback
- **Graceful Degradation**: Continue with available resources

## Testing Strategy
- Unit tests for AsyncSwapManager
- Integration tests with scheduler
- Performance benchmarks
- Stress testing with high swap load

## Migration Strategy
- Implement alongside existing sync swapping
- Feature flag to enable/disable async swapping
- Gradual rollout with monitoring
- Rollback capability if issues arise 