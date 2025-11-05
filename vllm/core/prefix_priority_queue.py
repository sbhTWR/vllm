
from collections import deque
from typing import Any, Optional, List
import math
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

class PrefixPriorityQueue:
    """
    Multi-level priority queue for vLLM scheduler.
    Assigns priority based on KV cache prefix match length.
    """
    
    def __init__(self, num_levels: int = 15, max_match_len: int = 200000, 
                 bucketing: str = "logarithmic"):
        """
        Args:
            num_levels: Number of priority levels (recommended: 10-15)
            max_match_len: Maximum expected prefix match length (e.g., 200k tokens)
            bucketing: Strategy to map match lengths to levels
                - "logarithmic": log-based (recommended for wide range)
                - "linear": equal-sized buckets
        """
        self.num_levels = num_levels
        self.max_match_len = max_match_len
        self.bucketing = bucketing
        
        # Each level is a deque (FIFO within same priority)
        self.queues: List[deque] = [deque() for _ in range(num_levels)]
        self.total_size = 0
        
        # For debugging/monitoring
        self.stats_enqueued_by_level = [0] * num_levels
        self._highest_priority_level = None  # Track first non-empty level
        
        # ADD PERFORMANCE METRICS
        self.metrics = {
            'append_count': 0,
            'append_time': 0.0,
            'popleft_count': 0,
            'popleft_time': 0.0,
            'getitem_count': 0,
            'getitem_time': 0.0,
            'getitem_0_count': 0,  # Special tracking for [0] access
            'getitem_0_time': 0.0,
            'appendleft_count': 0,
            'appendleft_time': 0.0,
            'extendleft_count': 0,
            'extendleft_time': 0.0,
            'remove_count': 0,
            'remove_time': 0.0,
            'iter_count': 0,
            'iter_time': 0.0,
            'update_level_count': 0,
            'update_level_time': 0.0,
        }
    
    def _match_length_to_level(self, match_length: int) -> int:
        """
        Map a match length to a priority level.
        
        Args:
            match_length: Number of matching prefix tokens (0 to max_match_len)
        
        Returns:
            Priority level (0 = lowest, num_levels-1 = highest)
        """
        if match_length == 0:
            return 0
        
        if self.bucketing == "logarithmic":
            # Logarithmic bucketing for wide range
            # Gives finer granularity to shorter matches
            log_val = math.log2(match_length + 1)
            max_log = math.log2(self.max_match_len + 1)
            level = int((log_val / max_log) * (self.num_levels - 1))
            return min(max(level, 0), self.num_levels - 1)
        
        elif self.bucketing == "linear":
            # Linear bucketing: divide range into equal chunks
            bucket_size = max(1, self.max_match_len // (self.num_levels - 1))
            level = min(match_length // bucket_size, self.num_levels - 1)
            return level
        
        else:
            # Default: simple scaling
            return min(match_length, self.num_levels - 1)
    
    def append(self, item: Any, match_length: Optional[int] = None):
        """
        Add item to queue based on prefix match length.
        
        Args:
            item: The SequenceGroup to enqueue
            match_length: Number of matching prefix tokens
                         If None, assumes no match (priority 0)
        """
        start_time = time.perf_counter()
        
        if match_length is None:
            match_length = 0
        
        level = self._match_length_to_level(match_length)
        self.queues[level].append(item)
        self.total_size += 1
        self.stats_enqueued_by_level[level] += 1
        
        # Update highest priority level
        if self._highest_priority_level is None or level > self._highest_priority_level:
            self._highest_priority_level = level
        
        # ADD METRICS
        self.metrics['append_count'] += 1
        self.metrics['append_time'] += time.perf_counter() - start_time
    
    def popleft(self) -> Any:
        """O(1) dequeue using cached level."""
        start_time = time.perf_counter()
        
        if self._highest_priority_level is None:
            raise IndexError("popleft from empty PrefixPriorityQueue")
        
        # Pop from the cached highest priority level
        item = self.queues[self._highest_priority_level].popleft()
        self.total_size -= 1
        
        # Update cache if this level is now empty
        if not self.queues[self._highest_priority_level]:
            self._update_highest_priority_level()
        
        # ADD METRICS
        self.metrics['popleft_count'] += 1
        self.metrics['popleft_time'] += time.perf_counter() - start_time
        
        return item
    
    def _update_highest_priority_level(self):
        """Find the new highest priority non-empty level."""
        start_time = time.perf_counter()
        
        for level in range(self.num_levels - 1, -1, -1):
            if self.queues[level]:
                self._highest_priority_level = level
                
                # ADD METRICS
                self.metrics['update_level_count'] += 1
                self.metrics['update_level_time'] += time.perf_counter() - start_time
                return
        
        self._highest_priority_level = None
        
        # ADD METRICS
        self.metrics['update_level_count'] += 1
        self.metrics['update_level_time'] += time.perf_counter() - start_time
    
    def __getitem__(self, index: int) -> Any:
        """
        Access item by index (for compatibility with deque interface).
        Index 0 = highest priority item.
        """
        start_time = time.perf_counter()
        
        if index < 0:
            raise IndexError("Negative indexing not supported")
        
        # Fast path for index 0 (most common case)
        if index == 0:
            if self._highest_priority_level is None:
                raise IndexError("Index 0 out of range (empty queue)")
            result = self.queues[self._highest_priority_level][0]
            
            # ADD METRICS
            self.metrics['getitem_0_count'] += 1
            self.metrics['getitem_0_time'] += time.perf_counter() - start_time
            self.metrics['getitem_count'] += 1
            self.metrics['getitem_time'] += time.perf_counter() - start_time
            return result
        
        count = 0
        # Iterate from highest to lowest priority
        for level in range(self.num_levels - 1, -1, -1):
            queue_len = len(self.queues[level])
            if count + queue_len > index:
                result = self.queues[level][index - count]
                
                # ADD METRICS
                self.metrics['getitem_count'] += 1
                self.metrics['getitem_time'] += time.perf_counter() - start_time
                return result
            count += queue_len
        
        raise IndexError(f"Index {index} out of range (size={self.total_size})")
    
    def __len__(self) -> int:
        return self.total_size
    
    def __bool__(self) -> bool:
        return self.total_size > 0
    
    def appendleft(self, item: Any):
        """
        Add to front (for compatibility - uses highest priority).
        Used when preempted requests need to be re-queued.
        """
        start_time = time.perf_counter()
        
        # When re-queueing preempted requests, give them high priority
        # Assume they already had some cache hit previously
        level = self.num_levels - 1  # Highest priority
        self.queues[level].appendleft(item)
        self.total_size += 1
        
        # Update highest priority level
        if self._highest_priority_level is None or level > self._highest_priority_level:
            self._highest_priority_level = level
        
        # ADD METRICS
        self.metrics['appendleft_count'] += 1
        self.metrics['appendleft_time'] += time.perf_counter() - start_time
    
    def extendleft(self, items):
        """
        Add multiple items to front (for compatibility).
        """
        start_time = time.perf_counter()
        
        for item in items:
            self.appendleft(item)
        
        # ADD METRICS (don't double-count appendleft metrics)
        self.metrics['extendleft_count'] += 1
        self.metrics['extendleft_time'] += time.perf_counter() - start_time
    
    def remove(self, item: Any):
        """
        Remove the first occurrence of item from the queue.
        
        Args:
            item: The item to remove
            
        Raises:
            ValueError: If item is not found in the queue
        """
        start_time = time.perf_counter()
        
        # Search through all levels from highest to lowest priority
        for level in range(self.num_levels - 1, -1, -1):
            try:
                self.queues[level].remove(item)
                self.total_size -= 1
                
                # Update highest priority level if we removed from it
                if level == self._highest_priority_level and not self.queues[level]:
                    self._update_highest_priority_level()
                
                # ADD METRICS
                self.metrics['remove_count'] += 1
                self.metrics['remove_time'] += time.perf_counter() - start_time
                return
            except ValueError:
                # Item not in this level, continue to next
                continue
        
        # If we get here, item was not found in any queue
        raise ValueError(f"Item not found in PrefixPriorityQueue")

    def clear(self):
        """Clear all queues."""
        for q in self.queues:
            q.clear()
        self.total_size = 0
        self._highest_priority_level = None
    
    def get_stats(self) -> dict:
        """Get queue statistics for monitoring/debugging."""
        return {
            "total_size": self.total_size,
            "by_level": {
                f"level_{i}": len(self.queues[i]) 
                for i in range(self.num_levels)
            },
            "enqueued_by_level": {
                f"level_{i}": self.stats_enqueued_by_level[i]
                for i in range(self.num_levels)
            }
        }
    
    def get_performance_metrics(self) -> dict:
        """
        Get detailed performance metrics for all operations.
        
        Returns:
            Dictionary with operation counts, total times, and average times.
        """
        metrics = {}
        for op in ['append', 'popleft', 'getitem', 'getitem_0', 'appendleft', 
                   'extendleft', 'remove', 'iter', 'update_level']:
            count = self.metrics[f'{op}_count']
            total_time = self.metrics[f'{op}_time']
            avg_time = (total_time / count * 1000) if count > 0 else 0  # Convert to ms
            
            metrics[op] = {
                'count': count,
                'total_time_s': total_time,
                'avg_time_ms': avg_time
            }
        
        return metrics
    
    def print_performance_metrics(self):
        """Log formatted performance metrics."""
        metrics = self.get_performance_metrics()
        
        logger.info("=" * 80)
        logger.info("PrefixPriorityQueue Performance Metrics")
        logger.info("=" * 80)
        logger.info(f"{'Operation':<20} {'Count':>12} {'Total Time':>15} {'Avg Time':>15}")
        logger.info("-" * 80)
        
        total_time_all_ops = 0.0
        total_calls_all_ops = 0
        
        for op, data in sorted(metrics.items(), key=lambda x: x[1]['total_time_s'], reverse=True):
            if data['count'] > 0:
                logger.info(f"{op:<20} {data['count']:>12} {data['total_time_s']:>12.3f}s {data['avg_time_ms']:>12.3f}ms")
                total_time_all_ops += data['total_time_s']
                total_calls_all_ops += data['count']
        
        logger.info("=" * 80)
        logger.info(f"{'TOTAL (all ops)':<20} {total_calls_all_ops:>12} {total_time_all_ops:>12.3f}s")
        logger.info("=" * 80)
        
        # Key highlights
        if metrics['getitem_0']['count'] > 0:
            logger.info("Key Insights:")
            logger.info(f"  - [0] access (most critical): {metrics['getitem_0']['avg_time_ms']:.4f}ms avg, "
                       f"{metrics['getitem_0']['total_time_s']:.3f}s total")
            logger.info(f"  - [0] access called {metrics['getitem_0']['count']} times")
            if total_time_all_ops > 0:
                pct_getitem_0 = (metrics['getitem_0']['total_time_s'] / total_time_all_ops) * 100
                logger.info(f"  - [0] access represents {pct_getitem_0:.1f}% of total queue operation time")
            if metrics['getitem']['count'] > metrics['getitem_0']['count']:
                other_getitem = metrics['getitem']['count'] - metrics['getitem_0']['count']
                logger.info(f"  - Other [index] access: {other_getitem} times")
    
    def reset_performance_metrics(self):
        """Reset all performance metrics to zero."""
        for key in self.metrics:
            self.metrics[key] = 0 if key.endswith('_count') else 0.0

    def __iter__(self):
        """
        Make PrefixPriorityQueue iterable for compatibility with sorted().
        Iterates from highest to lowest priority.
        """
        start_time = time.perf_counter()
        
        # Iterate from highest to lowest priority
        for level in range(self.num_levels - 1, -1, -1):
            for item in self.queues[level]:
                yield item
        
        # ADD METRICS
        self.metrics['iter_count'] += 1
        self.metrics['iter_time'] += time.perf_counter() - start_time


