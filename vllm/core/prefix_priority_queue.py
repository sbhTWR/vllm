
from collections import deque
from typing import Any, Optional, List
import math

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
        if match_length is None:
            match_length = 0
        
        level = self._match_length_to_level(match_length)
        self.queues[level].append(item)
        self.total_size += 1
        self.stats_enqueued_by_level[level] += 1
    
    def popleft(self) -> Any:
        """
        Dequeue from the highest priority non-empty queue.
        
        Returns:
            The next item
        
        Raises:
            IndexError: If queue is empty
        """
        # Start from highest priority level
        for level in range(self.num_levels - 1, -1, -1):
            if self.queues[level]:
                self.total_size -= 1
                return self.queues[level].popleft()
        
        raise IndexError("popleft from empty PrefixPriorityQueue")
    
    def __getitem__(self, index: int) -> Any:
        """
        Access item by index (for compatibility with deque interface).
        Index 0 = highest priority item.
        """
        if index < 0:
            raise IndexError("Negative indexing not supported")
        
        count = 0
        # Iterate from highest to lowest priority
        for level in range(self.num_levels - 1, -1, -1):
            queue_len = len(self.queues[level])
            if count + queue_len > index:
                return self.queues[level][index - count]
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
        # When re-queueing preempted requests, give them high priority
        # Assume they already had some cache hit previously
        level = self.num_levels - 1  # Highest priority
        self.queues[level].appendleft(item)
        self.total_size += 1
    
    def extendleft(self, items):
        """
        Add multiple items to front (for compatibility).
        """
        for item in items:
            self.appendleft(item)
    
    def clear(self):
        """Clear all queues."""
        for q in self.queues:
            q.clear()
        self.total_size = 0
    
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

