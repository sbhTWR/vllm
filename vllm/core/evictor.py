# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from vllm.logger import init_logger

logger = init_logger(__name__)

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class SwapStrategy(enum.Enum):
    SWAP_LRU = enum.auto()
    PERSIST = enum.auto()
    SWAP_HINTS = enum.auto()

class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, reuse_expected_time_s: float = None,
                 last_accessed_by_user: str = None):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.reuse_expected_time_s = reuse_expected_time_s
        self.last_accessed_by_user = last_accessed_by_user


class FreeBlockSwapScheduler:
    def __init__(self, swap_strategy = SwapStrategy.SWAP_LRU):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.swap_strategy = swap_strategy

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table


    def evict(self) -> Tuple[int, BlockMetaData]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.

            # add to heap depending upon the strategy
            if self.swap_strategy == SwapStrategy.PERSIST or\
            self.swap_strategy == SwapStrategy.SWAP_LRU:
                last_accessed, _, block_id, content_hash = heapq.heappop(
                    self.priority_queue)
                if (block_id in self.free_table and
                        self.free_table[block_id].last_accessed == last_accessed):
                    
                    block_metadata = self.free_table[block_id]
                    self.free_table.pop(block_id)
                    return block_id, block_metadata
                
            elif self.swap_strategy == SwapStrategy.SWAP_HINTS:
                reuse_expected_time, last_accessed, _, block_id, content_hash = heapq.heappop(
                    self.priority_queue)
                if (block_id in self.free_table and
                        self.free_table[block_id].last_accessed == last_accessed):
                    block_metadata = self.free_table[block_id]
                    self.free_table.pop(block_id)
                    return block_id, block_metadata
                

        raise ValueError("No usable cache memory left")

    # def evict_n(self, n) -> List[int]:
    #     evicted_block_ids = []

    #     if len(self.free_table) == 0:
    #         return []

    #     while self.priority_queue:
    #         # We do not remove outdated entries from the priority queue at the
    #         # time of updating the last_accessed timestamp. Instead, outdated
    #         # entries are filtered out here during eviction. Outdated entries
    #         # would either not in the free table, or have older last accessed
    #         # time.

    #         if len(evicted_block_ids) >= n:
    #             break

    #         last_accessed, _, block_id, content_hash = heapq.heappop(
    #             self.priority_queue)
            
    #         if (block_id in self.free_table and
    #                 self.free_table[block_id].last_accessed == last_accessed):
    #             self.free_table.pop(block_id)
    #             evicted_block_ids.append((block_id, content_hash))
        
    #     return evicted_block_ids

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, reuse_expected_time_s: float = None, 
            last_accessed_by_user: str = None):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  reuse_expected_time_s,
                                                  last_accessed_by_user)

        logger.info("[elasticswap] adding block_id=%d to swap scheduler" % block_id)
        # add to heap depending upon the strategy
        if self.swap_strategy == SwapStrategy.PERSIST or\
        self.swap_strategy == SwapStrategy.SWAP_LRU:

            # use an LRU policy 
            heapq.heappush(
                self.priority_queue,
                (last_accessed, -num_hashed_tokens, block_id, content_hash))
            self._cleanup_if_necessary()
        
        elif self.swap_strategy == SwapStrategy.SWAP_HINTS:
            # use hint based scoring 
            
            reuse_expected_time = last_accessed
            assert reuse_expected_time_s is not None,\
                "reuse hint must be provided when using SWAP_HINTS"
            reuse_expected_time += reuse_expected_time_s

            heapq.heappush(
                self.priority_queue,
                (-reuse_expected_time, last_accessed, -num_hashed_tokens, block_id, content_hash))
            self._cleanup_if_necessary()

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():

            # add to heap depending upon the strategy
            if self.swap_strategy == SwapStrategy.PERSIST or\
            self.swap_strategy == SwapStrategy.SWAP_LRU:
                new_priority_queue.append(
                    (block.last_accessed, -block.num_hashed_tokens, block_id,
                    block.content_hash))
            
            elif self.swap_strategy == SwapStrategy.SWAP_HINTS:
                reuse_expected_time_s = block.reuse_expected_time_s
                assert reuse_expected_time_s is not None, "reuse hint must be provided when using SWAP_HINTS"
                reuse_expected_time = block.last_accessed + reuse_expected_time_s
                new_priority_queue.append(
                    (-reuse_expected_time, block.last_accessed, -block.num_hashed_tokens, block_id,
                    block.content_hash))

        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    def get_and_reset_swap_blocks(self):
        # if self.swap_strategy == SwapStrategy.SWAP_LRU:
        #     block_list= [(block_id, block) for block_id, block in self.free_table.items()]
        #     self.free_table = {}
        #     return block_list
        # elif self.swap_strategy == SwapStrategy.PERSIST:
        #     return []
        # else:
        #     raise ValueError("invalid swap strategy")

        """
        Removing this implementation in the favor of on-demand eviction 
        strategy.
        """
        return []

    
    @property
    def num_blocks(self) -> int:
        return len(self.free_table)
    

class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
