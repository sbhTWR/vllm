# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
from abc import ABC, abstractmethod
import time
import random
from typing import Dict, List, Tuple, TYPE_CHECKING
from vllm.logger import init_logger
from vllm.sequence import ToolUsageHint

if TYPE_CHECKING:
    from vllm.config import PredictorConfig

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
    SWAP_RANDOM = enum.auto()
    SWAP_INFERCEPT = enum.auto()
    SWAP_MAV = enum.auto()
    SWAP_PRED = enum.auto()

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
                 last_accessed_by_user: str = None,
                 avg_tool_call_time: float = None,
                 latest_model_forward_time: float = None,
                 tool_hints: List[ToolUsageHint] = None):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.reuse_expected_time_s = reuse_expected_time_s
        self.last_accessed_by_user = last_accessed_by_user

        self.avg_tool_call_time = avg_tool_call_time
        self.latest_model_forward_time = latest_model_forward_time

        self.tool_hints = tool_hints
        self.cached_reuse_prob: float = None
        self.last_eval_walltime: float = 0.0 


class FreeBlockSwapScheduler:
    def __init__(self, swap_strategy = SwapStrategy.SWAP_LRU,
                 pinned_blocks_thresh: int = 0):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.swap_strategy = swap_strategy
        self.pinned_blocks_thresh = pinned_blocks_thresh
        logger.info("[evictor] [swap_strategy=%s] [pinned_blocks_thresh=%d]",
                    self.swap_strategy.name, self.pinned_blocks_thresh)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table


    def evict(self, cache_pin_ttl=None, force_evict=False) -> Tuple[int, BlockMetaData]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        if len(self.free_table) <= self.pinned_blocks_thresh and not force_evict:
            # if the number of free blocks is less than or equal to the pinned
            # blocks threshold, we do not evict any block.
            logger.info("[evictor] [pinned] No eviction, free blocks: %d, pinned blocks threshold: %d, force_evict: %s",
                        len(self.free_table), self.pinned_blocks_thresh, force_evict)
            logger.warning("[DEBUG] evict() returning None, None due to pinned blocks threshold")
            return None, None

        if self.swap_strategy == SwapStrategy.SWAP_RANDOM:
            block_id = random.choice(list(self.free_table.keys()))
            block_metadata = self.free_table[block_id]
            self.free_table.pop(block_id)
            logger.info("[evictor] [swap_random] evicting block_id=%d", block_id)
            return block_id, block_metadata


        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.

            # add to heap depending upon the strategy
            if self.swap_strategy == SwapStrategy.PERSIST or\
            self.swap_strategy == SwapStrategy.SWAP_LRU:
                last_accessed, num_hashed_tokens, block_id, content_hash = heapq.heappop(
                    self.priority_queue)
                if (block_id in self.free_table and
                        self.free_table[block_id].last_accessed == last_accessed):
                    
                    # delta = time.time() - last_accessed
                    # logger.info("[evictor] [lru] delta=%s cache_pin_ttl=%s"
                    #     % (delta, cache_pin_ttl))

                    # if cache_pin_ttl and (delta < cache_pin_ttl):
                    #     # restore the block in evictor 
                    #     heapq.heappush(
                    #         self.priority_queue,
                    #         (last_accessed, num_hashed_tokens, block_id, content_hash)
                    #     )
                    #     return None, None

                    block_metadata = self.free_table[block_id]
                    self.free_table.pop(block_id)
                    logger.info("[evictor] [freeblock_swap_scheduler] [swap_lru] evicting block_id=%d", block_id)
                    return block_id, block_metadata
                
            elif self.swap_strategy == SwapStrategy.SWAP_HINTS:
                reuse_expected_time, last_accessed, num_hashed_tokens, block_id, content_hash = heapq.heappop(
                    self.priority_queue
                )
                if (block_id in self.free_table and
                        self.free_table[block_id].last_accessed == last_accessed):
                    
                    agent_id = self.free_table[block_id].last_accessed_by_user
                    delta = abs(reuse_expected_time) - time.time()
                    # logger.info("[evictor] [hint] delta=%s cache_pin_ttl=%s agent_id=%s" 
                    #                         % (delta, cache_pin_ttl, agent_id))
                    if cache_pin_ttl\
                    and abs(reuse_expected_time) + 5 > time.time()\
                    and (delta < cache_pin_ttl):
                        
                        # persist 
                        # restore the block in evictor 
                        heapq.heappush(
                            self.priority_queue,
                            (reuse_expected_time, last_accessed, num_hashed_tokens, block_id, content_hash)
                        )
                        return None, None

                    block_metadata = self.free_table[block_id]
                    self.free_table.pop(block_id)
                    logger.info("[evictor] [swap_hints] evicting block_id=%d", block_id)
                    return block_id, block_metadata
            
            elif self.swap_strategy == SwapStrategy.SWAP_INFERCEPT:
                priority, last_accessed, num_hashed_tokens, block_id, content_hash = heapq.heappop(
                    self.priority_queue
                )
                if (block_id in self.free_table and
                        self.free_table[block_id].last_accessed == last_accessed):
                    
                    block_metadata = self.free_table[block_id]
                    agent_id = block_metadata.last_accessed_by_user
                    
                    # logger.info(f"[preserve_discard_evict] Evicting block_id={block_id}, "
                    #         f"priority={priority:.3f}, agent={agent_id}")
                    
                    logger.info("[evictor] [swap_infercept] evicting block_id=%d", block_id)
                    self.free_table.pop(block_id)
                    return block_id, block_metadata
            
            elif self.swap_strategy == SwapStrategy.SWAP_MAV:
                next_reuse_expected_time, last_accessed, num_hashed_tokens, block_id, content_hash = heapq.heappop(
                    self.priority_queue
                )
                if (block_id in self.free_table and
                        self.free_table[block_id].last_accessed == last_accessed):
                    
                    block_metadata = self.free_table[block_id]
                    agent_id = block_metadata.last_accessed_by_user
        
                    # Optional: Add logging
                    logger.info(f"[evictor] [swap_mav] Evicting block_id={block_id}, "
                            f"next_reuse_expected_time={next_reuse_expected_time:.3f}s, agent={agent_id}")

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
            last_accessed_by_user: str = None,
            avg_tool_call_time: float = None,           # NEW
            latest_model_forward_time: float = None,    # NEW
            num_blocks_in_hashless: int = None,
            tool_hints: List[ToolUsageHint] = None):        # NEW

        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  reuse_expected_time_s,
                                                  last_accessed_by_user,
                                                  avg_tool_call_time,
                                                  latest_model_forward_time)

        # logger.info("[elasticswap] adding block_id=%d to swap scheduler" % block_id)
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
        
        elif self.swap_strategy == SwapStrategy.SWAP_RANDOM:
            pass
        
        elif self.swap_strategy == SwapStrategy.SWAP_INFERCEPT:

            # preserve = Average_tool_call_time
            preserve = avg_tool_call_time if avg_tool_call_time is not None else 0.0
            
            # discard = current_model_forwarding_time * (1 + num_blocks_in_hashless_allocator)
            # Convert model_forward_time from milliseconds to seconds for consistency
            model_fwd_time_s = (latest_model_forward_time / 1000.0) if latest_model_forward_time is not None else 0.0
            num_hashless = num_blocks_in_hashless if num_blocks_in_hashless is not None else 0
            discard = model_fwd_time_s * (1 + num_hashless)
            
            # Priority = preserve - discard
            # Higher priority = more worth preserving = should evict LAST
            # We want LOW priority blocks to evict FIRST, so negate it
            priority = -(preserve - discard)
            
            # logger.info(f"[preserve_discard] block_id={block_id}: "
            #         f"preserve={preserve:.3f}s (avg_tool_time), "
            #         f"discard={discard:.3f}s (model_fwd={model_fwd_time_s:.3f}s * (1+{num_hashless})), "
            #         f"priority={priority:.3f}, "
            #         f"user={last_accessed_by_user}")
            
            heapq.heappush(
                self.priority_queue,
                (priority, last_accessed, -num_hashed_tokens, block_id, content_hash))
            self._cleanup_if_necessary()

        elif self.swap_strategy == SwapStrategy.SWAP_MAV:
            mav_time = avg_tool_call_time if avg_tool_call_time is not None else 0.0
            next_reuse_expected_time = last_accessed + mav_time
            heapq.heappush(
                self.priority_queue,
                (-next_reuse_expected_time, last_accessed, -num_hashed_tokens, block_id, content_hash))
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
            
            elif self.swap_strategy == SwapStrategy.SWAP_RANDOM:
                pass
            
            elif self.swap_strategy == SwapStrategy.SWAP_INFERCEPT:
                preserve = block.avg_tool_call_time if block.avg_tool_call_time is not None else 0.0
                model_fwd_time_s = (block.latest_model_forward_time / 1000.0) if block.latest_model_forward_time is not None else 0.0
                # Note: num_blocks_in_hashless needs to be fetched dynamically or stored
                # For cleanup, we can use a snapshot value or 0
                discard = model_fwd_time_s  # Simplified for cleanup
                priority = -(preserve - discard)
                
                new_priority_queue.append(
                    (priority, block.last_accessed, -block.num_hashed_tokens, block_id,
                    block.content_hash))
            
            elif self.swap_strategy == SwapStrategy.SWAP_MAV:
                avg_tool_call_time = block.avg_tool_call_time if block.avg_tool_call_time is not None else 0.0
                next_reuse_expected_time = block.last_accessed + avg_tool_call_time
                new_priority_queue.append(
                    (-next_reuse_expected_time, block.last_accessed, -block.num_hashed_tokens, block_id,
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
                logger.info("[evictor] [swap_lru] evicting block_id=%d", block_id)
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


class Predictor(ABC):
    @abstractmethod
    def expected_tool_duration(self, tool_name: str, tool_args: str,
                               last_access_duration_s: float) -> float:
        """
        Predicts expected time until reuse for the block associated with
        (tool_name, tool_args) given its age (seconds since last access).
        """

class TestPredictor(Predictor):
    def __init__(self, value: float = 30.0):
        self.value = value

    def expected_tool_duration(self, tool_name, tool_args, last_access_duration_s):
        return self.value

def make_predictor(predictor_config: "PredictorConfig") -> Predictor:
    if predictor_config.pred_type == "test":
        return TestPredictor(value=predictor_config.pred_params.get("value", 30.0))
    elif predictor_config.pred_type == "gittins":
        from vllm.core.gittins_predictor import GittinsPredictor
        return GittinsPredictor(model_path=predictor_config.pred_params.get("model_path", "gittins_model.pkl"))
    elif predictor_config.pred_type == "classifier":
        raise NotImplementedError("classifier predictor not implemented")
    else:
        raise ValueError(f"Unknown predictor type: {predictor_config.pred_type}")


class PredictiveEvictor(Evictor):
    def __init__(self, predictor: Predictor, score_ttl_s: float, heap_rebuild_interval_s: float = 10.0):
        self.predictor = predictor
        self.score_ttl_s = score_ttl_s
        self.free_table: dict[int, BlockMetaData] = {}
        self.heap: list[tuple[float, float, int, int]] = []  # (neg_expected, last_accessed, block_id, content_hash)
        self._last_rebuild = 0.0
        self.swap_strategy = SwapStrategy.SWAP_PRED
        self._score_cache: Dict[
            Tuple[str, str, str], Tuple[float, float]
        ] = {}

        self.heap_rebuild_interval_s = heap_rebuild_interval_s

    def __contains__(self, block_id):
        return block_id in self.free_table

    def add(self, block_id, content_hash, num_hashed_tokens, last_accessed,
            reuse_expected_time_s=None, last_accessed_by_user=None,
            avg_tool_call_time=None, latest_model_forward_time=None,
            num_blocks_in_hashless=None,
            tool_hints: List[ToolUsageHint] = None):
        metadata = BlockMetaData(content_hash, num_hashed_tokens,
                                 last_accessed, reuse_expected_time_s,
                                 last_accessed_by_user, avg_tool_call_time,
                                 latest_model_forward_time, tool_hints)
        metadata.cached_reuse_prob = None
        metadata.last_eval_walltime = 0.0
        self.free_table[block_id] = metadata
        score = self._score_block(metadata)
        heapq.heappush(self.heap, (-score, last_accessed, block_id, content_hash))

    def evict(self, cache_pin_ttl=None, force_evict=False):
        # logger.info("[predictive_evictor] evicting block")
        self._maybe_rebuild_heap()
        while self.heap:
            neg_score, last_accessed, block_id, content_hash = heapq.heappop(self.heap)
            if block_id not in self.free_table:
                continue
            block = self.free_table[block_id]
            if block.last_accessed != last_accessed:
                continue  # stale entry
            
            # logger.info(f"[predictive_evictor] evicting block_id={block_id} with score={neg_score}")
            self.free_table.pop(block_id)
            logger.info("[evictor] [predictive_evictor] evicting block_id=%d", block_id)
            return block_id, block

        # Heap drained but free_table still has entries: force rebuild once.
        # logger.info("[predictive_evictor] heap drained but free_table still has entries: force rebuild once.")
        if self.free_table:
            self._last_rebuild = 0.0
            self._maybe_rebuild_heap()
            while self.heap:
                neg_score, last_accessed, block_id, content_hash = heapq.heappop(self.heap)
                if block_id not in self.free_table:
                    continue
                block = self.free_table[block_id]
                if block.last_accessed != last_accessed:
                    continue
                self.free_table.pop(block_id)
                logger.info("[evictor] [predictive_evictor] evicting block_id=%d", block_id)
                return block_id, block

        logger.info("[predictive_evictor] no usable cache memory left")
        raise ValueError("No usable cache memory left")

    def update(self, block_id, last_accessed):
        block = self.free_table[block_id]
        block.last_accessed = last_accessed
        block.cached_reuse_prob = None  # invalidate

    def remove(self, block_id):
        self.free_table.pop(block_id, None)

    @property
    def num_blocks(self):
        return len(self.free_table)

    def _score_block(self, block: BlockMetaData) -> float:
        now = time.time()
        if block.cached_reuse_prob is not None and (now - block.last_eval_walltime) < self.score_ttl_s:
            return block.cached_reuse_prob
        if block.tool_hints is None or block.tool_hints == []:
            score = 999999999.0  # fallback
            # logger.info(f"[predictive_evictor] no tool hints found for block_id={block.last_accessed_by_user}")
        else:
            score = 0.0
            age = now - block.last_accessed
            agent_id = block.last_accessed_by_user or ""
            for hint in block.tool_hints:
                key = (agent_id, hint.tool_name, hint.tool_args)
                cached = self._score_cache.get(key)
                if cached and (now - cached[1]) < self.score_ttl_s:
                    expected_duration = cached[0]
                else:
                    expected_duration = self.predictor.expected_tool_duration(
                        hint.tool_name, hint.tool_args, age
                    )
                    self._score_cache[key] = (expected_duration, now)
                score += expected_duration

            # OLD uncached version 
            # age = now - block.last_accessed
            # for hint in block.tool_hints:
            #     expected_duration = self.predictor.expected_tool_duration(hint.tool_name, hint.tool_args, age)
            #     logger.info(f"[predictive_evictor] expected_duration={expected_duration} for tool_name={hint.tool_name} tool_args={hint.tool_args} age={age}")
            #     score += expected_duration


        block.cached_reuse_prob = score
        block.last_eval_walltime = now
        return score

    def _maybe_rebuild_heap(self):
        # logger.info("[predictive_evictor] maybe_rebuild_heap")
        now = time.time()
        if (now - self._last_rebuild) < self.heap_rebuild_interval_s:
            return
        # logger.info("[predictive_evictor] rebuilding heap")
        rebuilt = []
        for block_id, block in self.free_table.items():
            score = self._score_block(block)
            rebuilt.append((-score, block.last_accessed, block_id, block.content_hash))
        heapq.heapify(rebuilt)
        self.heap = rebuilt
        self._last_rebuild = now
        # logger.info("[predictive_evictor] heap rebuilt- took %s seconds" % (time.time() - now))
    
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
