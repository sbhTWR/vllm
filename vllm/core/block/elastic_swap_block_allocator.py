"""This file implement a block allocator that supports CPU KV cache offloading

The key idea of this implementation is to maintain those allocated blocks 
that didn't hit the cache, and constantly copy them into CPU after each 
scheduler step.

This idea is borrowed from ConServe
(paper link: https://arxiv.org/abs/2410.01228), based on the assumption 
that the CPU-GPU bandwidth is much higher than GPU KV cache generation 
throughput. Thanks Yifan for this idea.

This implementation also allows vLLM to gracefully handle preemption by 
recomputation.
"""
from collections import deque
import heapq
import sys
import time
from typing import Deque, Dict, List, Optional, Tuple

from vllm.core.evictor import SwapStrategy, FreeBlockSwapScheduler
from vllm.core.evictor import BlockMetaData
from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId,
                                        DeviceAwareBlockAllocator)
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator, ElasticSwapBlockAllocator, BlockTracker, assert_prefix_caching_block_or_none
from vllm.config import SwapBudgetType
from vllm.sequence import SequenceGroup
from vllm.utils import Device
from vllm.logger import init_logger

logger = init_logger(__name__)

class AllocationContext:
    def __init__(self):
        self.seq_group: SequenceGroup = None

    def set_context(self, seq_group: SequenceGroup):
        self.seq_group = seq_group
    
    def unset_context(self):
        self.seq_group = None

class CpuOffloadingBlockAllocator(CpuGpuBlockAllocator):
    """A block allocator that supports CPU KV cache offloading

    This class extends the `CpuGpuBlockAllocator` so that the CPU can be used 
    for prefix caching.
    
    It will internally maintain uncached blocks, and trying to copy uncached
    blocks into CPU upon the end of scheduler step (i.e. calling 
    `get_and_reset_swaps`).

    This implementation also allows vLLM to gracefully handle preemption by 
    recomputation.
    """

    allocators: Dict[Device, PrefixCachingBlockAllocator]

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        swap_strategy: SwapStrategy = SwapStrategy.SWAP_LRU,
        enable_swap_budget: bool = False,
        swap_budget_type: SwapBudgetType = SwapBudgetType.FIXED,
        swap_budget_frac: float = 0.5,
    ) -> DeviceAwareBlockAllocator:
        """Initiate CpuOffloadingBlockAllocator. Similar to 
        CpuGpuBlockAllocator.create() but only support prefix caching

        Args:
            allocator_type (str): The type of block allocator to use for CPU
                and GPU blocks. Currently supported values are "naive" and
                "prefix_caching".
            num_gpu_blocks (int): The number of blocks to allocate for GPU
                memory.
            num_cpu_blocks (int): The number of blocks to allocate for CPU
                memory.
            block_size (int): The size of each block in number of tokens.

        Returns:
            DeviceAwareBlockAllocator: A CpuOffloadingBlockAllocator instance 
                with the specified configuration.

        Notes:
            - The block IDs are assigned contiguously, with GPU block IDs coming
                before CPU block IDs.
        """
        # assert num_gpu_blocks < num_cpu_blocks, "CPU offloading block "\
        #     "allocator requires the allocated CPU memory capacity to be larger"\
        #     " than GPU memory capacity."
        block_ids = list(range(num_gpu_blocks + num_cpu_blocks))
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        assert allocator_type == "prefix_caching", "CpuOffloadingBlock"\
            "Allocator should be only used together with prefix caching."

        # prefix caching block is now the default.
        gpu_allocator = ElasticSwapBlockAllocator(
            num_blocks=num_gpu_blocks,
            block_size=block_size,
            block_ids=gpu_block_ids,
            swap_strategy=swap_strategy,
            enable_swap_budget=enable_swap_budget,
            swap_budget_type=swap_budget_type,
            swap_budget_frac=swap_budget_frac
        )

        cpu_allocator: BlockAllocator = NaiveBlockAllocator(
                create_block=NaiveBlock,  # type: ignore
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )

        return CpuOffloadingBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )

    def __init__(self, cpu_block_allocator: PrefixCachingBlockAllocator,
                 gpu_block_allocator: PrefixCachingBlockAllocator):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        super().__init__(cpu_block_allocator, gpu_block_allocator)
        self._allocators: Dict[Device,
                               PrefixCachingBlockAllocator] = {  # type: ignore
                                   Device.CPU: cpu_block_allocator,
                                   Device.GPU: gpu_block_allocator
                               }

        self.swap_strategy = gpu_block_allocator.swap_scheduler.swap_strategy
        self.num_gpu_blocks = gpu_block_allocator.get_num_total_blocks()
        self.num_cpu_blocks = cpu_block_allocator.get_num_total_blocks()

        self._swap_mapping: Dict[int, int] = {}

        self._block_ids_to_allocator: Dict[int, BlockAllocator] = {}
        for _, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator


        self._cached_blocks_cpu: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

        self.allocation_ctx = AllocationContext()

    def memory_pressure_evict(self, n):
        # if self._allocators[Device.GPU].swap_scheduler.swap_strategy == SwapStrategy.PERSIST:
        #     self.evict_from_swap_scheduler(n)
        # else:
        #     self.evict_from_cpu_swap_evictor(n)

        return None

    def add_to_cpu_swap_evictor(self, block_id, block_metadata: BlockMetaData):
        self._cached_blocks_cpu[block_id] = block_metadata
        reuse_expected_time_s = block_metadata.reuse_expected_time_s
        last_accessed = block_metadata.last_accessed
        num_hashed_tokens = block_metadata.num_hashed_tokens
        content_hash = block_metadata.content_hash

        if self.swap_strategy == SwapStrategy.PERSIST or\
            self.swap_strategy == SwapStrategy.SWAP_LRU:
            heapq.heappush(
                self.priority_queue,
                (last_accessed, -num_hashed_tokens, block_id, content_hash))
        
        elif self.swap_strategy == SwapStrategy.SWAP_HINTS:
            reuse_expected_time = last_accessed + reuse_expected_time_s
            heapq.heappush(
                self.priority_queue,
                (-reuse_expected_time, last_accessed, -num_hashed_tokens, block_id, content_hash))
        
        self._cleanup_cpu_swap_evictor_if_necessary()
    
    def remove_from_cpu_swap_evictor(self, block_id):
        self._cached_blocks_cpu.pop(block_id)
    
    def _cleanup_cpu_swap_evictor_if_necessary(self):
        if len(self.priority_queue) > 50 * len(
                self._cached_blocks_cpu):
            self._cleanup_cpu_swap_evictor()

    def _cleanup_cpu_swap_evictor(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self._cached_blocks_cpu.items():
            if self.swap_strategy == SwapStrategy.PERSIST or\
                self.swap_strategy == SwapStrategy.SWAP_LRU:
                new_priority_queue.append(
                    (block.last_accessed, -block.num_hashed_tokens, block_id,
                    block.content_hash))
            
            elif self.swap_strategy == SwapStrategy.SWAP_HINTS:
                reuse_expected_time = block.last_accessed + block.reuse_expected_time_s
                heapq.heappush(
                    self.priority_queue,
                    (-reuse_expected_time, block.last_accessed, -block.num_hashed_tokens, 
                     block_id, block.content_hash))

        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    def evict_from_swap_scheduler(self, n):
        num_evicted = self._allocators[Device.GPU].evict_n_from_swap_scheduler(n)
        return num_evicted

    def evict_from_cpu_swap_evictor(self, n):
        num_evicted = 0
        
        if len(self._cached_blocks_cpu) == 0:
            return None
        
        gpu_allocator = self._allocators[Device.GPU]
        block_tracker = gpu_allocator._block_tracker
        while self.priority_queue:
            if num_evicted >= n:
                break

            # for block_id, block in self._cached_blocks_cpu.items():
            if self.swap_strategy == SwapStrategy.PERSIST or\
                self.swap_strategy == SwapStrategy.SWAP_LRU:
                last_accessed, _, block_id, content_hash = heapq.heappop(
                    self.priority_queue)
            elif self.swap_strategy == SwapStrategy.SWAP_HINTS:
                    _, last_accessed, _, block_id, content_hash = heapq.heappop(
                    self.priority_queue)

            if (block_id in self._cached_blocks_cpu and 
                last_accessed == self._cached_blocks_cpu[block_id].last_accessed):
                self._cached_blocks_cpu.pop(block_id)
                # print('popped block_id=%d' % block_id)

                assert content_hash in gpu_allocator._cached_blocks, "content_hash=%s" % content_hash


                # evict content_hash only if not swapped-in into gpu 
                tmp_block_id = gpu_allocator._cached_blocks[content_hash]

                if not self._is_gpu_block_unsafe(tmp_block_id):
                    # logger.info("[elasticswap] heirarchical_cache: evicting content_hash=%s" 
                    #                                             % content_hash)
                    
                    
                    # replace in hash map 
                    gpu_allocator._cached_blocks.pop(content_hash)
                    # remove from block tracker 
                    if block_id in block_tracker:
                        block_metadata = block_tracker[block_id]
                        # print('evicting %s' % block_metadata.last_accessed_by_user)
                        block_tracker.pop(block_id)
                    self._allocators[Device.CPU]._free_block_id(block_id)

                    # free the block 
                    num_evicted += 1
                # else:
                #     logger.info("[elasticswap] heirarchical_cache: NOT evicting content_hash=%s as it is now in GPU cache" 
                #                                                 % content_hash)
        
        # perform a cleanup to ensure no stale blocks are remaining in heap evictor.
        # self._cleanup_cpu_swap_evictor()
        return num_evicted

    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               device: Device,
                               extra_hash: Optional[int] = None) -> Block:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated mutable block.
        """
        # assert device == Device.GPU, "Calls to CPU offloading block allocator "\
        #     "should always use Device.GPU --- CPU offloading block allocator "\
        #     "handles CPU offloading internally."\
        # # mark this block as uncached

        # block = self._allocators[device].allocate_mutable_block(
        #     prev_block, extra_hash=extra_hash)
        # return block

        block = self.allocate_mutable_block_gpu_with_heirarchical_cache(
            prev_block=prev_block,
            device=device,
            extra_hash=extra_hash
        )

        # get allocation context to see if hints are there 
        if self.swap_strategy == SwapStrategy.SWAP_HINTS:
            hints = self.allocation_ctx.seq_group.hints
            assert hints is not None, "hints must be provided with SWAP_HINTS"
            block.reuse_expected_time_s = hints.kv_reuse_expected_duration_s
        
        block.last_accessed_by_user = self.allocation_ctx.seq_group.user_id

        return block
    
    def allocate_mutable_block_gpu_with_heirarchical_cache(self,
                               prev_block: Optional[Block],
                               device: Device,
                               extra_hash: Optional[int] = None,
                               return_block_id = False) -> Block:

        assert device == Device.GPU, "Calls to CPU offloading block allocator "\
            "should always use Device.GPU --- CPU offloading block allocator "\
            "handles CPU offloading internally."\
        
        gpu_allocator: ElasticSwapBlockAllocator = self._allocators[Device.GPU]
        cpu_allocator: NaiveBlockAllocator = self._allocators[Device.CPU]
        swap_scheduler: FreeBlockSwapScheduler = gpu_allocator.swap_scheduler
        # 1. First try hashless allocator 
        assert_prefix_caching_block_or_none(prev_block)
        hashless_block_id = gpu_allocator._maybe_allocate_hashless_block_id()
        if hashless_block_id is not None:
            if return_block_id:
                return hashless_block_id
            block = gpu_allocator._block_pool.init_block(prev_block=prev_block,
                                            token_ids=[],
                                            block_size=gpu_allocator._block_size,
                                            physical_block_id=hashless_block_id,
                                            extra_hash=extra_hash)
            assert not block.computed
            assert block.content_hash is None
            return block

        # 2. If not, then evict from swap scheduler 
            # if not, throw no free block error 
        
        gpu_block_id = None 
        block_metadata = None
        if swap_scheduler.num_blocks > 0:
            gpu_block_id, block_metadata = swap_scheduler.evict()
            # logger.info(">>>> gpu_block_id=%d evicted from swap scheduler. user_id=%s" 
            #             % (gpu_block_id, block_metadata.last_accessed_by_user))
        else:
            raise BlockAllocator.NoFreeBlocksError()

        # 2.a -- perform book-keeping on the block
        content_hash_to_evict = block_metadata.content_hash
        # Sanity checks
        assert content_hash_to_evict in gpu_allocator._cached_blocks
        _block_id = gpu_allocator._cached_blocks[content_hash_to_evict]
        assert gpu_allocator._refcounter.get(_block_id) == 0
        assert _block_id == gpu_block_id

        # logger.info("[elasticswap] evicting content_hash=%s" 
        #             % content_hash_to_evict)
        gpu_allocator._cached_blocks.pop(content_hash_to_evict)

        gpu_allocator._refcounter.incr(gpu_block_id)
        gpu_allocator._track_block_id(gpu_block_id, computed=False)


        # 3. check if there is space in CPU cache for swapping out 
            # if not, evict from cpu 

        if cpu_allocator.get_num_free_blocks() == 0:
            # evict cpu block
            num_evicted = self.evict_from_cpu_swap_evictor(1)
            assert num_evicted >= 1, "num_evicted=%s" % num_evicted
            assert cpu_allocator.get_num_free_blocks() > 0
        # 4. then swap to cpu 
        # allocate a block from cpu 
        cpu_block_id = cpu_allocator._allocate_block_id_unsafe()
        assert cpu_block_id is not None

        now = time.time()
        # init the swap process 
        self.replace_block_ids_in_cached_allocator(
                gpu_block_id,
                cpu_block_id,
                block_metadata.content_hash,
                now,
                block_metadata,
                untrack_old=False
            )

        # schedule swap
        self._swap_mapping[gpu_block_id] = cpu_block_id

        if return_block_id:
            return gpu_block_id

        # allocate a block object 
        block = gpu_allocator._block_pool.init_block(prev_block=prev_block,
                                            token_ids=[],
                                            block_size=gpu_allocator._block_size,
                                            physical_block_id=gpu_block_id,
                                            extra_hash=extra_hash)
        assert not block.computed
        assert block.content_hash is None
        return block

    def num_blocks_cached_for_token_ids(
                self,
                prev_block: Optional[Block],
                block_token_ids: List[List[int]],
                extra_hash: Optional[int] = None) -> int:
        
        num_blocks_cached = 0
        block_ids_cached = self._allocators[Device.GPU].blocks_cached_for_token_ids(
            prev_block=prev_block,
            block_token_ids=block_token_ids,
            extra_hash=extra_hash
        )

        for block_id in block_ids_cached:
            if self._is_gpu_block_unsafe(block_id):
                num_blocks_cached += 1
        
        return num_blocks_cached


    def allocate_immutable_blocks(
            self,
            prev_block: Optional[Block],
            block_token_ids: List[List[int]],
            device: Device,
            extra_hash: Optional[int] = None) -> List[Block]:
        """Allocates a new group of immutable blocks with the provided block 
        token IDs on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            block_token_ids (List[int]): The list of block token IDs to be 
                stored in the new blocks.
            device (Device): The device on which to allocate the new block.

        Returns:
            List[Block]: The newly allocated list of immutable blocks 
                containing the provided block token IDs.
        """
        assert device == Device.GPU, "Calls to CPU offloading block allocator "\
            "should always use Device.GPU --- CPU offloading block allocator"\
            "handles CPU offloading internally."

        # repeatedly call allocate_immutable_block
        # because it handles CPU-GPU offloading related logics.
        blocks = []
        for token_ids in block_token_ids:
            prev_block = self.allocate_immutable_block(prev_block=prev_block,
                                                       token_ids=token_ids,
                                                       device=device,
                                                       extra_hash=extra_hash)
            blocks.append(prev_block)
        return blocks

    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 device: Device,
                                 extra_hash: Optional[int] = None) -> Block:
        """Allocates a new immutable block with the provided token IDs on the
        specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            token_ids (List[int]): The list of token IDs to be stored in the new
                block.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated immutable block containing the provided
                token IDs.
        """

        assert device == Device.GPU, "Calls to CPU offloading block allocator"\
            " should always use Device.GPU --- CPU offloading block allocator"\
            " handles CPU offloading internally."

        # check if hash exists and whether its a CPU block 
        _block_id_tmp, content_hash = self._allocators[device].check_if_hash_exists(
            prev_block=prev_block,
            token_ids=token_ids,
            extra_hash=extra_hash
        )

        replacement_was_needed = False
        reuse = True
        if _block_id_tmp:
            assert content_hash is not None
            if not self._is_gpu_block_unsafe(_block_id_tmp):

                # TODO: do we need to lock the cpu block?

                block_metadata = self._allocators[device]._block_tracker[_block_id_tmp]
                replacement_was_needed = True
                # allocate a gpu block 
                # gpu_block_id_replacement = self._allocators[device]._allocate_block_id() 
                gpu_block_id_replacement = self.allocate_mutable_block_gpu_with_heirarchical_cache(
                    prev_block=None,
                    device=device,
                    extra_hash=None,
                    return_block_id=True
                )
                # and replace the mapping

                # print('[allocate_immutable] cpu (%d) -> gpu (%d)' % (_block_id_tmp, gpu_block_id_replacement))

                # decrement the reference count since it will be incremented 
                # in immutable allocation 
                self._allocators[device]._refcounter.decr(gpu_block_id_replacement)
                self.replace_block_ids_in_cached_allocator(
                    _block_id_tmp,
                    gpu_block_id_replacement,
                    content_hash,
                    now=False,
                    block_metadata=block_metadata
                )

                self._swap_mapping[_block_id_tmp] = gpu_block_id_replacement

        # allocate a GPU block
        block = self._allocators[device].allocate_immutable_block(
            prev_block, token_ids, extra_hash=extra_hash)
        
        if not block:
            block = self.allocate_mutable_block_gpu_with_heirarchical_cache(
                prev_block, Device.GPU, extra_hash=extra_hash)
            
            block.append_token_ids(token_ids)
            reuse = False

        if replacement_was_needed:
            assert block.block_id == gpu_block_id_replacement

        block_id = block.block_id
        assert block_id is not None
        block_computed = self._allocators[device].block_is_computed(block_id)

        if replacement_was_needed:
            assert block_computed == True
        
        if self.swap_strategy == SwapStrategy.SWAP_HINTS:
            # update hints 
            hints = self.allocation_ctx.seq_group.hints
            user_id = self.allocation_ctx.seq_group.user_id
            assert hints is not None, "hints must be initialized when using SWAP_HINTS"
            assert user_id is not None, "user_id must be provided when using SWAP_HINTS"
            block_tracker_obj = self._allocators[device]._block_tracker[block_id]

            if reuse:
                # update 
                old_reuse_time_s = block_tracker_obj.reuse_expected_time_s
                old_user_id = block_tracker_obj.last_accessed_by_user
                new_reuse_time_s = hints.kv_reuse_expected_duration_s
                new_user_id = user_id

                last_access_time = block_tracker_obj.last_accessed

                delta = time.time() - last_access_time

                # logger.info("[elasticswap_debug] old_reuse_time_s=%f old_user_id=%s reuse=True replacement_was_needed=%s" % 
                #             (old_reuse_time_s if old_reuse_time_s else 0.0, old_user_id, replacement_was_needed))

                if (delta >= old_reuse_time_s):
                    # reset 
                    block_tracker_obj.reuse_expected_time_s = new_reuse_time_s
                    
                elif old_user_id != new_user_id:
                    block_tracker_obj.reuse_expected_time_s = min(new_reuse_time_s, old_reuse_time_s - delta)

                block_tracker_obj.last_accessed_by_user = new_user_id

                assert block_tracker_obj.reuse_expected_time_s >= 0.0
            else:
                # assign 
                block_tracker_obj.reuse_expected_time_s = hints.kv_reuse_expected_duration_s
                # logger.info("[elasticswap] block_id=%d hint<kv_reuse_expected_duration_s>=%f" %
                #             (block_id, block_tracker_obj.reuse_expected_time_s))
                block_tracker_obj.last_accessed_by_user = user_id

        # print('block_id (%d) -> %s' % (block_id, self.allocation_ctx.seq_group.user_id))

        # deal with prefix caching, three cases in total:
        # 1. cache hit on GPU
        # 2. no cache hit on GPU but cache hit on CPU
        # 3. no cache hit
        # if block_computed:
        #     # cache hit on GPU, no need to put it into uncached blocks
        #     if not self._is_gpu_block_unsafe(block.block_id):
        #         # cpu block id, needs swap_in 
        #         # allocate a block_id from cpu 
        #         gpu_block_id = self._allocators[device]._allocate_block_id()
        #         self._swap_mapping[block.block_id] = gpu_block_id

        #         # replace block_ids
        #         self.replace_block_ids_in_cached_allocator(
        #             block.block_id,
        #             gpu_block_id,
        #             block.content_hash,
        #             now=False
        #         )

        #         block.block_id = gpu_block_id
        # else:
        #     # check if we can hit cache on CPU by trying to allocate CPU block
        #     cpu_block = self._allocators[Device.CPU].allocate_immutable_block(
        #         prev_block, token_ids, extra_hash=extra_hash)
        #     cpu_block_id = cpu_block.block_id
        #     assert cpu_block_id is not None
        #     cpu_block_computed = self._allocators[
        #         Device.CPU].block_is_computed(cpu_block_id)
        #     if cpu_block_computed:
        #         # CPU cache hit
        #         # mark the GPU block as computed
        #         self._allocators[Device.GPU].mark_blocks_as_computed(
        #             [block_id])
        #         # copy the CPU cache to GPU
        #         self._swap_mapping[cpu_block_id] = block_id
        #         # and don't free this block until `get_and_reset_swap` is called
        #         self._allocated_cpu_blocks.append(cpu_block)
        #     else:
        #         # No cache hit
        #         # mark the GPU block as uncached
        #         self._uncached_blocks.append(block)
        #         # and free cpu block
        #         self._allocators[Device.CPU].free(cpu_block)

        return block

    def swap(self, blocks: List[Block], src_device: Device,
             dst_device: Device) -> Dict[int, int]:

        raise NotImplementedError("CPU offloading block allocator only "
                                  "support preemption by recomputation.")

    def _is_gpu_block(self, block_id: int) -> bool:
        return block_id in self._allocators[Device.GPU].all_block_ids

    def _is_gpu_block_unsafe(self, block_id: int) -> bool:
        """Faster version of `_is_gpu_block` that doesn't check the block ID.
        But assumes the that the block IDs are assigned contiguously, with GPU 
        block IDs coming before the CPU block IDs.
        """
        return block_id < self.num_gpu_blocks

    def _get_physical_block_id_unsafe(self, block_id: int) -> int:
        """Returns the physical block ID of the given block ID.

        This function avoids using the `allocator.get_physical_block_id()`
        which is slow (O(NlogN)). Instead, this is based on the assumption
        that the block IDs are assigned contiguously, with GPU block IDs coming
        before CPU block IDs.

        Args:
            block_id (int): The block ID to get the physical block ID of.

        Returns:
            int: The physical block ID of the given block ID.

        Note:
            Please see the implementation of 
            `CpuOffloadingBlockAllocator.create` for how the block IDs are
            assigned.
        """
        if self._is_gpu_block_unsafe(block_id):
            return block_id
        else:
            return block_id - self.num_gpu_blocks

    def replace_block_ids_in_cached_allocator(
            self, old_block_id, new_block_id, hash_value, now, block_metadata=None, 
            untrack_old=True):
        
        # logger.info("[elasticswap] src (%d)")

        gpu_allocator = self._allocators[Device.GPU]

        # replace in hash map
        gpu_allocator._cached_blocks[hash_value] = new_block_id
        
        # replace in block tracker
        block_tracker = gpu_allocator._block_tracker
        assert old_block_id in block_tracker

        old_block_tracker_obj = block_tracker[old_block_id]

        # logger.info("replacing old_block_id=%d --> new_block_id=%d old block is active = %s user_id=%s hash_value=%s" % 
        #       (old_block_id, new_block_id,
        #        old_block_tracker_obj.active,
        #        old_block_tracker_obj.last_accessed_by_user,
        #        hash_value))
        
        now = time.time()
        last_accessed_diff = None 
        kv_reuse_time_diff = None 
        if block_metadata:
            last_accessed_diff = now - block_metadata.last_accessed
            if block_metadata.reuse_expected_time_s:
                kv_reuse_time_diff = block_metadata.reuse_expected_time_s - last_accessed_diff
            
        else:
            last_accessed_diff = now - old_block_tracker_obj.last_accessed
            if old_block_tracker_obj.reuse_expected_time_s:
                kv_reuse_time_diff = old_block_tracker_obj.reuse_expected_time_s - last_accessed_diff
            
        caller = sys._getframe(1).f_code.co_name
        
        if self._is_gpu_block_unsafe(old_block_id):
            logger.info("[caller=%s] swapping_out old_block_id=%d new_block_id=%d user_id=%s hash_value=%s last_accessed_diff=%s kv_reuse_time_diff=%s" % 
                (caller, old_block_id, new_block_id,
                old_block_tracker_obj.last_accessed_by_user,
                hash_value,
                last_accessed_diff,
                kv_reuse_time_diff
                ))
        else:
            logger.info("[caller=%s] swapping_in old_block_id=%d new_block_id=%d user_id=%s hash_value=%s last_accessed_diff=%s kv_reuse_time_diff=%s" % 
                (caller, old_block_id, new_block_id,
                old_block_tracker_obj.last_accessed_by_user,
                hash_value,
                last_accessed_diff,
                kv_reuse_time_diff
                ))
        

        # if block_metadata:
        #     print('last_accessed_by_user=%s' % block_metadata.last_accessed_by_user)
        
        new_block_tracker_obj = BlockTracker()
        new_block_tracker_obj.computed = True
        new_block_tracker_obj.active = False
        # print("new block_id active %s" % new_block_tracker_obj.active)

        if block_metadata:
            new_block_tracker_obj.last_accessed_by_user = block_metadata.last_accessed_by_user
            new_block_tracker_obj.reuse_expected_time_s = block_metadata.reuse_expected_time_s
        else:
            new_block_tracker_obj.last_accessed_by_user = old_block_tracker_obj.last_accessed_by_user
            new_block_tracker_obj.reuse_expected_time_s = old_block_tracker_obj.reuse_expected_time_s

        if now:
            new_block_tracker_obj.last_accessed = now
        elif block_metadata:
            new_block_tracker_obj.last_accessed = block_metadata.last_accessed
        else:
            new_block_tracker_obj.last_accessed = old_block_tracker_obj.last_accessed

        block_tracker[new_block_id] = new_block_tracker_obj
        
        # logger.info("user_id=%s" % block_metadata.last_accessed_by_user)
        # logger.info("replaced old_block_id=%d --> new_block_id=%d new block is active = %s user_id=%s" % 
            #   (old_block_id, new_block_id,
            #    new_block_tracker_obj.active, 
            #    new_block_tracker_obj.last_accessed_by_user))
        
        # print(old_block_id)
        if old_block_tracker_obj.active and untrack_old:
            block_tracker[old_block_id].disable()
        
        if self._is_gpu_block_unsafe(old_block_id):
            assert block_metadata is not None
            self.add_to_cpu_swap_evictor(new_block_id, block_metadata)
        # else:
        #     # the old block is a cpu block
        #     # invalidate any entries in cpu swap evictor 
        #     if old_block_id in self._cached_blocks_cpu:
        #         self._cached_blocks_cpu.pop(old_block_id)
        #         self._cleanup_cpu_swap_evictor()

        

    def free(self, block: Block) -> None:
        """Frees the memory occupied by the given block.

        Args:
            block (Block): The block to be freed.
        """
        # # Null block should never be freed
        # if isinstance(block, NullBlock):
        #     return
        block_id = block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        allocator.free(block)


    def get_and_reset_swaps(self,
                            now: float) -> Tuple[List[Tuple[int, int]], ...]:
        """Returns and clears the mapping of source to destination block IDs.
        Will be called right before scheduler step finishes.
        
        This function will do the following things:
            1. Iterate over uncached blocks and see if we can copy it to CPU
            2. Update all allocated CPU block time stamp
            3. Free CPU blocks
            4. Return and clear all swapping status
            
        Args:
            now (float): The time stamp used to update CPU access time, so 
            that CPU evictor can work.
        
        Returns:
            A tuple of two lists: (blocks_to_swap_out, blocks_to_swap_in).
            Each list is a List[Tuple[int, int]], containing the mapping of 
            source to destination block IDs. The block IDs are physical block
            IDs and it's expected to be used by the cache engine directly.
        """

        swap_scheduler = self._allocators[Device.GPU].swap_scheduler
        num_blocks = swap_scheduler.num_blocks
        # logger.info("[elasticswap] blocks in swap_scheduler=%d" % num_blocks)


        blocks_to_swap_out = []
        blocks_to_swap_in = []

        gpu_allocator = self._allocators[Device.GPU]
        cpu_allocator = self._allocators[Device.CPU]

        gpu_blocks_to_swap_out = gpu_allocator.get_and_reset_swaps()

        while gpu_blocks_to_swap_out:
            gpu_block_id, block_metadata = gpu_blocks_to_swap_out.pop()
            cpu_block_id = cpu_allocator._allocate_block_id_unsafe()

            if not cpu_block_id:
                gpu_blocks_to_swap_out.append((gpu_block_id, block_metadata))
                break

            hash_value = block_metadata.content_hash
            
            self.replace_block_ids_in_cached_allocator(
                gpu_block_id,
                cpu_block_id,
                hash_value,
                now,
                block_metadata
            )

            # increment reference counter and release block to hashless allocator
            gpu_allocator._hashless_allocator._refcounter.incr(gpu_block_id)
            gpu_allocator._hashless_allocator._free_block_id(gpu_block_id)

            # print('swap_out gpu (%d) --> cpu (%d)' % (gpu_block_id, cpu_block_id))
        
            src = self._get_physical_block_id_unsafe(gpu_block_id)
            dst = self._get_physical_block_id_unsafe(cpu_block_id)
            blocks_to_swap_out.append((src, dst))

        if gpu_blocks_to_swap_out:
            # print('releasing some blocks back to swap_scheduler=%d' % len(gpu_blocks_to_swap_out))
            while gpu_blocks_to_swap_out:
                block_id, block_metadata = gpu_blocks_to_swap_out.pop()
                # add them back to swap scheduler 
                gpu_allocator.add_to_swap_scheduler(block_id, block_metadata)

        for src, dst in self._swap_mapping.items():
            # only two possible cases: CPU -> GPU, or GPU -> CPU
            #if src in self._allocators[Device.GPU].all_block_ids:
            if self._is_gpu_block_unsafe(src):
                # swap out
                src = self._get_physical_block_id_unsafe(src)
                dst = self._get_physical_block_id_unsafe(dst)
                blocks_to_swap_out.append((src, dst))
            else:
                # free cpu blocks 
                self._allocators[Device.CPU]._free_block_id(src)
                if src in self._cached_blocks_cpu:
                    self._cached_blocks_cpu.pop(src)
                # swap in
                src = self._get_physical_block_id_unsafe(src)
                dst = self._get_physical_block_id_unsafe(dst)
                blocks_to_swap_in.append((src, dst))
        self._swap_mapping.clear()

        # logger.info("[elasticswap] blocks_to_swap_out=%s blocks_to_swap_in=%s" 
        #             % (len(blocks_to_swap_out), len(blocks_to_swap_in)))
        return blocks_to_swap_out, blocks_to_swap_in

    def will_swap_in_cpu_blocks(self):
        """Check if there are CPU blocks that will be swapped in

        Returns:
            bool: True if there are CPU blocks that will be swapped in, False
                otherwise.
        """
        return bool(self._swap_mapping)