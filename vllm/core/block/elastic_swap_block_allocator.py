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
from typing import Deque, Dict, List, Optional, Tuple

from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId,
                                        DeviceAwareBlockAllocator)
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator, ElasticSwapBlockAllocator, BlockTracker
from vllm.utils import Device


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
        assert num_gpu_blocks < num_cpu_blocks, "CPU offloading block "\
            "allocator requires the allocated CPU memory capacity to be larger"\
            " than GPU memory capacity."
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
        """
        GPU block should only be in one of the following three status:
          uncached: allocated blocks that didn't hit any cache
          cached: allocated blocks that are cached, either in GPU or in CPU
          free: the blocks are not allocated by block allocator
        This implementation aims to transform uncached blocks to cached blocks
        by performing GPU to CPU copy when calling `get_and_reset_swaps`
        
        As block allocator will automatically track free blocks, and we don't 
        need to specially handle cached blocks. So we only track uncached blocks
        """
        self._uncached_blocks: Deque[Block] = deque()
        """
        We probe CPU cache hit by trying to allocate a CPU 
        block and see if it is computed.
        If we hit the CPU cache, we cannot free this CPU block until the end 
        of scheduler step, in order to avoid the CPU cache being overwritten.
        so we track the cpu blocks we allocated, and free it after scheduler
        step (i.e. calling `get_and_reset_swaps`).
        """
        self._allocated_cpu_blocks: Deque[Block] = deque()

        self.num_gpu_blocks = gpu_block_allocator.get_num_total_blocks()
        self.num_cpu_blocks = cpu_block_allocator.get_num_total_blocks()

        self._swap_mapping: Dict[int, int] = {}

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
        assert device == Device.GPU, "Calls to CPU offloading block allocator "\
            "should always use Device.GPU --- CPU offloading block allocator "\
            "handles CPU offloading internally."\
        # mark this block as uncached

        block = self._allocators[device].allocate_mutable_block(
            prev_block, extra_hash=extra_hash)
        self._uncached_blocks.append(block)
        return block

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

        # allocate a GPU block
        block = self._allocators[device].allocate_immutable_block(
            prev_block, token_ids, extra_hash=extra_hash)
        block_id = block.block_id
        assert block_id is not None
        block_computed = self._allocators[device].block_is_computed(block_id)

        # deal with prefix caching, three cases in total:
        # 1. cache hit on GPU
        # 2. no cache hit on GPU but cache hit on CPU
        # 3. no cache hit
        if block_computed:
            # cache hit on GPU, no need to put it into uncached blocks
            if not self._is_gpu_block_unsafe(block.block_id):
                # cpu block id, needs swap_in 
                # allocate a block_id from cpu 
                gpu_block_id = self._allocators[device]._allocate_block_id()
                self._swap_mapping[block.block_id] = gpu_block_id

                # replace block_ids
                self.replace_block_ids_in_cached_allocator(
                    block.block_id,
                    gpu_block_id,
                    block.content_hash,
                    now=False
                )

                block.block_id = gpu_block_id
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
            self, old_block_id, new_block_id, hash_value, now):
        gpu_allocator = self._allocators[Device.GPU]

        # replace in hash map 
        gpu_allocator._cached_blocks[hash_value] = new_block_id
        
        # replace in block tracker
        block_tracker = gpu_allocator._block_tracker
        assert old_block_id in block_tracker
        new_block_tracker_obj = BlockTracker()
        new_block_tracker_obj.computed = True
        new_block_tracker_obj.active = True

        if now:
            new_block_tracker_obj.last_accessed = now
        else:
            new_block_tracker_obj.last_accessed = block_tracker[old_block_id].last_accessed

        block_tracker[new_block_id] = new_block_tracker_obj
        block_tracker[old_block_id].disable()
        


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

        blocks_to_swap_out = []
        blocks_to_swap_in = []

        gpu_allocator = self._allocators[Device.GPU]
        cpu_allocator = self._allocators[Device.CPU]

        gpu_blocks_to_swap_out = gpu_allocator.get_and_reset_swap_blocks()

        for gpu_block_id, block_metadata in gpu_blocks_to_swap_out:
            cpu_block_id = cpu_allocator._allocate_block_id()
            hash_value = block_metadata.content_hash
            
            self.replace_block_ids_in_cached_allocator(
                gpu_block_id,
                cpu_block_id,
                hash_value,
                now
            )

            # free gpu block id 
            gpu_allocator._hashless_allocator.free_block_id(gpu_block_id)

            src = self._get_physical_block_id_unsafe(gpu_block_id)
            dst = self._get_physical_block_id_unsafe(cpu_block_id)
            blocks_to_swap_out.append((src, dst))

        for src, dst in self._swap_mapping.items():
            # only two possible cases: CPU -> GPU, or GPU -> CPU
            #if src in self._allocators[Device.GPU].all_block_ids:
            if self._is_gpu_block_unsafe(src):
                # swap out
                src = self._get_physical_block_id_unsafe(src)
                dst = self._get_physical_block_id_unsafe(dst)
                blocks_to_swap_out.append((src, dst))
            else:
                # swap in
                src = self._get_physical_block_id_unsafe(src)
                dst = self._get_physical_block_id_unsafe(dst)
                blocks_to_swap_in.append((src, dst))
        self._swap_mapping.clear()
        return blocks_to_swap_out, blocks_to_swap_in

    def will_swap_in_cpu_blocks(self):
        """Check if there are CPU blocks that will be swapped in

        Returns:
            bool: True if there are CPU blocks that will be swapped in, False
                otherwise.
        """
        return bool(self._swap_mapping)