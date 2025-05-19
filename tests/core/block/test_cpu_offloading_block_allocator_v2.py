import pytest

from vllm.core.block.prefix_caching_block import ElasticSwapBlockAllocator
from vllm.core.block_manager import AllocationContextManager
from vllm.core.evictor import SwapStrategy, FreeBlockSwapScheduler
from vllm.core.block.elastic_swap_block_allocator import (
    CpuOffloadingBlockAllocator)
from vllm.sequence import SequenceGroup, SequenceGroupHints
from vllm.utils import Device, chunk_list
from ..utils import create_dummy_prompt


@pytest.mark.parametrize("num_cpu_blocks", [1024])
@pytest.mark.parametrize("num_gpu_blocks", [256])
@pytest.mark.parametrize("block_size", [2])
@pytest.mark.parametrize("allocator_type", ["prefix_caching"])
@pytest.mark.parametrize("swap_strategy", [SwapStrategy.SWAP_HINTS])
def test_swap_hints(num_cpu_blocks: int, num_gpu_blocks: int,
                                  block_size: int, allocator_type: str,
                                  swap_strategy: SwapStrategy):
    allocator = CpuOffloadingBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=block_size,
        swap_strategy=swap_strategy
    )

    unique_token_ids = list(
        range(2*(num_cpu_blocks + num_gpu_blocks) * block_size))

    gpu_token_ids = list(
        chunk_list(unique_token_ids[:num_gpu_blocks//2 * block_size], block_size))
    
    gpu_token_ids2 = list(
        chunk_list(
            unique_token_ids[1*num_gpu_blocks//2 * block_size: num_gpu_blocks *
                             block_size], block_size))
    
    gpu_token_ids3 = list(
        chunk_list(
            unique_token_ids[num_gpu_blocks * block_size: 3 * num_gpu_blocks *
                             block_size // 2], block_size))

    # gpu_token_ids4 = list(
    #     chunk_list(
    #         unique_token_ids[3*num_gpu_blocks * block_size:4 * num_gpu_blocks *
    #                          block_size], block_size))

    # gpu_token_ids5 = list(
    #     chunk_list(
    #         unique_token_ids[4*num_gpu_blocks * block_size:5 * num_gpu_blocks *
    #                          block_size], block_size))

    # set context 
    _, seq_group = create_dummy_prompt(str(0),
                                    prompt_length=block_size,
                                    block_size=block_size)
    
    seq_group.user_id = "test_agent"
    seq_group.hints = SequenceGroupHints(kv_reuse_expected_duration_s=1)

    # set the context 

    with AllocationContextManager(block_allocator=allocator, seq_group=seq_group):

        gpu_blocks = [
            allocator.allocate_immutable_block(prev_block=None,
                                            token_ids=token_ids,
                                            device=Device.GPU)
            for token_ids in gpu_token_ids
        ]

    seq_group.user_id = "test_agent2"
    seq_group.hints = SequenceGroupHints(kv_reuse_expected_duration_s=10)

    # set the context 

    with AllocationContextManager(block_allocator=allocator, seq_group=seq_group):
        gpu_blocks2 = [
            allocator.allocate_immutable_block(prev_block=None,
                                            token_ids=token_ids,
                                            device=Device.GPU)
            for token_ids in gpu_token_ids2
        ]

    # verify swap hints 
    assert allocator._allocators[Device.GPU]._block_tracker[0].reuse_expected_time_s == 1
    assert allocator._allocators[Device.GPU]._block_tracker[num_gpu_blocks//2 + 1].reuse_expected_time_s == 10

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    # assert len(allocator._uncached_blocks) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(0.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    # assert len(allocator._uncached_blocks) == num_gpu_blocks

    # maek the blocks as computed 
    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])
    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks2])

    print('[pre] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])
    # free blocks 
    for block in gpu_blocks:
        allocator.free(block)
    
    for block in gpu_blocks2:
        allocator.free(block)
    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    # free the blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks
    
    # number of blocks in swap evictor should be equal to allthe gpu blocks 
    assert allocator._allocators[Device.GPU].swap_scheduler.num_blocks == num_gpu_blocks

    # the agent that should be exiting is test_agent2
    
    seq_group.user_id = "test_agent3"
    seq_group.hints = SequenceGroupHints(kv_reuse_expected_duration_s=1)
    # after next allocation, all blocks in cpu must be from test_agent2
    with AllocationContextManager(block_allocator=allocator, seq_group=seq_group):
        gpu_blocks3 = [
            allocator.allocate_immutable_block(prev_block=None,
                                            token_ids=token_ids,
                                            device=Device.GPU)
            for token_ids in gpu_token_ids3
        ]
    
    # get blocks for swap 
    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(0.0)
    assert len(blocks_to_swap_out) == num_gpu_blocks // 2
    assert len(blocks_to_swap_in) == 0

    # # iterate over cpu block
    # assert allocator._allocators[Device.CPU]
    print(blocks_to_swap_out)
    cpu_block_id = blocks_to_swap_out[1][1] + num_gpu_blocks
    gpu_allocator: ElasticSwapBlockAllocator = allocator._allocators[Device.GPU]
    user = gpu_allocator._block_tracker[cpu_block_id].last_accessed_by_user

    print("user swapped out", user)

    # assert allocator._allocators[Device.CPU]
    
    


