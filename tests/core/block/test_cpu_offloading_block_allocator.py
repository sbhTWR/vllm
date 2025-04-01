import pytest

from vllm.core.evictor import SwapStrategy, FreeBlockSwapScheduler
from vllm.core.block.elastic_swap_block_allocator import (
    CpuOffloadingBlockAllocator)
from vllm.utils import Device, chunk_list


@pytest.mark.parametrize("num_cpu_blocks", [1024])
@pytest.mark.parametrize("num_gpu_blocks", [256])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("allocator_type", ["prefix_caching"])
def test_allocate_mutable_block(num_cpu_blocks: int, num_gpu_blocks: int,
                                block_size: int, allocator_type: str):
    allocator = CpuOffloadingBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=block_size,
    )

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    gpu_blocks = [
        allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        for _ in range(num_gpu_blocks)
    ]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    # assert len(allocator._uncached_blocks) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(0.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    # assert len(allocator._uncached_blocks) == num_gpu_blocks

    _ = [allocator.free(block) for block in gpu_blocks]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(1.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    # assert len(allocator._uncached_blocks) == 0


@pytest.mark.parametrize("num_cpu_blocks", [1024])
@pytest.mark.parametrize("num_gpu_blocks", [256])
@pytest.mark.parametrize("block_size", [2])
@pytest.mark.parametrize("allocator_type", ["prefix_caching"])
@pytest.mark.parametrize("swap_strategy", [SwapStrategy.SWAP_ALL])
def test_swap_all(num_cpu_blocks: int, num_gpu_blocks: int,
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
        range((num_cpu_blocks + num_gpu_blocks) * block_size))
    gpu_token_ids = list(
        chunk_list(unique_token_ids[:num_gpu_blocks * block_size], block_size))
    
    gpu_token_ids2 = list(
        chunk_list(
            unique_token_ids[1*num_gpu_blocks * block_size:2 * num_gpu_blocks *
                             block_size], block_size))
    
    gpu_token_ids3 = list(
        chunk_list(
            unique_token_ids[2*num_gpu_blocks * block_size:3 * num_gpu_blocks *
                             block_size], block_size))

    gpu_token_ids4 = list(
        chunk_list(
            unique_token_ids[3*num_gpu_blocks * block_size:4 * num_gpu_blocks *
                             block_size], block_size))

    gpu_token_ids5 = list(
        chunk_list(
            unique_token_ids[4*num_gpu_blocks * block_size:5 * num_gpu_blocks *
                             block_size], block_size))

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    # assert len(allocator._uncached_blocks) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(0.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    # assert len(allocator._uncached_blocks) == num_gpu_blocks

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])

    print('[pre] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])
    # free blocks 
    for block in gpu_blocks:
        allocator.free(block)
    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(1.0)

    assert len(blocks_to_swap_out) + len(blocks_to_swap_in) == num_gpu_blocks
    assert len(blocks_to_swap_out) == num_gpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks - num_gpu_blocks
    
    # assert len(allocator._uncached_blocks) == 0

    # here all GPU blocks have been freed! 
    # try acessing the prefix tree again! 

    # allocate another gpu sequence to flush out the CPU cache
    # this should cause all blocks to swapped_in

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(2.0)
    print(blocks_to_swap_in)
    assert len(blocks_to_swap_in) == num_gpu_blocks

    # assert that all CPU blocks are free 
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0

    """
    At this point, all GPU blocks are occupied. 
    Allocate and swap 4 different requests to fill the CPU swap space
    Allocate a 5th time to see if it results in an error.
    """

    print('---------------------------------------------------------')
    print('[pre] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])

    for block in gpu_blocks:
        allocator.free(block)

    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(3.0)

    assert len(blocks_to_swap_out) + len(blocks_to_swap_in) == num_gpu_blocks
    assert len(blocks_to_swap_out) == num_gpu_blocks
    

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks - num_gpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    print('---------------------------------------------------------')


    """
    # Pass 2 
    """

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids2
    ]

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])

    for block in gpu_blocks:
        allocator.free(block)

    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(4.0)

    assert len(blocks_to_swap_out) + len(blocks_to_swap_in) == num_gpu_blocks
    assert len(blocks_to_swap_out) == num_gpu_blocks
    

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks - 2 * num_gpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    """
    # Pass 3
    """

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids3
    ]

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])

    for block in gpu_blocks:
        allocator.free(block)

    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(5.0)

    assert len(blocks_to_swap_out) + len(blocks_to_swap_in) == num_gpu_blocks
    assert len(blocks_to_swap_out) == num_gpu_blocks
    

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks - 3 * num_gpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks


    """
    # Pass 4
    """

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids4
    ]

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])

    for block in gpu_blocks:
        allocator.free(block)

    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(6.0)

    assert len(blocks_to_swap_out) + len(blocks_to_swap_in) == num_gpu_blocks
    assert len(blocks_to_swap_out) == num_gpu_blocks
    

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks - 4 * num_gpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks


    """
    get swap scheduler 
    """

    swap_scheduler = allocator._allocators[Device.GPU].swap_scheduler

    """
    # Pass 5
    """

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids5
    ]

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])

    for block in gpu_blocks:
        allocator.free(block)

    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    print('num_free_blocks_gpu=%d' % allocator.get_num_free_blocks(Device.GPU))
    print('num_free_blocks_cpu=%d' % allocator.get_num_free_blocks(Device.CPU))
    print('swap_scheduler=%d' % swap_scheduler.num_blocks)

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(7.0)


    print('num_free_blocks_gpu=%d' % allocator.get_num_free_blocks(Device.GPU))
    print('num_free_blocks_cpu=%d' % allocator.get_num_free_blocks(Device.CPU))
    print('swap_scheduler=%d' % swap_scheduler.num_blocks)

    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    assert allocator.get_num_free_blocks(Device.CPU) == 0
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert allocator._allocators[Device.GPU].swap_scheduler.num_blocks == 256

    
    """
    Verify CPU evictor
    """


    allocator.evict_from_cpu_swap_evictor(256)
    assert allocator.get_num_free_blocks(Device.CPU) == 256

    allocator.evict_from_cpu_swap_evcitor(2048)
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks


    """
    Try swapping again
    """ 
    print('num_free_blocks_gpu=%d' % allocator.get_num_free_blocks(Device.GPU))
    print('num_free_blocks_cpu=%d' % allocator.get_num_free_blocks(Device.CPU))
    print('swap_scheduler=%d' % swap_scheduler.num_blocks)

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(8.0)

    assert len(blocks_to_swap_out) == num_gpu_blocks

    print('num_free_blocks_gpu=%d' % allocator.get_num_free_blocks(Device.GPU))
    print('num_free_blocks_cpu=%d' % allocator.get_num_free_blocks(Device.CPU))
    print('swap_scheduler=%d' % swap_scheduler.num_blocks)


    """
    Try allocating an older request for sanity 
    """
    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]

    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks - num_gpu_blocks


    print('all tests passed')
    

@pytest.mark.parametrize("num_cpu_blocks", [1024])
@pytest.mark.parametrize("num_gpu_blocks", [256])
@pytest.mark.parametrize("block_size", [2])
@pytest.mark.parametrize("allocator_type", ["prefix_caching"])
@pytest.mark.parametrize("swap_strategy", [SwapStrategy.PERSIST])
def test_persist(num_cpu_blocks: int, num_gpu_blocks: int,
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
        range((num_cpu_blocks + num_gpu_blocks) * block_size))
    gpu_token_ids = list(
        chunk_list(unique_token_ids[:num_gpu_blocks * block_size], block_size))
    
    gpu_token_ids2 = list(
        chunk_list(
            unique_token_ids[1*num_gpu_blocks * block_size:2 * num_gpu_blocks *
                             block_size], block_size))
    
    gpu_token_ids3 = list(
        chunk_list(
            unique_token_ids[2*num_gpu_blocks * block_size:3 * num_gpu_blocks *
                             block_size], block_size))

    gpu_token_ids4 = list(
        chunk_list(
            unique_token_ids[3*num_gpu_blocks * block_size:4 * num_gpu_blocks *
                             block_size], block_size))

    gpu_token_ids5 = list(
        chunk_list(
            unique_token_ids[4*num_gpu_blocks * block_size:5 * num_gpu_blocks *
                             block_size], block_size))

    
    gpu_blocks1 = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks1])

    print('[pre] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])
    # free blocks 
    for block in gpu_blocks1:
        allocator.free(block)
    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(1.0)

    assert len(blocks_to_swap_in) == 0
    assert len(blocks_to_swap_out) == 0

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert allocator._allocators[Device.GPU].swap_scheduler.num_blocks == 256


    gpu_blocks2 = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert allocator._allocators[Device.GPU].swap_scheduler.num_blocks == 0


    """
    Try allocating a different request
    Should result in block error
    """

    with pytest.raises(Exception) as e_info:
        gpu_blocks3 = [
            allocator.allocate_immutable_block(prev_block=None,
                                            token_ids=token_ids,
                                            device=Device.GPU)
            for token_ids in gpu_token_ids2
        ]
    
    print('[pre] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])
    for block in gpu_blocks2:
        allocator.free(block)

    print('[post] ref_count for block_id = 0', allocator._allocators[Device.GPU]._hashless_allocator._refcounter._refcounts[0])

    num_evicted = allocator.evict_from_swap_scheduler(2048)

    assert num_evicted == 256
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks
    assert allocator._allocators[Device.GPU].swap_scheduler.num_blocks == 0


    """
    Verify num cached blocks
    """
    gpu_blocks = allocator.allocate_immutable_blocks(prev_block=None,
                                           block_token_ids=gpu_token_ids,
                                           device=Device.GPU)


    num_cached = allocator.num_blocks_cached_for_token_ids(
        prev_block=None,
        block_token_ids=gpu_token_ids)
    assert num_cached == 256

