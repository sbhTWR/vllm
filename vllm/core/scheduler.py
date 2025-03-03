# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

import numpy as np

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceStatus, merge_seq_groups_recompute,
                           merge_seq_groups_persist)
from vllm.utils import Device, PyObjectCache, get_current_time

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    # Number of cached tokens in the batch.
    _num_cached_tokens: int = 0
    # Number of actual non-cached tokens in the batch.
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        # We allow num_new_tokens to be 0 when the entire sequence has
        # been cached.
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self,
                               req_id: str,
                               num_batched_tokens: int,
                               num_cached_tokens: int = 0):
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens
        self._num_cached_tokens += num_cached_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        assert 0 <= self.num_prefill_groups <= len(self.scheduled_seq_groups)

        def key_fn(group: ScheduledSequenceGroup):
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                # Sort sequence groups so that all prefills come before all
                # decodes as required by chunked prefill.
                return (not group.seq_group.is_prefill(), *key)
            return key

        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups,
                                           key=key_fn)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    # Optimization for fast-access to seq_group lists
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            decode_seq_groups_list=[],
            prefill_seq_groups_list=[],
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[ScheduledSequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


def seq_group_metadata_builder():
    return SequenceGroupMetadata(request_id="",
                                 is_prompt=False,
                                 seq_data={},
                                 sampling_params=None,
                                 block_tables={})


def scheduler_running_outputs_builder():
    return SchedulerRunningOutputs(decode_seq_groups=[],
                                   prefill_seq_groups=[],
                                   preempted=[],
                                   swapped_out=[],
                                   blocks_to_swap_out=[],
                                   blocks_to_copy=[],
                                   num_lookahead_slots=0,
                                   prefill_seq_groups_list=[],
                                   decode_seq_groups_list=[])


def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(SequenceGroup.__new__(SequenceGroup),
                                  token_chunk_size=0)
    # return ScheduledSequenceGroup(seq_group=None, token_chunk_size=0)


class SwapSchedulerPolicy(enum.Enum):
    SWAP_ALL = enum.auto()
    SWAP_EXPONENTIAL = enum.auto()
    SWAP_TB = enum.auto()

class SwapSchedulerDirection(enum.Enum):
    SWAP_IN = enum.auto()
    SWAP_OUT = enum.auto()

class SwapSchedulerDevice(enum.Enum):
    CPU = enum.auto()
    GPU = enum.auto()

@dataclass
class SwapScheduleRequest:
    policy: SwapSchedulerPolicy
    seq_group: SequenceGroup
    device: SwapSchedulerDevice
    direction: SwapSchedulerDirection
    blocks_to_swap: list = field(default_factory=lambda: [])
    state: dict = field(default_factory=lambda: {})
    blocks_to_swap_pending: list = field(default_factory=lambda: [])

class SwapScheduler:
    def __init__(self, 
                 scheduler: 'Scheduler',
                 pending_iteration_delay: int = 0.01) -> None:
        
        self.scheduler = scheduler
        self.pending = {}
        self.swap_state = {}
        self.pop_on_next_iteration = set()
        self.do_schedule_deadlock_resolve = False
        self.pending_iteration_delay = pending_iteration_delay
        self.engine_iteration_in_progress = False 
        self.noop_length_history = defaultdict(lambda : [])
        
        self.memory_pressure = 0
        # self.memory_pressure_thresh = 500000
        self.memory_pressure_thresh = self.scheduler.scheduler_config.memory_pressure_thresh
        self.tb_mem_pressure_adaptive = self.scheduler.scheduler_config.tb_mem_pressure_adaptive

        self.token_bucket = 0

        # self.token_bucket_rate = 0.001
        self.token_bucket_rate = self.scheduler.scheduler_config.tb_rate

        # print('tb_rate', self.token_bucket_rate)

        # self.token_bucket_max = 10000
        self.token_bucket_max = self.scheduler.scheduler_config.tb_max

        # print('tb_max', self.token_bucket_max)

        self.memory_flux_fraction = self.scheduler.scheduler_config.memory_flux_fraction


        # input()
        self.num_blocks_swapped_for_deadlocks = 0

    def is_memory_pressure_beyond_thresh(self):
        return self.memory_pressure >= self.memory_pressure_thresh

    def engine_iteration_start(self):
        self.engine_iteration_in_progress = True 
    
    def engine_iteration_stop(self):
        # print("engine stop triggered --> requests to be popped: ", 
                # self.pop_on_next_iteration)
        
        self.engine_iteration_in_progress = False 

    def schedule_resolve_deadlock(self):
        self.do_schedule_deadlock_resolve = True

    def resolve_deadlock(self, 
                         blocks_to_swap_out: list,
                         finished_seq_groups: list):
        # run an iteration to resolve 
        self.do_schedule_deadlock_resolve = False
        
        """do not run deadlock scheduler when no swapping budget is allowed"""
        if self.token_bucket_max == 0:
            return

        top = None 
        # 1. Determine how much to swap-out 
        # peek over the returning queue to see how much allocation 
        # is needed 

        # 2. If returning queue is empty, then peek the waiting queue 
        
        # 3. Look at the paused queue and swap out entire requests until 
        # requirement is fulfilled.  

        # 4. If requirement still not fulfilled, then swap out requests from the 
        # returning queue until fulfilled. This should ignore the first request 
        # in the queue ofcourse, as we are trying to allocate it. 
        if self.scheduler.returning:
            top = self.scheduler.returning[0]
        
        elif self.scheduler.waiting:
            top = self.scheduler.waiting[0]
        else:
            ValueError("Deadlock scheduler wrongly called!")

        # print('top', top)
        # input()

        required_blocks = self.scheduler.block_manager._get_blocks_for_swap_in(
            top
        )

        if required_blocks is None:
            required_blocks = self.scheduler.block_manager.num_blocks_required_to_allocate(
                top
            )
        
        required_num_blocks = len(required_blocks) if isinstance(required_blocks, list) else required_blocks

        # if required_blocks > 0
        # print('required blocks', required_blocks)
        # input()

        # print('required_blocks', required_blocks)
        # input()

        blocks_scheduled = 0
        for _, swap_request in self.pending.items():
            swap_request: SwapScheduleRequest
            num_blocks = len(swap_request.blocks_to_swap_pending)
            _blocks_to_swap_out = []
            # self.scheduler._swap_out(swap_request.seq_group, _blocks_to_swap_out, status=None)
            seq_group = swap_request.seq_group
            _blocks_to_swap_out = self.scheduler.block_manager.swap_out_passthrough(seq_group.first_seq, swap_request.blocks_to_swap_pending)
            blocks_to_swap_out += _blocks_to_swap_out
            swap_request.blocks_to_swap_pending = []
            finished_seq_groups.append(swap_request.seq_group)
            self.pop_on_next_iteration.add(swap_request.seq_group.request_id)

            blocks_scheduled += num_blocks

            if blocks_scheduled >= required_num_blocks:
                break
        
        # print('blocks_scheduled', blocks_scheduled)
        # print('blocks_to_swap_out', len(blocks_to_swap_out))
        # input()

        if required_num_blocks > blocks_scheduled:
            # look into the returning queue 
        
            # look into the waiting queue 
            if len(self.scheduler.returning) >= 2:
                returning = list(self.scheduler.returning)[1:]
                for seq_group in returning:
                    _blocks_to_swap = []

                    remaining_blocks = self.scheduler.block_manager._get_blocks_remaining_for_swap_out(
                        seq_group=seq_group
                    )

                    num_blocks = len(remaining_blocks)
                    blocks_to_swap_out += self.scheduler.block_manager.swap_out_passthrough(seq_group.first_seq, remaining_blocks)
                    blocks_scheduled += num_blocks

                    if blocks_scheduled >= required_num_blocks:
                        break 

        if required_num_blocks > blocks_scheduled:                    
            # look into the waiting queue 
            if len(self.scheduler.waiting) >= 2:
                waiting = list(self.scheduler.waiting)[1:]
                for seq_group in waiting:
                    _blocks_to_swap = []

                    remaining_blocks = self.scheduler.block_manager._get_blocks_remaining_for_swap_out(
                        seq_group=seq_group
                    )

                    num_blocks = len(remaining_blocks)
                    blocks_to_swap_out += self.scheduler.block_manager.swap_out_passthrough(seq_group.first_seq, remaining_blocks)

                    blocks_scheduled += num_blocks

                    if blocks_scheduled >= required_num_blocks:
                        break 
        
        # deadlock should have been resolved here!!
        # print('released blocks: %d' % blocks_scheduled)
        # input()
        return blocks_scheduled

    def add_sequence(self, swap_request: SwapScheduleRequest):
        if swap_request.device == SwapSchedulerDevice.GPU:
            raise ValueError("swap scheduler: offload to GPU not supported!")
        
        swap_request.state['swap_iters'] = 0
        swap_request.state['time_scheduled'] = get_current_time()
        # self.pending.append(swap_request)
        # print('adding request %s' % swap_request.seq_group.request_id)
        # input()
        user_id = swap_request.seq_group.user_id
        assert user_id not in self.pending, "user_id already in self.pending!"
        self.pending[swap_request.seq_group.user_id] = swap_request
        user_id = swap_request.seq_group.user_id

        logger.info("[swap_scheduler] adding swap request for user_id=%s " 
                    "blocks_to_swap=%d direction=%s" 
                    % (user_id, len(swap_request.blocks_to_swap), swap_request.direction))

        # if request_id == '0':
        #     print('adding request=%s' % request_id)
        #     print('pending', self.pending.keys())
        #     print('waiting', self.scheduler.waiting)
        #     print('returning', self.scheduler.returning)
        #     input()

    def pending_requests(self) -> bool:
        return len(self.pending) > 0

    def schedule(self, 
                 blocks_to_swap_in: list, 
                 blocks_to_swap_out: list,
                 finished_seq_groups: list):
        
        while self.pop_on_next_iteration:
            user_id = self.pop_on_next_iteration.pop()
            # print("removing request %s" % request_id)
            # input()
            self.pending.pop(user_id)

        tb_requests = []
        pending_requests = list(self.pending.values())
        # if len(self.pending.keys()) > 1:
            # print(self.pending.keys())
            # input()
        for swap_request in pending_requests:
            swap_request: SwapScheduleRequest

            if swap_request.policy == SwapSchedulerPolicy.SWAP_ALL:
                self.schedule_swap_all(
                    swap_request,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    finished_seq_groups=finished_seq_groups
                )
            
            elif swap_request.policy == SwapSchedulerPolicy.SWAP_EXPONENTIAL:
                self.schedule_swap_exponential(
                    swap_request,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    finished_seq_groups=finished_seq_groups
                )
            
            elif swap_request.policy == SwapSchedulerPolicy.SWAP_TB:
                tb_requests.append(swap_request)
            else:
                raise ValueError("invalid swap scheduler policy: %s" 
                                % self.swap_scheduler_policy)

        # self.schedule_swap_tb(
        #     tb_requests=tb_requests,
        #     blocks_to_swap_in=blocks_to_swap_in,
        #     blocks_to_swap_out=blocks_to_swap_out,
        #     finished_seq_groups=finished_seq_groups
        # )

        if self.do_schedule_deadlock_resolve:
            num_blocks_swapped_for_deadlock = self.resolve_deadlock(
                blocks_to_swap_out=blocks_to_swap_out,
                finished_seq_groups=finished_seq_groups
            )

            if num_blocks_swapped_for_deadlock:
                self.num_blocks_swapped_for_deadlocks += num_blocks_swapped_for_deadlock

    def get_memory_pressure(self):
        """
        use self.scheduler to fetch memory pressure 
        memory pressure = number of tokens queued 
        """
        num_tokens_queued = 0 
        for seq_group in self.scheduler.waiting:
            num_tokens_queued += seq_group.seqs[0].get_len()
        
        # print('memory pressure', num_tokens_queued)
        return num_tokens_queued

    def record_noop_length(self, request_id, noop_length: float):
        self.noop_length_history[request_id].append(noop_length)

    def get_api_length_prediction(self, user_id, method='average'):
        """
        get api length prediction from swap request history
        """
        if user_id in self.noop_length_history:
            if method == 'average':
                return np.mean(self.noop_length_history[user_id])
            else:
                raise ValueError('Noop prediction method=%s not supported' % method)
        else:
            return None

    def calculate_memory_flux(self, swap_request: SwapScheduleRequest):
        user_id = swap_request.seq_group.user_id
        time_now = get_current_time()
        time_scheduled = swap_request.state['time_scheduled']
        time_elapsed = time_now - time_scheduled
        time_prediction = self.get_api_length_prediction(user_id)

        if not time_prediction:
            # for first time, incline to swap 
            flux = 1e9
        else:
            time_remaining = max(time_prediction - time_elapsed, 0)
            mem_num_tokens = len(swap_request.blocks_to_swap_pending) * self.scheduler.cache_config.block_size
            
            if time_remaining > 0:
                flux = mem_num_tokens * time_remaining
            else:
                flux = 1e9

        return flux

    def schedule_swap_tb(self,
                          tb_requests: list,
                          blocks_to_swap_in: list, 
                          blocks_to_swap_out: list,
                          finished_seq_groups: list):
        
        # if get_current_time() > 500:
        #     print(self.pending.keys())
        #     # print(self.scheduler.api_times)s
        #     input()
        # print(self.pending.keys())
        # print('tb_requests', len(tb_requests))
        # fill token bucket 

        self.memory_pressure = self.get_memory_pressure()

        if self.tb_mem_pressure_adaptive:
            self.token_bucket = min(self.token_bucket + self.token_bucket_rate * self.memory_pressure,
                                    self.token_bucket_max)
        else:
            self.token_bucket = self.token_bucket_max
        
        # active_work = len(self.pending.keys()) > 0 and self.memory_pressure > 0
        active_work = False
        if active_work:
            print('token_bucket (before)', self.token_bucket)
            print('memory_pressure (before)', self.memory_pressure)
            print('requests: ', self.pending.keys())
            for req_id, req in self.pending.items():
                req: SwapScheduleRequest
                print('pending req_id %s --> %s direction=%s' 
                      % (
                          req_id, 
                          req.policy,
                          req.direction
                        )
                    )
            # input()
        # calculate memory flux for all requests and sort 

        priority_queue = []
        for swap_request in tb_requests:
            swap_request: SwapScheduleRequest
            # print('active swap request --> %s (%s) direction=%s' 
            #       % (
            #           swap_request.seq_group.request_id, 
            #           swap_request.policy,
            #           swap_request.direction
            #     ))
            ele = (
                -self.calculate_memory_flux(swap_request), 
                int(swap_request.seq_group.user_id), 
                swap_request
            )
            # print(ele[0])
            heapq.heappush(priority_queue, ele)

        if active_work:
            print('--- pq ----')
            for ele in priority_queue:
                print(ele[0], ele[1])
            
            print('--------s')

        while priority_queue:
            flux, req_id, swap_request = heapq.heappop(priority_queue)
            
            num_blocks = int(min(self.token_bucket, 
                             self.memory_flux_fraction * (-flux)
                             ) / self.scheduler.cache_config.block_size)

            # if active_work:
            #     print('req_id=%d num_blocks_proposed=%d' % (req_id, num_blocks))

            assert num_blocks >= 0

            num_blocks_pending = len(swap_request.blocks_to_swap_pending)
            num_blocks_to_swap = min(num_blocks_pending, num_blocks)

            num_blocks_remaining = num_blocks_pending - num_blocks_to_swap

            # print(num_blocks_remaining)
            # print("token_bucket", self.token_bucket)
            if num_blocks_to_swap > 0:
                seq_group = swap_request.seq_group
                blocks_to_swap = swap_request.blocks_to_swap_pending[num_blocks_remaining:]
                # print(blocks_to_swap)
                swap_request.blocks_to_swap_pending = swap_request.blocks_to_swap_pending[:num_blocks_remaining]

                blocks_to_swap_out += self.scheduler.block_manager.swap_out_passthrough(seq_group.first_seq, blocks_to_swap)
                num_tokens_to_schedule = num_blocks_to_swap * self.scheduler.cache_config.block_size
                self.token_bucket = max(self.token_bucket - num_tokens_to_schedule, 0)
            
            if num_blocks_remaining == 0:
                # print('completed')
                finished_seq_groups.append(swap_request.seq_group)
                self.pop_on_next_iteration.add(swap_request.seq_group.user_id)

            if self.token_bucket == 0:
                break 
        
        # if active_work:
        #     print('token_bucket (after)', self.token_bucket)
        #     print('memory_pressure (after)', self.memory_pressure)
        #     print('paused swap pending requests', self.pending.keys())
        #     print('num_blocks_scheduled', len(blocks_to_swap_out))

            block_tables = self.scheduler.block_manager.block_tables
            for key, value in block_tables.items():
                
                # print('request=%d' % key, len(value.physical_block_ids))
                gpu_blocks, cpu_blocks = self.scheduler.block_manager._get_gpu_cpu_distribution(key)
                print('request=%d' % key, 
                        len(value.physical_block_ids), " | gpu: ", gpu_blocks, " cpu: ", cpu_blocks
                )

            # input()

    def schedule_swap_all(self, 
                          swap_request: SwapScheduleRequest,
                          blocks_to_swap_in: list, 
                          blocks_to_swap_out: list,
                          finished_seq_groups: list
                        ):
        

        if swap_request.direction == SwapSchedulerDirection.SWAP_OUT:
            # blocks_to_swap_out += swap_request.blocks_to_swap
            blocks_to_swap_out += self.scheduler.block_manager.swap_out_passthrough(
                swap_request.seq_group.first_seq, swap_request.blocks_to_swap)

            # logger.info("[swap_scheduler] scheduling swap all for user_id=%s "
            #             "blocks_to_swap_out=%s"
            #             % (swap_request.seq_group.user_id, blocks_to_swap_out))

        elif swap_request.direction == SwapSchedulerDirection.SWAP_IN:
            blocks_to_swap_in += self.scheduler.block_manager.swap_in(swap_request.seq_group)
        
        else:
            raise ValueError("invalid swap scheduler direction: %s" 
                                % swap_request.direction)
        
        finished_seq_groups.append(swap_request.seq_group)
        self.pop_on_next_iteration.add(swap_request.seq_group.user_id)
        # self.pending.pop(swap_request.seq_group.request_id)

        logger.info("[swap_scheduler] scheduling swap all for user_id=%s "
                    "blocks_to_swap_out=%d" 
                    % (swap_request.seq_group.user_id, len(blocks_to_swap_out)))
    
    def schedule_swap_exponential(self, 
                        swap_request: SwapScheduleRequest,
                        blocks_to_swap_in: list, 
                        blocks_to_swap_out: list,
                        finished_seq_groups: list
                    ):

        # print("schedule_swap_exponential for req_id=", swap_request.seq_group.request_id)

        if swap_request.direction == SwapSchedulerDirection.SWAP_OUT:
            
            num_iters_elapsed = swap_request.state['swap_iters']
            exponent = num_iters_elapsed + 1 
            num_blocks_to_swap = int(pow(self.swap_exp_base, exponent))
            # print("num_blocks_to_swap, power calculation", num_blocks_to_swap)
            # input()
            num_blocks = len(swap_request.blocks_to_swap_pending)
            num_blocks_to_swap = min(num_blocks_to_swap, num_blocks)
            num_blocks_remaining = num_blocks - num_blocks_to_swap

            # print('Iteration: %d, number of blocks to swap: %d remaining: %d' 
                #   % (exponent, num_blocks_to_swap, num_blocks_remaining))

            # input()
            if num_blocks_to_swap > 0:
                seq_group = swap_request.seq_group
                blocks_to_swap = swap_request.blocks_to_swap_pending[num_blocks_remaining:]
                # print(blocks_to_swap)
                swap_request.blocks_to_swap_pending = swap_request.blocks_to_swap_pending[:num_blocks_remaining]
                swap_request.state['swap_iters'] = num_iters_elapsed + 1

                # schedule 
                blocks_to_swap_out += self.scheduler.block_manager.swap_out_passthrough(seq_group.first_seq, blocks_to_swap)

                # print('pending blocks', swap_request.blocks_to_swap_pending)
                # print("swap scheduled")
                # print(len(blocks_to_swap_out))
                # input()

            else:
                # completed 
                # print('completed')
                finished_seq_groups.append(swap_request.seq_group)
                self.pop_on_next_iteration.add(swap_request.seq_group.user_id)

        elif swap_request.direction == SwapSchedulerDirection.SWAP_IN:
            raise ValueError("swap_in not supported in exponential mode: %s" 
                                % swap_request.direction)
        
        else:
            raise ValueError("invalid swap scheduler direction: %s" 
                                % swap_request.direction)


    def interrupt_swap_schedule_for_request(self, seq_group: SequenceGroup):
        user_id = seq_group.user_id
        # print('interrupting req_id', user_id)
        # input()
        if user_id in self.pending:
            self.pending.pop(user_id)
        
        # check if it is in next iteration pop queue 
        filtered_pop_list = [req_id for req_id in self.pop_on_next_iteration if req_id is not user_id]
        self.pop_on_next_iteration = set(filtered_pop_list)

        # if user_id == '0':
        #     print('interrupting request=%s' % user_id)
        #     print('pending', self.pending.keys())
        #     print('waiting', self.scheduler.waiting)
        #     print('returning', self.scheduler.returning)

    def check_finished(self, seq_group: SequenceGroup):
        if seq_group.user_id in self.pending:
            return False 
        else:
            return True


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "selfattn"
        if (self.scheduler_config.runner_type == "pooling"
                or self.cache_config.is_attention_free):
            version = "placeholder"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()

        # Sequence groups that have been paused 
        self.paused: dict[str, SequenceGroup] = {}

        # Sequence groups that are in the RETURNING state.
        self.returning: Deque[SequenceGroup] = deque()

        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

        # Used to cache python objects
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # For async output processing, we need to swap cache buffers between
        # iterations. I.e. since the output processing is lagged one step,
        # we cannot reuse the cached objects immediately when the schedule()
        # is called again, but only when schedule() is called the second time.
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(
                PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder))

        # For async postprocessor, the extra decode run cannot be done
        # when the request reaches max_model_len. In this case, the request
        # will be stopped during schedule() call and added to this stop list
        # for processing and deallocation by the free_finished_seq_groups()
        self._async_stopped: List[SequenceGroup] = []

        self.swap_scheduler = SwapScheduler(
            scheduler=self
        )

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def update_seq_group(self, new_seq_group: SequenceGroup, 
                            seq_group: SequenceGroup,
                            fr_policy) -> SequenceGroup:
        
        assert new_seq_group.user_id == seq_group.user_id, "user_id does not match!"
        # TODO: add update logic here
        if fr_policy == "pause_recompute":
            seq_group = merge_seq_groups_recompute(new_seq_group, seq_group)
        elif fr_policy == "pause_swap":
            seq_group = merge_seq_groups_persist(new_seq_group, seq_group)
        elif fr_policy == "pause_persist":
            seq_group = merge_seq_groups_persist(new_seq_group, seq_group)
        else:
            raise ValueError("invalid fr_policy=%s" % fr_policy)

        return seq_group

    def requeue_seq_group(self, seq_group: SequenceGroup) -> None:
        # TODO add prio requeue logic here
        self.add_seq_group(seq_group)

    def resume_seq_group(self, new_seq_group: SequenceGroup) -> None:
        user_id = new_seq_group.user_id 
        user_args = new_seq_group.user_args

        message_type = user_args.get('type', None)
        if message_type == 'fin':
            # kill the sequence group and exit 
            logger.info("[elasticswap] killing sequence group for user_id=%s" % user_id)
            if user_id in self.paused:
                seq_group, _ = self.paused.pop(user_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

                self._free_seq_group_cross_attn_blocks(seq_group)
            
            return

        seq_group, fr_policy = self.paused[user_id]
        if fr_policy == "pause_recompute":
            seq_group = self.update_seq_group(new_seq_group, seq_group, fr_policy)
            self.requeue_seq_group(seq_group)
        
        elif fr_policy == "pause_swap":
            seq_group = self.update_seq_group(new_seq_group, seq_group, fr_policy)
            self.swap_scheduler.interrupt_swap_schedule_for_request(seq_group)
            self.requeue_seq_group(seq_group)
        
        elif fr_policy == "pause_persist":
            seq_group = self.update_seq_group(new_seq_group, seq_group, fr_policy)
            self.requeue_seq_group(seq_group)
        else:
            raise ValueError("invalid fr_policy %s" % fr_policy)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the swapped queue.
        # Only for testing purposes.
        self.swapped.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        """
        Free a sequence group from a cross-attention block table.
        Has no effect on decoder-only models.
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0 or self.swap_scheduler.pending_requests()

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self) -> bool:
        return self.block_manager.reset_prefix_cache()

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = \
            self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            # We discard the cached tokens info here because we don't need it
            # for running sequence:
            #   1. If a sequence is running with chunked prefill, the cached
            #      tokens info was already used for the first prefill.
            #   2. If a sequence is running with non-chunked prefill, then
            #      there it's a decoding sequence, and the cached tokens info is
            #      irrelevant.
            num_uncached_new_tokens, _ = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.RUNNING, enable_chunking,
                    budget))

            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if self.use_async_output_proc and seq_group.seqs[0].get_len(
            ) > self.scheduler_config.max_model_len:
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = \
                    self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.SWAPPED, enable_chunking,
                    budget))

            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(
                        seq_group,
                        token_chunk_size=num_new_tokens_uncached +
                        num_new_tokens_cached,
                    ))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled and \
                not self.scheduler_config.is_multi_step:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _get_priority(self,
                      seq_group: SequenceGroup) -> Tuple[Optional[int], float]:
        """ Get the priority of the sequence group.
        Highest preference to user-defined priority, followed by arrival time.
        Args:
            seq_group: The sequence group input.
        Returns:
            The priority of the sequence group.
        """
        return seq_group.priority, seq_group.arrival_time

    def _schedule_priority_preemption(
        self,
        budget: SchedulingBudget,
    ) -> int:
        """Sorts waiting and running queue. Also, force preempt requests
        from the running queue if their priority is lower.
        Priority-based preemption is used with the priority policy.
        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
        Returns:
            A count of priority-based preemptions.
        """

        waiting_queue = self.waiting

        running_queue = deque(sorted(self.running, key=self._get_priority))

        blocks_to_swap_out: List[Tuple[int, int]] = []
        force_preemption_count = 0

        if waiting_queue:
            seq_group = waiting_queue.popleft()
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, _ = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, False, budget))

            #Only preempt if priority inversion exists
            while running_queue and self._get_priority(
                    running_queue[-1]) > self._get_priority(seq_group):
                #Only preempt if waiting sequence cannot be allocated
                can_allocate = self.block_manager.can_allocate(seq_group)
                if (num_new_tokens_uncached > 0
                        and can_allocate == AllocStatus.OK
                        and budget.can_schedule(
                            num_new_tokens=num_new_tokens_uncached,
                            num_new_seqs=num_new_seqs,
                        )):
                    break

                #Adjust budget to remove the victim sequence group
                vseq_group = running_queue.pop()
                num_running_tokens_uncached, _ = (
                    self._get_num_new_uncached_and_cached_tokens(
                        vseq_group, SequenceStatus.RUNNING, False, budget))
                budget.subtract_num_batched_tokens(
                    vseq_group.request_id, num_running_tokens_uncached)
                num_running_seqs = vseq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(vseq_group.request_id,
                                         num_running_seqs)

                #Preempt out the victim sequence group
                self._preempt(vseq_group, blocks_to_swap_out)
                waiting_queue.appendleft(vseq_group)
                force_preemption_count += 1
            #Put the sequence back into the waiting queue
            waiting_queue.appendleft(seq_group)

        waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))

        self.waiting = waiting_queue
        self.running = running_queue
        return force_preemption_count

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking and not seq_group.returning:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens,\
                    "num_new_tokens=%s num_prompt_tokens=%s" % (num_new_tokens, num_prompt_tokens)

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # # TODO (Shubham): Check if swap is needed 
            # If swap is needed, then check if can swap 
            if seq_group.first_seq.swap_in_required:
                logger.info("[elasticswap] Swap in required for seq_group %s" % seq_group.request_id)

                seq = seq_group.first_seq
                # Check if can swap_in
                can_swap_in = self.block_manager.can_swap_in_elastic(seq)

                if can_swap_in == AllocStatus.LATER:
                    break
                elif can_swap_in == AllocStatus.NEVER:
                    raise ValueError("should always be swappable")
                
                elif can_swap_in == AllocStatus.OK:
                    mapping = self.block_manager.swap_in_elastic(seq)
                    logger.info("[elasticswap] Swapped in seq_group %s ; mapping=%s" 
                                % (seq_group.request_id, mapping))
                    seq_group.first_seq.swap_in_required = False
                    # TODO:(Shubham): add to swap scheduler later.

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens
                    >= self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)

        if len(prefills.seq_groups
               ) == 0 and self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        paused_blocks_to_swap_in = []
        paused_blocks_to_swap_out = []
        finished_seq_groups = []

        self.swap_scheduler.schedule(
            blocks_to_swap_in=paused_blocks_to_swap_in,
            blocks_to_swap_out=paused_blocks_to_swap_out,
            finished_seq_groups=finished_seq_groups
        )

        # logger.info(
        #         "[elasticswap] Paused swap out: %s", paused_blocks_to_swap_out)

        sched_outputs = SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in + paused_blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + paused_blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

        # if len(paused_blocks_to_swap_out) > 0:
        #     logger.info("[elasticswap] sched_outputs: %s" % sched_outputs)

        return sched_outputs

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(budget,
                                                   curr_loras,
                                                   enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

        prefills = self._schedule_prefills(budget,
                                           curr_loras,
                                           enable_chunking=True)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # Update new running requests.
        # By default, vLLM scheduler prioritizes prefills.
        # Once chunked prefill is enabled,
        # the policy is changed to prioritize decode requests.
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        # Put prefills first due to Attention backend ordering assumption.
        scheduled_seq_groups = (prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups)
        num_prefill_groups = (len(prefills.seq_groups) +
                              len(swapped_in.prefill_seq_groups) +
                              len(running_scheduled.prefill_seq_groups))
        # If all prompts, then we set num_lookahead_slots to 0
        # this allows us to go through the `no_spec` path in
        # `spec_decode_worker.py`
        all_prefills = (len(scheduled_seq_groups) == num_prefill_groups)
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        

        paused_blocks_to_swap_in = []
        paused_blocks_to_swap_out = []
        finished_seq_groups = []

        self.swap_scheduler.schedule(
            blocks_to_swap_in=paused_blocks_to_swap_in,
            blocks_to_swap_out=paused_blocks_to_swap_out,
            finished_seq_groups=finished_seq_groups
        )

        
        logger.info(
                "[elasticswap] Paused swap out: %s", paused_blocks_to_swap_out)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in + paused_blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + paused_blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup,
                          enable_chunking: bool) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        if is_prefill and num_lookahead_slots > 0:
            # Appending prefill slots only happens multi-step and
            # chunked-prefill are enabled together.
            assert self.scheduler_config.is_multi_step and enable_chunking

        return self.block_manager.can_append_slots(
            seq_group=seq_group, num_lookahead_slots=num_lookahead_slots)

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        # async_output_proc is allowed only when we have a single sequence
        # in the sequence group
        no_single_seq = seq_group.sampling_params is None or (
            seq_group.sampling_params.n == 1)
        return no_single_seq

    def schedule(
            self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + num_computed_tokens
                        < seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=seq_group.multi_modal_data
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    multi_modal_placeholders=seq_group.multi_modal_placeholders
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    mm_processor_kwargs=seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(
                    seq_group)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # if len(seq_group_metadata_list) != len(
        #     scheduler_outputs.scheduled_seq_groups):
        #     logger.info("[elasticswap][scheduler] scheduler_outputs: %s" % scheduler_outputs)
        #     logger.info("[elasticswap][scheduler] seq_group_metadata_list: %s" % seq_group_metadata_list)
        # Return results

        return (seq_group_metadata_list, scheduler_outputs,
                allow_async_output_proc)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def add_seq_group_to_paused(self, seq_group: SequenceGroup, fr_policy):
        user_id = seq_group.user_id
        self.paused[user_id] = (seq_group, fr_policy)
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                seq.status = SequenceStatus.FINISHED_PAUSED
                # TODO: Remove this in future: 
                seq.swap_in_required = True

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        logger.info("[elasticswap] freeing seq_id=%d" % seq.seq_id)
        self.block_manager.free(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                logger.info("[elasticswap] freeing seq_id=%d" % seq.seq_id)
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            # Free cross-attention block table, if it exists
            self._free_seq_group_cross_attn_blocks(seq_group)

            # Add the finished requests to the finished requests list.
            # This list will be used to update the Mamba cache in the
            # next step.
            self._finished_requests_ids.append(seq_group.request_id)

        fr_policy = self.scheduler_config.finished_requests_policy
        logger.info("finishing request=%s user_id=%s with fr_policy=%s" 
                        % (seq_group.request_id, seq_group.user_id, fr_policy))
        
        if fr_policy == "default":
            # the default policy is to free and "forget" the request

            # Free finished seqs
            self._free_finished_seqs(seq_group)
        
        elif fr_policy == "pause_recompute":
            # add it to the paused queue and also free it for recomputation later
            self._free_finished_seqs(seq_group)
            self.add_seq_group_to_paused(seq_group, fr_policy)

        elif fr_policy == "pause_swap":
            # Do not free and add the seq_group to paused as is 
            self.add_seq_group_to_paused(seq_group, fr_policy)

            blocks_to_swap_pending = self.block_manager._get_blocks_remaining_for_swap_out(seq_group)
            
            logger.info('[elasticswap] swapping out %d blocks' % len(blocks_to_swap_pending))

            self.swap_scheduler.add_sequence(
                SwapScheduleRequest(
                    policy=SwapSchedulerPolicy.SWAP_ALL,
                    seq_group=seq_group,
                    blocks_to_swap=blocks_to_swap_pending,
                    device=SwapSchedulerDevice.CPU,
                    direction=SwapSchedulerDirection.SWAP_OUT
                )
            )
        
        elif fr_policy == "pause_persist":
            # Do not free and add the seq_group to paused as is 
            self.add_seq_group_to_paused(seq_group, fr_policy)

        else:
            raise ValueError("Invalid finished_requests_policy!") 

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)

        # logger.info("free_finished_seq_groups: seq_groups remaining = %s" % remaining)
        self.running = remaining

        # Handle async stopped sequence groups
        # (ones that reached max model len)
        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)

                # Free finished seqs
                self._free_finished_seqs(seq_group)

            self._async_stopped.clear()

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(self,
                      seq_group: SequenceGroup,
                      blocks_to_copy: List[Tuple[int, int]],
                      enable_chunking: bool = False) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
            enable_chunking (bool): True if chunked prefill is enabled.
        """
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=enable_chunking)

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            # In multi-step chunked-prefill any sequence type can have
            # slots appended.
            seq_status = None

        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt(self, seq_group: SequenceGroup,
                 blocks_to_swap_out: List[Tuple[int, int]]) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.user_specified_preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()
        self._free_seq_group_cross_attn_blocks(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = ((now - earliest_arrival_time)
                            > (self.scheduler_config.delay_factor *
                               self.last_prompt_latency) or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool,
                                 enable_chunking: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.

        When chunking is enabled with multi-step, we allocate lookahead slots
        for the prefills for when the prefills turn into decodes in the first
        step.
        """
        if is_prefill:
            if self.scheduler_config.is_multi_step and enable_chunking:
                # num_lookahead_slots was introduced in the context of decodes,
                # in Speculative Decoding.
                # When the num_scheduler_steps is 8, say, then the
                # num_lookahead_slots is 7. Meaning, we are doing a 1-step of
                # decode anyways and we wish to do 7 more.
                #
                # "lookaheads" for prefills, is introduced in support for
                # Chunked-Prefill in Multi-Step.
                return self.scheduler_config.num_lookahead_slots + 1
            else:
                return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_uncached_and_cached_tokens(
        self,
        seq_group: SequenceGroup,
        status: SequenceStatus,
        enable_chunking: bool,
        budget: SchedulingBudget,
    ) -> Tuple[int, int]:
        """
        Returns the number of new uncached and cached tokens to schedule for a
        given sequence group that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns (0, 0) if the new token cannot be computed due to token budget.

        The cached tokens's blocks are already computed, and the attention
        backend will reuse the cached blocks rather than recomputing them. So
        the scheduler could schedule these cached tokens "for free".

        Args:
            seq_group: The sequence group to get the number of new tokens to
                schedule.
            status: The status of the sequences to get the number of new tokens
                to schedule.
            enable_chunking: Whether to chunk the number of tokens to compute.
            budget: The budget to chunk the number of tokens to compute.


        Returns:
            A tuple of two ints. The first int is the number of new uncached
            tokens to schedule. The second int is the number of cached tokens.
            If no more new tokens can be scheduled, returns (0, 0).
        """
        num_cached_new_tokens = 0
        num_uncached_new_tokens = 0

        seqs = seq_group.get_seqs(status=status)
        # Compute the number of new uncached and cached tokens for
        # each sequence.
        for seq in seqs:
            if not seq.is_prefill():
                # Decode sequences should always just have 1 uncached token
                # TODO(rickyx): Actually is this still correct for multi-step?
                num_uncached_new_tokens += 1
                continue

            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            if not self.cache_config.enable_prefix_caching:
                # If prefix caching is not enabled, all new tokens are uncached.
                num_uncached_new_tokens += all_num_new_tokens_seq
                continue

            # NOTE: the cache token might be currently in a block that's in an
            # evictor meaning that it's not yet allocated. However, we don't
            # exclude such tokens in the cache count because it will be
            # guaranteed to be allocated later if the sequence can be allocated.
            num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(
                seq)

            # Sanity check.
            if num_cached_tokens_seq < num_computed_tokens_seq:
                # This should only happen with chunked prefill, and
                # the seq is still in prefill. The `num_cached_tokens_seq`
                # is the value we calculated on scheduling the first prefill.
                # For subsequent continuous prefill steps, we cached the
                # number of cache tokens for the sequence so the cached token
                # count could be less than the number of computed tokens.
                # See comments on `ComputedBlocksTracker` for more details.
                assert (
                    seq.is_prefill() and seq.status == SequenceStatus.RUNNING
                    and self.scheduler_config.chunked_prefill_enabled
                ), ("Number of cached tokens should not be less than the "
                    "number of computed tokens for a sequence that's still "
                    f"in prefill. But there are {num_cached_tokens_seq} cached "
                    f"tokens and {num_computed_tokens_seq} computed tokens "
                    f"for sequence {seq.seq_id}.")

            num_cached_new_tokens_seq = max(
                0, num_cached_tokens_seq - num_computed_tokens_seq)
            num_uncached_new_tokens_seq = (all_num_new_tokens_seq -
                                           num_cached_new_tokens_seq)

            num_uncached_new_tokens += num_uncached_new_tokens_seq
            num_cached_new_tokens += num_cached_new_tokens_seq

        if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
            # For a fully cached hit sequence, we actually need to recompute the
            # last token. So we need at least 1 uncached token to schedule.
            # See ModelRunner._compute_for_prefix_cache_hit for more details.
            num_uncached_new_tokens = 1
            num_cached_new_tokens -= 1

        if enable_chunking and len(seqs) == 1:
            # Chunk if a running request cannot fit in the given budget.
            # If number of seq > 1, it means it is doing beam search
            # in a decode phase. Do not chunk.
            num_uncached_new_tokens = self._chunk_new_tokens_to_schedule(
                self.scheduler_config,
                self.cache_config,
                budget,
                self._get_prompt_limit(seq_group),
                num_uncached_new_tokens,
            )

        return num_uncached_new_tokens, num_cached_new_tokens

    @staticmethod
    def _chunk_new_tokens_to_schedule(
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        budget: SchedulingBudget,
        prompt_limit: int,
        num_new_tokens: int,
    ) -> int:
        """
        Chunks the number of new tokens to schedule based on the budget when
        chunked prefill is enabled.

        Args:
            scheduler_config: The scheduler config.
            cache_config: The cache config.
            budget: The budget to chunk the number of tokens to compute.
            prompt_limit: The maximum number of tokens allowed in a prompt.
            num_new_tokens: The number of new tokens to schedule.

        Returns:
            The number of new tokens to schedule after chunking.
        """
        remaining_token_budget = budget.remaining_token_budget()
        if scheduler_config.is_multi_step:
            # The current multi-step + chunked prefill capability does
            # not actually support chunking prompts.
            #
            # Therefore, `num_new_tokens` is computed in the same fashion
            # for both multi-step+chunked-prefill &
            # multi-step+chunked-prefill+APC
            #
            # Prompts with more tokens than the current remaining budget
            # are postponed to future scheduler steps
            if num_new_tokens > prompt_limit:
                # If the seq_group is in prompt-stage, pass the
                # num_new_tokens as-is so the caller can ignore
                # the sequence.
                return num_new_tokens

            return (0 if num_new_tokens > remaining_token_budget else
                    num_new_tokens)

        if cache_config.enable_prefix_caching:
            # Adjust the remaining token budget to be divisible by the block
            # size when prefix caching is enabled.

            # When prefix caching is enabled, we always allocate
            # the number of new tokens that is dividable by the block
            # size to avoid partial block matching.
            block_size = cache_config.block_size
            remainder = budget.token_budget % block_size
            if remainder != 0:
                raise ValueError("When enabling chunked prefill and "
                                 "prefix caching, max_num_batched_tokens "
                                 "(chunk size) must be dividable by "
                                 "block size, but got chunk_size "
                                 f"({budget.token_budget}) % block_size "
                                 f"({block_size}) = {remainder}")
            # Round down to block size.
            remaining_token_budget = (remaining_token_budget // block_size *
                                      block_size)

        num_new_tokens = min(num_new_tokens, remaining_token_budget)

        return num_new_tokens
