import asyncio
from typing import List, Optional
from openai import OpenAI
import httpx
from node import Node, ContextStore, ToolCallNode
import time
from collections import deque

class AdaptiveDAGRateController:
    """
    Adaptive rate controller for DAG submission that adjusts submission rate
    based on LLMCall throughput (not DAG throughput).
    """
    def __init__(
        self,
        initial_rate: float = 0.01,  # DAGs per second
        min_rate: float = 0.01,
        max_rate: float = 10.0,
        adjustment_step: float = 0.01,
        observation_window: float = 20.0,  # Seconds to observe before adjusting
        enable: bool = False
    ):
        self.enable = enable
        if not enable:
            return
            
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adjustment_step = adjustment_step
        self.observation_window = observation_window
        
        # Track submission times
        self.submission_times = deque()
        self.last_submission_time = 0.0
        
        # Track LLMCall completions (this is what we care about)
        self.llmcall_completion_times = deque()  # Timestamps of completed LLMCalls
        
        # Throughput tracking
        self.last_throughput = 0.0  # LLMCalls per second
        self.last_observation_time = time.time()

        self.total_llmcalls_completed = 0
        
        # State tracking
        self.rate_increased_last = False
        
    def can_submit(self, current_time: float) -> bool:
        """Check if we can submit a DAG now based on current rate."""
        if not self.enable:
            return True
        

        if self.current_rate <= 0:
            return False 
        
        min_interval = 1.0 / self.current_rate

        if not self.submission_times:
            return True 
        
        time_since_last = current_time - self.last_submission_time

        if time_since_last >= min_interval:
            return True 
        
        return False
        # # Remove old submission records (older than 1 second)
        # while self.submission_times and current_time - self.submission_times[0] >= 1.0:
        #     self.submission_times.popleft()
        
        # # Check if we've reached the rate limit
        # if len(self.submission_times) >= self.current_rate:
        #     return False
        
        # # # Record this submission
        # # self.submission_times.append(current_time)
        # # self.last_submission_time = current_time
        # return True
    
    def record_submission(self, current_time: float):
        """Record when a DAG is submitted."""
        if not self.enable:
            return

        while len(self.submission_times) > 1000:
            self.submission_times.popleft()
        
        self.submission_times.append(current_time)
        self.last_submission_time = current_time
    
    def record_llmcall_completion(self, completion_time: float):
        """Record when an LLMCall completes."""
        if not self.enable:
            return

        self.total_llmcalls_completed += 1
        
        self.llmcall_completion_times.append(completion_time)
        
        # Keep only recent completions (within 2x observation window)
        max_window = self.observation_window * 2
        while (self.llmcall_completion_times and 
               completion_time - self.llmcall_completion_times[0] > max_window):
            self.llmcall_completion_times.popleft()
    
    async def update_throughput(self, current_time: float):
        """Update LLMCall throughput and adjust rate if needed."""
        if not self.enable:
            return
        
        # Only adjust periodically
        if current_time - self.last_observation_time < self.observation_window:
            return
        
        # Calculate current LLMCall throughput (LLMCalls per second)
        window_start = current_time - self.observation_window
        recent_completions = [
            t for t in self.llmcall_completion_times 
            if t >= window_start
        ]
        current_throughput = len(recent_completions) / self.observation_window
        
        # Compare with previous throughput
        if self.last_observation_time > 0:
            throughput_change = current_throughput - self.last_throughput
            
            # Adjust rate based on whether throughput improved
            if self.rate_increased_last:
                # We increased rate last time
                if throughput_change > 0:
                    # Throughput improved, keep increasing
                    new_rate = min(self.current_rate + self.adjustment_step, self.max_rate)
                    print(
                        f"[DAG_PACER] LLMCall throughput improved ({self.last_throughput:.2f} -> "
                        f"{current_throughput:.2f} calls/s), increasing DAG rate: "
                        f"{self.current_rate:.2f} -> {new_rate:.2f} DAGs/s"
                    )
                    self.current_rate = new_rate
                    self.rate_increased_last = True
                else:
                    # Throughput didn't improve, decrease
                    new_rate = max(self.current_rate - self.adjustment_step, self.min_rate)
                    print(
                        f"[DAG_PACER] LLMCall throughput didn't improve ({self.last_throughput:.2f} -> "
                        f"{current_throughput:.2f} calls/s), decreasing DAG rate: "
                        f"{self.current_rate:.2f} -> {new_rate:.2f} DAGs/s"
                    )
                    self.current_rate = new_rate
                    self.rate_increased_last = False
            else:
                # We decreased rate last time, try increasing again
                new_rate = min(self.current_rate + self.adjustment_step, self.max_rate)
                print(
                    f"[DAG_PACER] Trying higher DAG rate: {self.current_rate:.2f} -> "
                    f"{new_rate:.2f} DAGs/s (LLMCall throughput: {current_throughput:.2f} calls/s)"
                )
                self.current_rate = new_rate
                self.rate_increased_last = True
        
        # Update tracking
        self.last_throughput = current_throughput
        self.last_observation_time = current_time
        # self.current_rate = 0.05
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        if not self.enable:
            return {}
        
        current_time = time.time()
        window_start = current_time - self.observation_window
        recent_completions = [
            t for t in self.llmcall_completion_times 
            if t >= window_start
        ]
        current_throughput = len(recent_completions) / self.observation_window if self.observation_window > 0 else 0
        
        return {
            'current_dag_rate': self.current_rate,
            'llmcall_throughput': current_throughput,
            'total_llmcalls_completed': len(self.llmcall_completion_times),
            'recent_llmcalls': len(recent_completions),
            'total_llmcalls_completed': self.total_llmcalls_completed
        }


class AsyncDAGExecutor:
    def __init__(self, nodes: List[Node],
                 model: str = "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
                 agentid: Optional[str] = None,
                 message_init: str = "You are a helpful assistant"):
        self.model = model
        self.agentid = agentid
        self.nodes = nodes
        self.store = ContextStore(agentid=agentid, message_init=message_init, model=model)
        self.client = None

    def init_openai_client(self, port: int = 8000):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
            http_client=httpx.Client(timeout=None)
        )

    async def run(self) -> ContextStore:
        tasks = [asyncio.create_task(node.execute(self.store)) for node in self.nodes]
        await asyncio.gather(*tasks)
        return self.store

    async def run_openai_client(self) -> ContextStore:
        print(f"[{self.agentid}] Starting DAG execution with {len(self.nodes)} nodes")
        try:
            tasks = [asyncio.create_task(node.execute_openai_client(self.store, self.client)) for node in self.nodes]
            print(f"[{self.agentid}] Created {len(tasks)} tasks, waiting for completion...")
            await asyncio.gather(*tasks)
            print(f"[{self.agentid}] All tasks completed successfully")
            return self.store
        except Exception as e:
            print(f"[{self.agentid}] DAG execution FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Close the httpx client through the OpenAI client
            if self.client and hasattr(self.client, '_client') and self.client._client:
                self.client._client.close()


class MultiDAGExecutor:
    def __init__(self, model: str = "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct", 
                 port: int = 8000,
                 rate_controller = None):
        self.model = model
        self.port = port
        self.rate_controller = rate_controller
        self.queue = asyncio.Queue()
        self.pending_tasks = set()
        self._shutdown = False
        self._drain_on_shutdown = True


    def _check_task_exception(self, task):
        self.pending_tasks.discard(task)
        if task.cancelled():
            # print(f"=== TASK CANCELLED ===")
            # print(f"Task: {task.get_name()}")
            pass
        elif task.exception() is not None:
            exc = task.exception()
            # print(f"=== DAG EXECUTION FAILED ===")
            # print(f"Task: {task.get_name()}")
            # print(f"Exception: {exc}")
            # print(f"Exception type: {type(exc).__name__}")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            print(f"=== END ERROR ===")
        else:
            print(f"=== DAG COMPLETED SUCCESSFULLY ===")
            print(f"Task: {task.get_name()}")
            print(f"=== END SUCCESS ===")

    async def submit_dag(self, nodes: List[Node], dag_name: str):
        """Submit a new DAG to be executed."""

        if self.rate_controller and self.rate_controller.enable:
            # Wait until we can submit based on rate limit 
            current_time = time.time()
            while not self.rate_controller.can_submit(current_time):
                await asyncio.sleep(0.1)
                current_time = time.time()
            
            self.rate_controller.record_submission(current_time)

            from node import LLMCallNode
            for node in nodes:
                if isinstance(node, LLMCallNode):
                    node.rate_controller = self.rate_controller
            

        await self.queue.put((nodes, dag_name))

    async def run_forever(self):
        """Run DAGs as they are submitted, until shutdown."""

        if self.rate_controller and self.rate_controller.enable:
            asyncio.create_task(self._update_throughput_loop())

        while (not self._shutdown) or (self._drain_on_shutdown and (not self.queue.empty() or self.pending_tasks)):
            try:
                nodes, dag_name = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # allow graceful shutdown check

            dag_executor = AsyncDAGExecutor(nodes, model=self.model, agentid=dag_name)
            dag_executor.init_openai_client(port=self.port)
            task = asyncio.create_task(dag_executor.run_openai_client())
            self.pending_tasks.add(task)

            task.add_done_callback(self._check_task_exception)
        
        print("All DAGs complete. Executor exiting.")
    
    async def _update_throughput_loop(self):
        while not self._shutdown or self.pending_tasks:
            await asyncio.sleep(1.0)
            if self.rate_controller and self.rate_controller.enable:
                await self.rate_controller.update_throughput(time.time())
                stats = self.rate_controller.get_stats()
                if stats:
                    print(f"[DAG_PACER] Stats {stats}")

    def shutdown(self, drain: bool = True):
        """Request the executor loop to terminate."""
        self._shutdown = True
        self._drain_on_shutdown = drain

    async def await_all_done(self):
        """Waits for all submitted DAGs to finish"""
        while self.pending_tasks or not self.queue.empty():
            await asyncio.sleep(0.2)

def annotate_expected_durations(nodes: List[Node], is_batch_request=False):
    memo = {}

    def dfs(node: Node) -> float:
        if node.name in memo:
            return memo[node.name]

        max_time = node.expected_time if isinstance(node, ToolCallNode) else 0.0

        for downstream in node.downstream:
            downstream_cost = dfs(downstream)
            max_time = max(max_time, downstream_cost)

        if is_batch_request:
            node.metadata["kv_reuse_expected_duration_s"] = 9999999.0
        else:
            node.metadata["kv_reuse_expected_duration_s"] = max_time

        memo[node.name] = max_time
        return max_time

    for node in nodes:
        dfs(node)
    
    # mark all leaf nodes as having 9999999 expected duration
    # for eviction 
    for node in nodes:
        if not node.downstream:
            node.metadata["kv_reuse_expected_duration_s"] = 9999999.0
