import asyncio
from typing import List, Optional
from openai import OpenAI
import httpx
from node import Node, ContextStore, ToolCallNode


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
                 port: int = 8000):
        self.model = model
        self.port = port
        self.queue = asyncio.Queue()
        self.pending_tasks = set()
        self._shutdown = False
        self._drain_on_shutdown = True


    def _check_task_exception(self, task):
        self.pending_tasks.discard(task)
        if task.cancelled():
            print(f"=== TASK CANCELLED ===")
            print(f"Task: {task.get_name()}")
        elif task.exception() is not None:
            exc = task.exception()
            print(f"=== DAG EXECUTION FAILED ===")
            print(f"Task: {task.get_name()}")
            print(f"Exception: {exc}")
            print(f"Exception type: {type(exc).__name__}")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            print(f"=== END ERROR ===")
        else:
            print(f"=== DAG COMPLETED SUCCESSFULLY ===")
            print(f"Task: {task.get_name()}")
            print(f"=== END SUCCESS ===")

    async def submit_dag(self, nodes: List[Node], dag_name: str):
        """Submit a new DAG to be executed."""
        await self.queue.put((nodes, dag_name))

    async def run_forever(self):
        """Run DAGs as they are submitted, until shutdown."""
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
