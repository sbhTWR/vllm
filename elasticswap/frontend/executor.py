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
        tasks = [asyncio.create_task(node.execute_openai_client(self.store, self.client)) for node in self.nodes]
        await asyncio.gather(*tasks)
        return self.store


class MultiDAGExecutor:
    def __init__(self, model: str = "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct", 
                 port: int = 8000):
        self.model = model
        self.port = port
        self.queue = asyncio.Queue()
        self.pending_tasks = set()
        self._shutdown = False

    async def submit_dag(self, nodes: List[Node], dag_name: str):
        """Submit a new DAG to be executed."""
        await self.queue.put((nodes, dag_name))

    async def run_forever(self):
        """Run DAGs as they are submitted, until shutdown."""
        while not self._shutdown or not self.queue.empty() or self.pending_tasks:
            try:
                nodes, dag_name = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # allow graceful shutdown check

            dag_executor = AsyncDAGExecutor(nodes, model=self.model, agentid=dag_name)
            dag_executor.init_openai_client(port=self.port)
            task = asyncio.create_task(dag_executor.run_openai_client())
            self.pending_tasks.add(task)

            task.add_done_callback(self.pending_tasks.discard)
        
        print("All DAGs complete. Executor exiting.")

    def shutdown(self):
        """Request the executor loop to terminate."""
        self._shutdown = True

    async def await_all_done(self):
        """Waits for all submitted DAGs to finish"""
        while self.pending_tasks or not self.queue.empty():
            await asyncio.sleep(0.5)

def annotate_expected_durations(nodes: List[Node]):
    memo = {}

    def dfs(node: Node) -> float:
        if node.name in memo:
            return memo[node.name]

        max_time = node.expected_time if isinstance(node, ToolCallNode) else 0.0

        for downstream in node.downstream:
            downstream_cost = dfs(downstream)
            max_time = max(max_time, downstream_cost)

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
