import asyncio
import json
import openai
from typing import Dict, Any, Union, Callable, Optional
from uuid import uuid4
import httpx
import asyncio

class ContextStore:
    def __init__(self, agentid: Optional[str] = None, 
                 message_init: Optional[list[dict]] = "You are a helpful assistant",
                 model: Optional[str] = "gpt-4"):
        self.agentid = agentid or f"agent_{uuid4().hex[:6]}"
        self.futures: dict[str, asyncio.Future] = {}
        self.messages: list[dict] = []  # Optional: for OpenAI message history
        self.model = model

        if message_init:
            self.messages.append(
                {"role": "system", "content": message_init}
            )

    def get_future(self, name: str) -> asyncio.Future:
        if name not in self.futures:
            self.futures[name] = asyncio.Future()
        return self.futures[name]

    def set_value(self, name: str, value: any):
        fut = self.get_future(name)
        if not fut.done():
            fut.set_result(value)

    async def get_value(self, name: str) -> any:
        fut = self.get_future(name)
        return await fut

    def append(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def construct_openai_message(self) -> dict:
        return self.messages

class Node:
    def __init__(self, name: Optional[str] = None):
        self.name = name or f"{self.__class__.__name__}_{uuid4().hex[:6]}"
        self.inputs: Dict[str, Union["Node", Any]] = {}
        self.downstream = []
        self.metadata = {}

    def add_input(self, key: str, value: Union["Node", Any]):
        self.inputs[key] = value
        if isinstance(value, Node):
            value.downstream.append(self)

    async def resolve_inputs(self, store: ContextStore) -> Dict[str, Any]:
        resolved = {}
        for k, v in self.inputs.items():
            if isinstance(v, Node):
                resolved[k] = await store.get_value(v.name)
            else:
                resolved[k] = v
        return resolved

    async def execute(self, store: ContextStore):
        raise NotImplementedError
    
    async def execute_openai_server(self, store: ContextStore, 
                                    client: openai.OpenAI):
        raise NotImplementedError


class LLMCallNode(Node):
    def __init__(self, 
                 prompt_template: str = None,
                 prompt_token_ids: list[int] = None,  # NEW: Support raw tokens
                 target_output_tokens: int = 128,     # NEW: Control output length
                 turn_idx: int = None,                 # NEW: For logging
                 llm_name: str = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = prompt_template
        self.prompt_token_ids = prompt_token_ids
        self.target_output_tokens = target_output_tokens
        self.turn_idx = turn_idx
        self.llm_name = llm_name

    async def execute(self, store: ContextStore):
        inputs = await self.resolve_inputs(store)
        user_msg = self.prompt_template.format(**inputs)

        # Append prompt to context
        store.append("user", user_msg)

        # Simulated OpenAI call
        await asyncio.sleep(0.1)
        response = f"[LLM:{self.llm_name}] Response to: {user_msg}"

        # Append assistant reply
        store.append("assistant", response)
        store.set_value(self.name, response)
    
    async def execute_openai_client(self, store: ContextStore, 
                                    client: Optional[openai.OpenAI] = None):
        
        assert client is not None
        
        # NEW: Support for raw token IDs
        if self.prompt_token_ids is not None:
            # IMPORTANT: Wait for dependencies (e.g., previous turn) before executing
            await self.resolve_inputs(store)
            
            payload = {
                "id": store.agentid,
                "type": "append",
                "hints": self.metadata,
            }
            print(f"[{store.agentid}] Turn {self.turn_idx}: Starting with {len(self.prompt_token_ids)} token IDs")
            print(f"[{store.agentid}] sending hints: {json.dumps(payload['hints'], indent=2)}")
            try:
                response = await asyncio.to_thread(
                    client.completions.create,
                    model=store.model,
                    prompt=self.prompt_token_ids,  # Use completions API with token IDs
                    max_tokens=self.target_output_tokens,
                    temperature=0,
                    user=json.dumps({
                        "id": store.agentid,
                        "type": "append",
                        "hints": self.metadata
                    })
                )
                result_text = response.choices[0].text
                print(f"[{store.agentid}] Turn {self.turn_idx}: Completed, output: {len(result_text)} chars")
                
            except Exception as e:
                print(f"[{store.agentid}] Turn {self.turn_idx}: FAILED: {e}")
                raise
            
            store.set_value(self.name, result_text)
            
        else:
            # EXISTING: Text-based prompt path
            inputs = await self.resolve_inputs(store)
            user_msg = self.prompt_template.format(**inputs)

            # Append prompt to context
            store.append("user", user_msg)
            payload = {
                "id": store.agentid,
                "type": "append",
                "hints": self.metadata,
            }

            print(f"[{store.agentid}] Starting OpenAI API call for {self.name}")
            print(f"[{store.agentid}] sending hints: {json.dumps(payload['hints'], indent=2)}")
            
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    messages=store.construct_openai_message(),
                    model=store.model,
                    temperature=0,
                    max_tokens=self.target_output_tokens,
                    user=json.dumps({
                        "id": store.agentid,
                        "type": "append",
                        "hints": self.metadata
                    })
                )
                print(f"[{store.agentid}] OpenAI API call completed for {self.name}")
            except Exception as e:
                print(f"[{store.agentid}] OpenAI API call FAILED for {self.name}: {e}")
                raise

            response = response.choices[0].message.content

            # Append assistant reply
            store.append("assistant", response)
            store.set_value(self.name, response)
            print(f"[{store.agentid}] Node {self.name} completed successfully")


class ToolCallNode(Node):
    def __init__(self, tool_fn: Callable, 
                 name: Optional[str] = None,
                 expected_time: Optional[float] = None
                ):
        super().__init__(name=name)
        self.tool_fn = tool_fn
        self.expected_time = expected_time

    async def execute(self, store: ContextStore):
        inputs = await self.resolve_inputs(store)

        print(inputs)

        if self.expected_time:
            # Simulate expected processing time
            await asyncio.sleep(self.expected_time)
        else:
            await asyncio.sleep(0.05)
        result = self.tool_fn(**inputs)
        # store.append("tool", f"Tool result: {result}")
        store.set_value(self.name, result)

    async def execute_openai_client(self, store: ContextStore,
                                    client: Optional[openai.OpenAI] = None):
        inputs = await self.resolve_inputs(store)

        print(f"[{store.agentid}] Starting ToolCall {self.name} with inputs: {inputs}")

        if self.expected_time:
            print(f"[{store.agentid}] ToolCall {self.name} sleeping for {self.expected_time} seconds")
            # Simulate expected processing time
            await asyncio.sleep(self.expected_time)
        else:
            await asyncio.sleep(0.05)
        
        print(f"[{store.agentid}] ToolCall {self.name} executing function")
        result = self.tool_fn(**inputs)
        # store.append("tool", f"Tool result: {result}")
        store.set_value(self.name, result)
        print(f"[{store.agentid}] ToolCall {self.name} completed successfully")


class WhileLoopNode(Node):
    def __init__(self, condition_fn: Callable[[Dict[str, Any]], bool],
                 body_nodes: list[Node], max_iters: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.condition_fn = condition_fn
        self.body_nodes = body_nodes
        self.max_iters = max_iters

    async def execute(self, context: Dict[str, asyncio.Future]):
        iteration = 0
        context[self.name].set_result(f"LOOP_STARTED")
        last_outputs = {}

        while self.condition_fn(context) and iteration < self.max_iters:
            # Create a local context for body
            local_context = {**context}
            local_futures = {node.name: asyncio.Future() for node in self.body_nodes}
            tasks = [asyncio.create_task(node.execute(local_futures)) for node in self.body_nodes]
            await asyncio.gather(*tasks)

            # Copy back relevant outputs to main context
            context.update({k: v for k, v in local_futures.items()})
            last_outputs = {k: v.result() for k, v in local_futures.items()}

            iteration += 1

        context[self.name].set_result(f"LOOP_DONE_{iteration}_iters")



