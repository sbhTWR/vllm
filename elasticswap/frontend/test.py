from executor import AsyncDAGExecutor, MultiDAGExecutor, annotate_expected_durations
import asyncio
from lorem_text import lorem
from node import Node, LLMCallNode, ToolCallNode, WhileLoopNode


# # DAG 1
# llm1 = LLMCallNode("Capital of {country}?", llm_name="gpt-4")
# llm1.add_input("country", "France")

# tool1 = ToolCallNode(lambda text: f"Length: {len(text)}")
# tool1.add_input("text", llm1)

# dag1 = [llm1, tool1]

# # DAG 2
# llm2 = LLMCallNode("Population of {country}?", llm_name="gpt-4")
# llm2.add_input("country", "Germany")

# tool2 = ToolCallNode(lambda text: f"Uppercase: {text.upper()}")
# tool2.add_input("text", llm2)

# dag2 = [llm2, tool2]

# # Run both DAGs
# executor = MultiDAGExecutor([dag1, dag2])
# import asyncio
# contexts = asyncio.run(executor.run())

# # View results
# for i, store in enumerate(contexts):
#     print(f"\n--- DAG {i+1} Chat History ---")
#     for msg in store.messages:
#         print(f"{msg['role']}: {msg['content']}")

"""
async def main():
    multi_executor = MultiDAGExecutor()

    # Start background executor loop
    asyncio.create_task(multi_executor.run_forever())

    # Dynamically submit DAGs
    dag1 = [...]  # list[Node]
    dag2 = [...]  # list[Node]

    await multi_executor.submit_dag(dag1, "agent_1")
    await multi_executor.submit_dag(dag2, "agent_2")

    await asyncio.sleep(10)  # let DAGs run
    multi_executor.shutdown()

asyncio.run(main())
"""

async def run_workloads():
    request_size = 10  # Number of words for the request
    num_interrupts = 3
    interrupt_time = 1.5
    context = lorem.words(request_size)
    model = "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct"

    """
    Create DAG
    """
    dag = []
    llm = LLMCallNode(context + "Please summarize the following text")
    dag.append(llm)
    for _ in range(num_interrupts):
        tool = ToolCallNode(lambda text: f"Length: {len(text)}", 
                            expected_time=interrupt_time)
        tool.add_input("text", llm)

        dag.append(tool)

        llm = LLMCallNode(prompt_template="Response from tool: {tool_res}")
        llm.add_input("tool_res", tool)

        dag.append(llm)

    annotate_expected_durations(dag)

    for node in dag:
        print(f"Node: {node.name}, Expected Downstream Duration: {node.metadata.get('kv_reuse_expected_duration_s', 'N/A')} seconds")

    """
    Init multi executor 
    """

    # executor = MultiDAGExecutor([dag])
    executor = MultiDAGExecutor()
    # Start background executor loop
    asyncio.create_task(executor.run_forever())
    await executor.submit_dag(dag, "agent_1")
    executor.shutdown()

    await executor.await_all_done()

    print("All finished DAGs")

    # # View results
    # for i, store in enumerate(contexts):
    #     print(f"\n--- DAG {i+1} Chat History ---")
    #     print(store.messages)

asyncio.run(run_workloads())