import asyncio
import os
import random
import shutil
import time
from functools import partial
from lorem_text import lorem
import numpy as np
from utils import run_experiment, ensure_dir
from executor import AsyncDAGExecutor, MultiDAGExecutor, annotate_expected_durations
from node import Node, LLMCallNode, ToolCallNode, WhileLoopNode

random.seed(42)
np.random.seed(42)

def generate_variable_llm_toolcall_workload(num_requests: int, 
                                            seed: int = None, 
                                            vary_interrupts: bool = True):
    """
    Generate a list of LLM request parameter dicts:
    - request_size: Number of tokens (capped at 30,000)
    - num_interrupts: Number of tool calls
    - interrupt_len: (if vary_interrupts=False) Constant duration for all tool calls
    - interrupt_lens: (if vary_interrupts=True) List of durations for each tool call
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    
    workloads = []
    
    for _ in range(num_requests):
        # 1. Request size: log-normal distribution, capped at 30k
        # request_size = int(np.clip(
        #     np.random.lognormal(mean=np.log(15000), sigma=0.5),
        #     8000, 30000
        # ))

        request_size = 30000
        
        # 2. Number of tool call interrupts
        num_interrupts = int(np.random.choice(
            [1, 2, 3, 4, 5, 6, 8, 10],
            p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.15, 0.07, 0.03]
        ))
        
        # 3. Interrupt durations
        max_total_interrupt_time = 1 * request_size / 1000  # e.g., 15s for 30k

        if vary_interrupts:
            # Generate individual tool call durations
            durations = []
            for _ in range(num_interrupts):
                dur = float(np.clip(
                    rng.lognormal(mean=np.log(1.0), sigma=0.7),
                    0.1, 25.0
                ))
                durations.append(dur)
            
            # Normalize total time if necessary
            total = sum(durations)
            if total > max_total_interrupt_time:
                scale = max_total_interrupt_time / total
                durations = [round(d * scale, 3) for d in durations]
            else:
                durations = [round(d, 3) for d in durations]
            
            workload = {
                "request_size": request_size,
                "num_interrupts": num_interrupts,
                "interrupt_lens": durations
            }
        else:
            # Generate one constant duration
            # dur = float(np.clip(
            #     rng.lognormal(mean=np.log(1.0), sigma=0.7),
            #     0.1, 250.0
            # ))
            # dur = min(dur, max_total_interrupt_time / num_interrupts)
            num = rng.integers(1, 20)
            dur = int(num)
            workload = {
                "context": lorem.words(request_size),
                "num_interrupts": num_interrupts,
                "interrupt_len": dur
            }
        
        workloads.append(workload)
    
    return workloads


def generate_dag_constant_interrupt_len(context, num_interrupts, interrupt_len):
    dag = []
    # context = lorem.words(request_size)
    llm = LLMCallNode(context + "Please summarize the following text")
    dag.append(llm)
    for _ in range(num_interrupts):
        tool = ToolCallNode(lambda text: f"Length of the above text was: {len(text)}. Can you summarize it please?", 
                            expected_time=interrupt_len)
        tool.add_input("text", llm)

        dag.append(tool)

        llm = LLMCallNode(prompt_template="Response from tool: {tool_res}. Can you process it?")
        llm.add_input("tool_res", tool)

        dag.append(llm)

    annotate_expected_durations(dag)

    for node in dag:
        print(f"Node: {node.name}, Expected Downstream Duration: {node.metadata.get('kv_reuse_expected_duration_s', 'N/A')} seconds")
    
    return dag

async def run_dags_with_arrival_times(dags, 
                                      arrival_times):
    """
    Init multi executor 
    """

    executor = MultiDAGExecutor()
    # Start background executor loop
    asyncio.create_task(executor.run_forever())

    agent_id = 0
    for dag, arrival_time in zip(dags, arrival_times):
        time.sleep(arrival_time)
        print("submitting dag=%d" % agent_id)
        await executor.submit_dag(dag, "agent_%d" % agent_id)
        agent_id += 1 

    executor.shutdown()
    await executor.await_all_done()

    print("All finished DAGs")

async def run_dags_batch(dags):

    """
    Init multi executor 
    """

    executor = MultiDAGExecutor()
    # Start background executor loop
    asyncio.create_task(executor.run_forever())

    agent_id = 0
    for dag in dags:
        print("submitting dag=%d" % agent_id)
        await executor.submit_dag(dag, "agent_%d" % agent_id)
        agent_id += 1 

    executor.shutdown()
    await executor.await_all_done()

    print("All finished DAGs")

def execute_workload_variable_interrupts(port, 
                                         requests,
                                         arrival_times,
                                         seed=42):
    
    # rng = np.random.default_rng(seed=seed)
    dags = []
    # choose everything in variable fashion

    # rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # t = 500
    # num_events = int(rate * t)
    # exp_times = rng.exponential(scale=1/rate, size=num_events)
    
    # print(np.cumsum(exp_times))
    # arrival_times = list(exp_times)

    # print(arrival_times)
    # input()

    num_requests = len(arrival_times)
    assert len(requests) == num_requests

    # for i, req in enumerate(requests):
    #     print("id=%d %s" % (i, req))

    # input()
    # generate DAG for each request 
    for request in requests:
        dag = generate_dag_constant_interrupt_len(**request)
        dags.append(dag)


    # asyncio.run(run_dags_batch(dags))
    asyncio.run(run_dags_with_arrival_times(dags, arrival_times))

def main():

    # requests1 = generate_variable_llm_toolcall_workload(
    #     num_requests=2,
    #     seed=42,
    #     vary_interrupts=False
    # )

    # requests2 = generate_variable_llm_toolcall_workload(
    #     num_requests=2,
    #     seed=42,
    #     vary_interrupts=False
    # )

    # print(requests1)
    # print(requests2)
    
    # context1 = lorem.words(30)
    # context2 = lorem.words(30)
    # print(context1)
    # print(context2)
    # input()


    # num_requests_list = [5, 10, 15, 20]
    # num_requests_list = [15]


    # requests = generate_variable_llm_toolcall_workload(
    #     num_requests=15,
    #     seed=42,
    #     vary_interrupts=False
    # )

    # for i, req in enumerate(requests):
    #     print("id=%d %s" % (i, req))

    # input()

    # choose everything in variable fashion
    rng = np.random.default_rng(seed=42)
    # rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    rates = [0.1]
    # rates = [0.5]
    # rates = [0.6, 0.7, 0.9, 1.0]
    # rates = [0.05]
    # rates = [0.2]
    # rates = [0.2]
    # rates = [0.04]
    # rates = [0.1]
    t = 100
    enable_returning_queue = True
    enable_swap_budget = False
    swap_budget_type = "fixed"
    swap_budget_frac = 1

    for rate in rates:
        num_events = int(rate * t)
        exp_times = rng.exponential(scale=1/rate, size=num_events)
        
        print(np.cumsum(exp_times))
        arrival_times = list(exp_times)
        num_requests = len(arrival_times)

        requests = generate_variable_llm_toolcall_workload(
            num_requests=num_requests,
            seed=42,
            vary_interrupts=False
        )

        print("---- Generated requests -----")
        # for request in requests:
            

        print('Running experiment for rate=%.2f' % rate)

        env = {
            'CUDA_VISIBLE_DEVICES': '7',
            'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1'
        }
        
        exp_name = "oracle-test-57-num-rate-%d" % (int(rate * 100))
        results_path = "/vllm/vllm/elasticswap/results"
        abs_path = os.path.join("/vllm/vllm/elasticswap/results", exp_name)

        ensure_dir(abs_path)
        copied_script_name = "pipeline.py"
        shutil.copy(__file__, os.path.join(abs_path, copied_script_name))
        exec_workload_fn = partial(execute_workload_variable_interrupts,
                                   requests=requests,
                                   arrival_times=arrival_times,
                                   seed=42)

        exps = []
        for cache_ttl_value in [1, 3, 5, 7, 9, 11, 13, 15]:
        # for cache_ttl_value in [11]:
        
            exps.append(
                {
                    "execute_workload_fn": exec_workload_fn,
                    'results_path': results_path,
                    'exp_name': exp_name + '-ttl-%d' % cache_ttl_value,
                    'config_name': "swap-hint",
                    'env': env,
                    'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
                    'tp_size': 1, 
                    'pp_size': 1, 
                    'swap_space': 500,
                    'evict_token_thresh': 0,
                    'evict_token_count': 10000,
                    'enable_chunked_prefill': False,
                    'fr_policy': "default",
                    'swap_strategy': "swap-hints",
                    'block_allocator': "CpuOffloadingBlockAllocator",
                    'port': 8000,
                    'enable_returning_queue': enable_returning_queue,
                    'returning_queue_sched_policy': "prio",
                    'returning_queue_sort_freq': 10.0,
                    'enable_swap_budget': True,
                    'swap_budget_type': "fixed",
                    'swap_budget_frac': 0.0,
                    'enable_eager_evict': True,
                    'cache_pin_ttl': cache_ttl_value
                }
            )

            exps.append(
                {
                    "execute_workload_fn": exec_workload_fn,
                    'results_path': results_path,
                    'exp_name': exp_name + '-ttl-%d' % cache_ttl_value,
                    'config_name': "swap-lru",
                    'env': env,
                    'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
                    'tp_size': 1, 
                    'pp_size': 1, 
                    'swap_space': 500,
                    'evict_token_thresh': 0,
                    'evict_token_count': 7000,
                    'enable_chunked_prefill': False,
                    'fr_policy': "default",
                    'swap_strategy': "swap-lru",
                    'block_allocator': "CpuOffloadingBlockAllocator",
                    'port': 8000,
                    'enable_returning_queue': enable_returning_queue,
                    'returning_queue_sched_policy': "prio",
                    'returning_queue_sort_freq': 10.0,
                    'enable_swap_budget': True,
                    'swap_budget_type': "fixed",
                    'swap_budget_frac': 0.0,
                    'enable_eager_evict': True,
                    'cache_pin_ttl': cache_ttl_value
                }
            )

        for exp in exps:
            run_experiment(**exp)


if __name__ == "__main__":
    main()
    


