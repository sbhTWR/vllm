import asyncio
import os
import json
import random
import shutil
import time
from functools import partial
from lorem_text import lorem
import numpy as np
from utils import run_experiment, ensure_dir
from extract_trace_templates import generate_claude_trace_workload
from executor import AsyncDAGExecutor, MultiDAGExecutor, annotate_expected_durations
from node import Node, LLMCallNode, ToolCallNode, WhileLoopNode

DEBUG = os.environ.get('DEBUG', '0') == '1'

# ADD THIS NEW VARIABLE
PREFILL_ONLY = False
MAX_OUTPUT_TOKENS = 1 if PREFILL_ONLY else 128  # 0 = prefill only, 128 = normal generation

random.seed(42)
np.random.seed(42)

# MUST be set BEFORE importing vllm or transformers
os.environ["HF_HOME"] = "/vllm/models/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/vllm/models/.cache"
os.environ["HF_HUB_CACHE"] = "/vllm/models/.cache"
os.environ["HF_DATASETS_CACHE"] = "/vllm/models/.cache"

# def sample_toolcall_duration_fixed(
#     rng=None,
#     p_long=0.15,
#     interrupt_length=None
# ):
#     rng = np.random.default_rng() if rng is None else rng
#     is_long = rng.random() < p_long

#     if not is_long:
#         num = rng.integers(1, 100)
#         num = num / 100.0
#     else:
#         num = rng.integers(1, 5)
    
#     return float(num), is_long

def sample_toolcall_duration_fixed(
    rng=None,
    p_long=0.15,
    interrupt_length=None
):
    rng = np.random.default_rng() if rng is None else rng
    is_long = rng.random() < p_long

    if not is_long:
        num = rng.integers(1, 900)
        num = num / 100.0
    else:
        num = rng.integers(8, 15)
    
    return float(num), is_long

def sample_toolcall_duration(
    rng=None,
    p_long=0.15,
    # short (baseline) distribution
    short_mu=1.0, short_sigma=0.7, short_clip=(0.1, 20.0),
    # long distribution
    long_mu=15.0, long_sigma=0.9, long_clip=(20.0, 120.0),
    # optional: enforce minimum for long calls
    interrupt_length=None
):
    """
    Sample a ToolCall duration with a two-component mixture.
    Returns: (duration_seconds: float, is_long_call: bool)
    """
    rng = np.random.default_rng() if rng is None else rng
    is_long = rng.random() < p_long

    if not is_long:
        dur = float(np.clip(
            rng.lognormal(mean=np.log(short_mu), sigma=short_sigma),
            short_clip[0], short_clip[1]
        ))
        return dur, False

    low = long_clip[0]
    if interrupt_length is not None:
        low = max(low, float(interrupt_length))

    dur = float(np.clip(
        rng.lognormal(mean=np.log(long_mu), sigma=long_sigma),
        low, long_clip[1]
    ))
    return dur, True

def generate_variable_llm_toolcall_with_variable_request_size(num_requests: int, 
                                                              seed: int = None, 
                                                              vary_interrupts: bool = True):
    """
    Generate LLM requests in the following way:
    - request_size: bi-modal: 1k and 30k
    Smaller requests have smaller duration of ToolCalls but have more interrupts
    Larger requests have larger duration of ToolCalls but have fewer interrupts
    - num_interrupts: 
    - interrupt_lens: 
    - p_large_request: probability of generating a large request
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)

    p_large_request = 0.5
    is_large_request = False
    workloads = []

    if rng.random() < p_large_request:
        is_large_request = True
    else:
        is_large_request = False

    for i in range(num_requests):
        if is_large_request:
            request_size = 30000
            num_interrupts = 5
            interrupt_lens = [10.0] * num_interrupts
        else:
            request_size = 1000
            num_interrupts = 20
            interrupt_lens = [1.0] * num_interrupts

        # 1. Request size: log-normal distribution, capped at 30k
        # request_size = int(np.clip(
        #     np.random.lognormal(mean=np.log(15000), sigma=0.5),
        #     8000, 30000
        # ))

        workload = {
            "context": lorem.words(request_size) + " Respond by repeating the text and then summarizing it.",
            "num_interrupts": num_interrupts,
            "interrupt_lens": interrupt_lens
        }

        workloads.append(workload)

    return workloads

def generate_variable_llm_toolcall_workload(num_requests: int, 
                                            seed: int = None, 
                                            vary_interrupts: bool = True,
                                            num_interrupts: int = None):
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

    num_interrupts_list = rng.integers(1, 10, size=num_requests)
    
    for i in range(num_requests):
        # 1. Request size: log-normal distribution, capped at 30k
        # request_size = int(np.clip(
        #     np.random.lognormal(mean=np.log(15000), sigma=0.5),
        #     8000, 30000
        # ))

        # change this to 60k
        request_size = 60000
        
        # 2. Number of tool call interrupts
        # if num_interrupts is None:
        #     num_interrupts = int(np.random.choice(
        #     [1, 2, 3, 4, 5, 6, 8, 10],
        #     p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.15, 0.07, 0.03]
        # ))
            # num_interrupts = rng.integers(40, 60)
        num_interrupts = num_interrupts_list[i]
        
        # num_interrupts = 20

        # 3. Interrupt durations
        # max_total_interrupt_time = 1 * request_size / 1000  # e.g., 15s for 30k
        max_total_interrupt_time = 200.0  # seconds

        if vary_interrupts:
            # Generate individual tool call durations
            durations = []
            # generate occasional long tool calls 
            long_tool_call_prob = 0.2

            for i in range(num_interrupts):
                # if rng.random() < long_tool_call_prob:
                #     dur = float(np.clip(
                #         rng.lognormal(mean=np.log(1.0), sigma=0.7),
                #         0.1, 20.0
                #     ))
                dur, is_long = sample_toolcall_duration_fixed(
                    rng=rng,
                    p_long=long_tool_call_prob,
                    interrupt_length=max_total_interrupt_time / num_interrupts
                )
  
                durations.append(dur)
            
            # Normalize total time if necessary
            total = sum(durations)
            if total > max_total_interrupt_time:
                scale = max_total_interrupt_time / total
                durations = [round(d * scale, 3) for d in durations]
            else:
                durations = [round(d, 3) for d in durations]
            
            workload = {
                "context": lorem.words(request_size) + " Respond by repeating the text and then summarizing it.",
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
            if i % 2 == 0:
                num = rng.integers(1, 300)
                num = num // 100
            else:
                num = rng.integers(3, 10)

            # num = rng.integers(1, 20)
            dur = int(num)
            workload = {
                "context": lorem.words(request_size) + " Respond by repeating the text and then summarizing it.",
                "num_interrupts": num_interrupts,
                "interrupt_len": dur
            }
        
        workloads.append(workload)
    
    return workloads



def generate_variable_llm_toolcall_workload_small_interrupts(num_requests: int, 
                                            seed: int = None, 
                                            num_interrupts: int = None):
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
    
    for i in range(num_requests):
        # 1. Request size: log-normal distribution, capped at 30k
        # request_size = int(np.clip(
        #     np.random.lognormal(mean=np.log(15000), sigma=0.5),
        #     8000, 30000
        # ))

        request_size = 30000
        
        # 2. Number of tool call interrupts
        if num_interrupts is None:
            num_interrupts = int(np.random.choice(
            [1, 2, 3, 4, 5, 6, 8, 10],
            p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.15, 0.07, 0.03]
        ))
        
        # num = 0.5
        # num = rng.integers(1, 20)
        dur = 10
        durs = [dur] * num_interrupts
        workload = {
            "context": lorem.words(request_size) + " Respond by repeating the text and then summarizing it.",
            "num_interrupts": num_interrupts,
            "interrupt_lens": durs
        }
        
        workloads.append(workload)
    
    return workloads


def generate_variable_llm_toolcall_workload_multi_tenant(num_requests: int, 
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
    
    # this is similar to single tenent case except that 
    # depending on the a probability, the request can either be single llmcall
    # or a sequence of llmcalls and toolcalls

    workloads = []

    for i in range(num_requests):
        if rng.random() < 0.3:
            workload = generate_variable_llm_toolcall_workload_small_interrupts(num_requests=1, seed=seed)
            workloads.extend(workload)
        else:
            workload = generate_variable_llm_toolcall_workload(num_requests=1, seed=seed, vary_interrupts=vary_interrupts, num_interrupts=0)
            workloads.extend(workload)
    
    return workloads

def generate_dag_constant_interrupt_len(context, num_interrupts, interrupt_len):
    dag = []
    # context = lorem.words(request_size)
    llm = LLMCallNode(context + " Please ignore the previous text and generate a long story.", 
    target_output_tokens=MAX_OUTPUT_TOKENS)
    dag.append(llm)
    for _ in range(num_interrupts):
        tool = ToolCallNode(lambda text: f"Length of the above text was: {len(text)}. Please ignore the previous text and generate a long story.", 
                            expected_time=interrupt_len)
        tool.add_input("text", llm)

        dag.append(tool)

        llm = LLMCallNode(prompt_template="Response from tool: {tool_res}. Please ignore the previous text and generate a long story.",
        target_output_tokens=MAX_OUTPUT_TOKENS)
        llm.add_input("tool_res", tool)

        dag.append(llm)

    annotate_expected_durations(dag)

    for node in dag:
        print(f"Node: {node.name}, Expected Downstream Duration: {node.metadata.get('kv_reuse_expected_duration_s', 'N/A')} seconds")
    
    return dag


def generate_dag_variable_interrupt_len(context, num_interrupts, interrupt_lens, is_batch_request=False):
    dag = []
    # context = lorem.words(request_size)
    llm = LLMCallNode(context + " Please ignore the previous text and generate a long story.", 
            target_output_tokens=MAX_OUTPUT_TOKENS)
    dag.append(llm)
    for i in range(num_interrupts):
        tool = ToolCallNode(lambda text: f"Length of the above text was: {len(text)}. Please ignore the previous text and generate a long story.", 
                            expected_time=interrupt_lens[i])
        tool.add_input("text", llm)

        dag.append(tool)

        llm = LLMCallNode(prompt_template="Response from tool: {tool_res}. Please ignore the previous text and generate a long story.",
        target_output_tokens=MAX_OUTPUT_TOKENS)
        llm.add_input("tool_res", tool)

        dag.append(llm)

    annotate_expected_durations(dag, is_batch_request=is_batch_request)

    for node in dag:
        print(f"Node: {node.name}, Expected Downstream Duration: {node.metadata.get('kv_reuse_expected_duration_s', 'N/A')} seconds")
    
    return dag

async def run_dags_with_arrival_times(dags, 
                                      arrival_times,
                                      dag_names=None,  # NEW: Optional custom names
                                      port=8000,
                                      model="Qwen/Qwen2.5-Coder-32B-Instruct",
                                      wait_for_all_done=False):
    """
    Init multi executor 
    """
    executor = MultiDAGExecutor(port=port, model=model)
    # Start background executor loop
    executor_task = asyncio.create_task(executor.run_forever())

    try:
        agent_id = 0
        for dag, arrival_time in zip(dags, arrival_times):
            print("sleeping for %f seconds" % arrival_time)
            # time.sleep(arrival_time)
            await asyncio.sleep(arrival_time)
            
            # Use custom dag_name if provided, otherwise use agent_%d
            if dag_names is not None:
                dag_name = dag_names[agent_id]
            else:
                dag_name = "agent_%d" % agent_id
            
            print("submitting dag=%s" % dag_name)
            try:
                await executor.submit_dag(dag, dag_name)
            except Exception as e:
                print(f"Error submitting DAG: {e}")
                
            agent_id += 1 

        executor.shutdown(drain=wait_for_all_done)
        # if wait_for_all_done:
        #     await executor.await_all_done()

        await executor_task
        print("All finished DAGs")
    
    except Exception as e:
        print(f"Error: {e}")
        executor.shutdown(drain=False)
        executor_task.cancel()
        try:
            await executor_task
        except asyncio.CancelledError:
            pass

    # except asyncio.CancelledError:
    #     print("Executor timed out!")
    #     executor.shutdown(drain=False)
    #     executor_task.cancel()
    #     try:
    #         await executor_task
    #     except asyncio.CancelledError:
    #         pass 
    #     raise
       

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

async def run_with_timeout(async_func, timeout, **kwargs):
    try:
        await asyncio.wait_for(async_func(**kwargs), timeout)
        print(f"Task completed.")
    except asyncio.TimeoutError:
        print("Task timed out using timeout!")
    
    finally:
        print('Shutting down default executor')
        loop = asyncio.get_running_loop()
        if hasattr(loop, '_default_executor') and loop._default_executor:
            loop._default_executor.shutdown(wait=False)

def execute_workload_variable_interrupts(
                                         requests,
                                         arrival_times,
                                         seed=42,
                                         multi_tenant=False,
                                         port=8000, 
                                         model="Qwen/Qwen2.5-Coder-32B-Instruct",
                                         wait_for_all_done=False,
                                         timeout=3):
    
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
    for i, request in enumerate(requests):
        is_batch_request = (i % 2 == 0) and multi_tenant
        if 'interrupt_lens' in request:
            dag = generate_dag_variable_interrupt_len(**request, is_batch_request=is_batch_request)
        else:
            dag = generate_dag_constant_interrupt_len(**request, is_batch_request=is_batch_request)
        dags.append(dag)

    if DEBUG:
        return

    # asyncio.run(run_dags_batch(dags))
    # asyncio.run(run_dags_with_arrival_times(
    #                 dags, 
    #                 arrival_times, 
    #                 port,
    #                 wait_for_all_done=wait_for_all_done))
    # also proide args for async_func
    asyncio.run(run_with_timeout(run_dags_with_arrival_times, 
                                 timeout=timeout,
                                 dags=dags,
                                 arrival_times=arrival_times,
                                 port=port,
                                 model=model,
                                 wait_for_all_done=wait_for_all_done))



def execute_workload_claude_traces(
    num_requests,
    arrival_rate=0.5,
    seed=42,
    port=8000,
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    wait_for_all_done=False,
    timeout=300,
    claude_dataset_dir="/vllm/vllm/elasticswap/toolcall_dataset_claude",
    results_dir=None  # NEW: For saving metadata
):
    """
    Execute workload from Claude traces.
    
    Uses raw token IDs and actual tool execution patterns from Claude dataset.
    
    DETERMINISM GUARANTEE:
    Multiple executions with the same seed will generate identical requests
    (same token IDs, arrival times, and execution order). A verification hash
    is printed to confirm determinism across runs.
    """
    # Generate workload
    dags, dag_names, arrival_times, requests_meta = generate_claude_trace_workload(
        num_requests=num_requests,
        claude_dataset_dir=claude_dataset_dir,
        arrival_rate=arrival_rate,
        seed=seed,
        prefill_only=PREFILL_ONLY
    )
    
    # Save request metadata if results_dir is provided
    if results_dir is not None:
        # Compute template diversity statistics
        from collections import Counter
        template_counts = Counter()
        for meta in requests_meta:
            template_counts[meta['template_id']] += 1
        
        meta_file = os.path.join(results_dir, 'requests_meta.json')
        with open(meta_file, 'w') as f:
            json.dump({
                "arrival_times": arrival_times,
                "requests_meta": requests_meta,
                "template_diversity": {
                    "total_templates_available": len(set(m['template_id'] for m in requests_meta)),
                    "unique_templates_used": len(template_counts),
                    "template_usage": dict(template_counts),
                    "most_common": template_counts.most_common(10)
                }
            }, f, indent=2)
        print(f"\n✓ Saved request metadata to {meta_file}")
    
    if DEBUG:
        return
    
    # Run with timeout
    asyncio.run(run_with_timeout(
        run_dags_with_arrival_times,
        timeout=timeout,
        dags=dags,
        arrival_times=arrival_times,
        dag_names=dag_names,  # Pass request IDs as dag names
        port=port,
        model=model,
        wait_for_all_done=wait_for_all_done
    ))


def execute_workload_claude_traces_precomputed(workload, 
                                              port=8000,
                                              model="Qwen/Qwen2.5-Coder-32B-Instruct",
                                              wait_for_all_done=False,
                                              timeout=120,
                                              results_dir=None):
    """
    Execute workload with pre-generated DAGs to avoid construction overhead
    """
    dags = workload['dags']
    dag_names = workload['dag_names']
    arrival_times = workload['arrival_times']
    requests_meta = workload['requests_meta']
    
    # Save request metadata if results_dir is provided
    if results_dir is not None:
        from collections import Counter
        template_counts = Counter()
        for meta in requests_meta:
            template_counts[meta['template_id']] += 1
        
        meta_file = os.path.join(results_dir, 'requests_meta.json')
        with open(meta_file, 'w') as f:
            json.dump({
                "arrival_times": arrival_times,
                "requests_meta": requests_meta,
                "template_diversity": {
                    "total_templates_available": len(set(m['template_id'] for m in requests_meta)),
                    "unique_templates_used": len(template_counts),
                    "template_usage": dict(template_counts),
                    "most_common": template_counts.most_common(10)
                }
            }, f, indent=2)
        print(f"\n✓ Saved request metadata to {meta_file}")
    
    if DEBUG:
        return
    
    # Run with timeout - DAGs are already constructed!
    asyncio.run(run_with_timeout(
        run_dags_with_arrival_times,
        timeout=timeout,
        dags=dags,
        arrival_times=arrival_times,
        dag_names=dag_names,
        port=port,
        model=model,
        wait_for_all_done=wait_for_all_done
    ))

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
    # rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # rates = [0.01, 0.02, 0.03, 0.04, 0.05, 
    #         0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 
    #         0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # rates = [0.2]

    # rates = [2.0, 3.0, 4.0, 5.0]
    # rates = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    rates = [0.5]
    # rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.2, 0.3, 0.4, 0.5]
    # rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.1]
    # rates = [0.01]
    # rates = [0.1]
    # rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.02]
    # rates = [0.04]
    batch_sizes = [64]
    # batch_sizes = [1, 2, 4, 8, 16]
    # rates = [0.02]
    # rates = [0.06, 0.07, 0.08, 0.09]


    # rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]

    # rates = [0.2]
    # rates = [0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.2, 0.3, 0.4, 0.5]
    # rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.4, 0.5, 0.6]
    # rates = [0.5, 0.7, 0.9, 1.0]
    # rates = [0.6, 0.7, 0.8, 0.9, 1.0]
    # rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,\
        #  1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # rates = [0.5]
    # rates = [0.6, 0.7, 0.9, 1.0]
    # rates = [0.05]
    # rates = [0.2]
    # rates = [0.2]
    # rates = [0.04]
    # rates = [0.1]
    t = 600
    # t = 100
    # enable_returning_queue = True
    enable_swap_budget = False
    swap_budget_type = "fixed"
    swap_budget_frac = 1
    cache_ttl_value = 0
    pinned_memory_frac = 0.0
    enable_cache_heirarchy = False
    multi_tenant = False
    port = 8001
    max_num_seqs = 200
    cuda_device = '6,7'
    wait_for_all_done = False
    max_num_batched_tokens = 512
    timeout = t
    model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    rope_scaling = '{"rope_type":"yarn","factor":7.0,"original_max_position_embeddings":32768}'
    max_model_len = 200000
    tp_size = 2
    pp_size = 1
    enable_returning_queue = True


    # workloads = {}
    # for rate in rates:

    for rate in rates:
        num_events = int(rate * t)
        exp_times = rng.exponential(scale=1/rate, size=num_events)
        
        print(np.cumsum(exp_times))
        arrival_times = list(exp_times)
        num_requests = len(arrival_times)


        num_events = int(rate * t)
        exp_times = rng.exponential(scale=1/rate, size=num_events)
        arrival_times = list(exp_times)
        num_requests = len(arrival_times)
        
        print(f"Generating workload for rate={rate}, num_requests={num_requests}")
        
        # Generate workload once per rate
        dags, dag_names, arrival_times, requests_meta = generate_claude_trace_workload(
            num_requests=num_requests,
            claude_dataset_dir="/vllm/vllm/elasticswap/toolcall_dataset_claude",
            arrival_rate=rate,
            seed=42,
            prefill_only=PREFILL_ONLY
        )
        
        workload = {
            'dags': dags,
            'dag_names': dag_names,
            'arrival_times': arrival_times,
            'requests_meta': requests_meta,
            'num_requests': num_requests
        }


        
        # # uncomment this to use the single tenant workload
        # requests = generate_variable_llm_toolcall_workload(
        #     num_requests=num_requests,
        #     seed=42,
        #     vary_interrupts=True,
        #     num_interrupts=None
        # )

        # uncomment this to use the multi tenant workload
        # requests = generate_variable_llm_toolcall_workload_multi_tenant(
        #     num_requests=num_requests,
        #     seed=42,
        #     vary_interrupts=True
        # )

        # requests = generate_variable_llm_toolcall_with_variable_request_size(
        #     num_requests=num_requests,
        #     seed=42,
        #     vary_interrupts=True
        # )

        print("---- Generated requests -----")
        # for request in requests:
            

        print('Running experiment for rate=%.2f' % rate)
        env = {
            'CUDA_VISIBLE_DEVICES': str(cuda_device),
            'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1'
        }
        
        exp_name = "oracle-test-171-claude-num-rate-%d" % (int(rate * 100))
        results_path = "/vllm/vllm/elasticswap/results"
        abs_path = os.path.join("/vllm/vllm/elasticswap/results", exp_name)

        ensure_dir(abs_path)
        copied_script_name = "pipeline.py"
        shutil.copy(__file__, os.path.join(abs_path, copied_script_name))

        exec_workload_fn = partial(execute_workload_claude_traces_precomputed,
                            workload=workload,
                            port=port,
                            model=model,
                            wait_for_all_done=wait_for_all_done,
                            timeout=timeout,
                            results_dir=abs_path
                    )
        # requests_meta = []

        # for i, request in enumerate(requests):
        #     req_size = len(request['context'].split()) if 'context' in request else None
        #     if "interrupt_lens" in request:
        #         num_interrupts = len(request['interrupt_lens'])
        #         durations = list(request['interrupt_lens'])
        #     else:
        #         num_interrupts = request['num_interrupts']
        #         durations = [request['interrupt_len']] * num_interrupts

        #     requests_meta.append({
        #         'id': i,
        #         'request_size': req_size,
        #         'num_interrupts': num_interrupts,
        #         'interrupt_durations': durations
        #         })
            
        # with open(os.path.join(abs_path, 'requests_meta.json'), 'w') as f:
        #     json.dump({
        #         "arrival_times": arrival_times,
        #         "requests_meta": requests_meta
        #     }, f, indent=2)

        # exec_workload_fn = partial(execute_workload_variable_interrupts,
        #                            requests=requests,
        #                            arrival_times=arrival_times,
        #                            seed=42,
        #                            multi_tenant=multi_tenant,
        #                            port=port,
        #                            wait_for_all_done=wait_for_all_done,
        #                            model=model,
        #                            timeout=timeout)

        # exec_workload_fn = partial(execute_workload_claude_traces,
        #                             num_requests=num_requests,
        #                             arrival_rate=rate,
        #                             seed=42,
        #                             port=port,
        #                             model=model,
        #                             wait_for_all_done=wait_for_all_done,
        #                             timeout=timeout,
        #                             results_dir=abs_path  # Save metadata to results directory
        # )


        exps = []
        # for cache_ttl_value in [1, 3, 5, 7, 9, 11, 13, 15]:
        for cache_ttl_value in [0]:
            for pinned_memory_frac in [0.0]:
            # for pinned_memory_frac in [0.0]:
                for batch_size in batch_sizes:
                    # for mpl in [1, 3, 5]:
                    for mpl in [32, 64, 128, 256]:
                        exps.append(
                            {
                                "execute_workload_fn": exec_workload_fn,
                                'results_path': results_path,
                                'exp_name': exp_name + '-batch-%d-mpl-%s' % (batch_size, str(mpl)),
                                'config_name': "swap-hint",
                                'env': env,
                                'model': model,
                                'rope_scaling': rope_scaling,
                                'max_model_len': max_model_len,
                                'tp_size': tp_size, 
                                'pp_size': pp_size, 
                                'swap_space': 1,
                                'evict_token_thresh': 99999999999,
                                'evict_token_count': 0,
                                'enable_chunked_prefill': True,
                                'fr_policy': "default",
                                'swap_strategy': "swap-hints",
                                'block_allocator': "CpuOffloadingBlockAllocator",
                                'port': port,
                                'enable_returning_queue': enable_returning_queue, 
                                'returning_queue_sched_policy': "prio",
                                'returning_queue_sort_freq': 1.0,
                                'enable_swap_budget': False,
                                'swap_budget_type': "fixed",
                                'swap_budget_frac': 0.0,
                                'enable_eager_evict': False,
                                'cache_pin_ttl': cache_ttl_value,
                                'pinned_memory_frac': pinned_memory_frac,
                                'enable_cache_heirarchy': enable_cache_heirarchy,
                                'max_num_seqs': batch_size,
                                'max_num_batched_tokens': max_num_batched_tokens,
                                'enable_ws_control': False,
                                'ws_control_policy': "ws-hint",
                                'ws_control_deadline': 6.0,
                                'ws_size_fraction': 1.2,
                                'mpl': mpl,
                                'debug': DEBUG,
                                'timeout': timeout,
                            }
                        )

                        exps.append(
                            {
                                "execute_workload_fn": exec_workload_fn,
                                'results_path': results_path,
                                'exp_name': exp_name + '-batch-%d-mpl-%s' % (batch_size, str(mpl)),
                                'config_name': "swap-infercept",
                                'env': env,
                                'model': model,
                                'rope_scaling': rope_scaling,
                                'max_model_len': max_model_len,
                                'tp_size': tp_size, 
                                'pp_size': pp_size, 
                                'swap_space': 1,
                                'evict_token_thresh': 99999999999,
                                'evict_token_count': 0,
                                'enable_chunked_prefill': True,
                                'fr_policy': "default",
                                'swap_strategy': "swap-infercept",
                                'block_allocator': "CpuOffloadingBlockAllocator",
                                'port': port,
                                'enable_returning_queue': enable_returning_queue, 
                                'returning_queue_sched_policy': "prio",
                                'returning_queue_sort_freq': 1.0,
                                'enable_swap_budget': False,
                                'swap_budget_type': "fixed",
                                'swap_budget_frac': 0.0,
                                'enable_eager_evict': False,
                                'cache_pin_ttl': cache_ttl_value,
                                'pinned_memory_frac': pinned_memory_frac,
                                'enable_cache_heirarchy': enable_cache_heirarchy,
                                'max_num_seqs': batch_size,
                                'max_num_batched_tokens': max_num_batched_tokens,
                                'enable_ws_control': False,
                                'ws_control_policy': "ws-hint",
                                'ws_control_deadline': 6.0,
                                'ws_size_fraction': 1.2,
                                'mpl': mpl,
                                'debug': DEBUG,
                                'timeout': timeout,
                            }
                        )
                        
                        exps.append(
                            {
                                "execute_workload_fn": exec_workload_fn,
                                'results_path': results_path,
                                'exp_name': exp_name + '-batch-%d-mpl-%s' % (batch_size, str(mpl)),
                                'config_name': "swap-lru",
                                'env': env,
                                'model': model,
                                'rope_scaling': rope_scaling,
                                'max_model_len': max_model_len,
                                'tp_size': tp_size, 
                                'pp_size': pp_size, 
                                'swap_space': 1,
                                'evict_token_thresh': 99999999999,
                                'evict_token_count': 0,
                                'enable_chunked_prefill': True,
                                'fr_policy': "default",
                                'swap_strategy': "swap-lru",
                                'block_allocator': "CpuGpuBlockAllocator",
                                'port': port,
                                'enable_returning_queue': enable_returning_queue,
                                'returning_queue_sched_policy': "prio",
                                'returning_queue_sort_freq': 1.0,
                                'enable_swap_budget': False,
                                'swap_budget_type': "fixed",
                                'swap_budget_frac': 0.0,
                                'enable_eager_evict': False,
                                'cache_pin_ttl': cache_ttl_value,
                                'pinned_memory_frac': 0.0,
                                'enable_cache_heirarchy': enable_cache_heirarchy,
                                'max_num_seqs': batch_size,
                                'max_num_batched_tokens': max_num_batched_tokens,
                                'enable_ws_control': False,
                                'ws_control_policy': "ws-deadline",
                                'ws_control_deadline': 6.0,
                                'ws_size_fraction': 1.2,
                                'mpl': mpl,
                                'debug': DEBUG,
                                'timeout': timeout,
                            }
                        )
                
                        # exps.append(
                        #     {
                        #         "execute_workload_fn": exec_workload_fn,
                        #         'results_path': results_path,
                        #         'exp_name': exp_name + '-batch-%d' % (batch_size),
                        #         'config_name': "swap-random",
                        #         'env': env,
                        #         'model': model,
                        #         'rope_scaling': rope_scaling,
                        #         'max_model_len': max_model_len,
                        #         'tp_size': tp_size, 
                        #         'pp_size': pp_size, 
                        #         'swap_space': 1,
                        #         'evict_token_thresh': 99999999999,
                        #         'evict_token_count': 0,
                        #         'enable_chunked_prefill': True,
                        #         'fr_policy': "default",
                        #         'swap_strategy': "swap-random",
                        #         'block_allocator': "CpuOffloadingBlockAllocator",
                        #         'port': port,
                        #         'enable_returning_queue': enable_returning_queue, 
                        #         'returning_queue_sched_policy': "prio",
                        #         'returning_queue_sort_freq': 1.0,
                        #         'enable_swap_budget': False,
                        #         'swap_budget_type': "fixed",
                        #         'swap_budget_frac': 0.0,
                        #         'enable_eager_evict': False,
                        #         'cache_pin_ttl': cache_ttl_value,
                        #         'pinned_memory_frac': pinned_memory_frac,
                        #         'enable_cache_heirarchy': enable_cache_heirarchy,
                        #         'max_num_seqs': batch_size,
                        #         'max_num_batched_tokens': max_num_batched_tokens,
                        #         'enable_ws_control': False,
                        #         'ws_control_policy': "ws-hint",
                        #         'ws_control_deadline': 6.0,
                        #         'ws_size_fraction': 1.2,
                        #         'debug': DEBUG,
                        #         'timeout': timeout,
                        #     }
                        # )

        for exp in exps:
            run_experiment(**exp)


if __name__ == "__main__":
    main()
    


