import os
import sys
from threading import Thread
import threading
import time
from queue import Queue, Empty
from os import environ
from subprocess import call, Popen, PIPE
from lorem_text import lorem
import multiprocessing
import json
import numpy as np
# from workload import workload0, workload1
# from elasticswap.test_cyclic_workload_v2 import generate_workload
import shutil
import traceback

rng = np.random.default_rng(seed=42)

ON_POSIX = 'posix' in sys.builtin_module_names

class AsyncOutputReader:
    def __init__(self, out, filename, flush_interval=100, flush_thresh=3):
        self.filename = filename
        self.flush_interval = flush_interval
        self.queue = Queue()
        self.out = out
        self.last_flushed = time.time()
        self.flush_thresh = flush_thresh
        self.fp = open(filename, "wb+")
        self._terminate = False

    def enqueue_output(self):
        for line in iter(self.out.readline, b''):
            self.queue.put(line)
            if self._terminate:
                break
        self.out.close()

    def readline_async(self):
        # read line without blocking
        try:  
            line = self.queue.get_nowait() # or q.get(timeout=.1)
            self.fp.write(line)
            now = time.time()
            if now - self.last_flushed >= self.flush_thresh:
                self.fp.flush()
                self.last_flushed = now
        except Empty:
            time.sleep(1)
            # print('no output yet')
            return b''
        else:
            return line
    
    def flush_to_file_loop(self):
        while True:
            self.readline_async()
            if self._terminate:
                break 
    
    def terminate(self):
        self._terminate = True

def run_sync(cmd, env=None):
    abs_env = environ.copy()
    if env: 
        abs_env.update(env)

    p = call(cmd, env=abs_env)


def run_async(cmd, output_filename, block_until_output=None, timeout=None, env=None):
    abs_env = environ.copy()
    if env:
        abs_env.update(env)

    print(' '.join(cmd))
    # input()
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, env=abs_env)

    # q_stdout = Queue()
    p_stdout = AsyncOutputReader(p.stdout, output_filename+".stdout")
    t_stdout = Thread(target=p_stdout.enqueue_output, args=())
    t_stdout.daemon = True # thread dies with the program
    t_stdout.start()

    p_stderr = AsyncOutputReader(p.stderr, output_filename+".stderr")
    t_stderr = Thread(target=p_stderr.enqueue_output, args=())
    t_stderr.daemon = True # thread dies with the program
    t_stderr.start()

    oom_line = "CUDA out of memory"
    fnotfound = "FileNotFoundError"

    if block_until_output:
        start_time = time.time()
    while True:
        stdout_line = str(p_stdout.readline_async())
        stderr_line = str(p_stderr.readline_async())
        print(stdout_line)
        print(stderr_line)
        if block_until_output in stdout_line\
            or block_until_output in stderr_line:
            return p, p_stdout, p_stderr, t_stdout, t_stderr
        elif oom_line in stderr_line or\
            oom_line in stdout_line:
            raise ValueError("CUDA OOM error was raised!")
        elif fnotfound in stderr_line or \
        fnotfound in stdout_line:
            raise ValueError("File not found error was raised!")
        else:
            now = time.time()
            if timeout and now - start_time > timeout:
                raise ValueError(f"Timeout in laucnhing vLLM")


# def execute_workload(port=8000):
#     # workload0(port)
#     workload1(port)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_experiment(
    execute_workload_fn,
    results_path = "/vllm/vllm/elasticswap/results",
    exp_name = "test-all",
    config_name = "swapall-le",
    env = {
        'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1'
    },
    model = "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
    rope_scaling = None,
    max_model_len = None,
    tp_size = 1,
    pp_size = 1,
    swap_space = 100,
    evict_token_thresh = 1000000,
    evict_token_count = 10000,
    enable_chunked_prefill = False,
    fr_policy = "default",
    swap_strategy = "swap-lru",
    block_allocator = "CpuOffloadingBlockAllocator",
    port = 8000,
    enable_returning_queue = False,
    returning_queue_sort_freq = 5.0,
    returning_queue_sched_policy = "roundrobin",
    enable_swap_budget = False,
    swap_budget_type = "fixed",
    swap_budget_frac = 0.5,
    enable_eager_evict = False,
    cache_pin_ttl = 5,
    pinned_memory_frac = 0.25,
    enable_cache_heirarchy = True,
    max_num_seqs = 100,
    max_num_batched_tokens = 2048,
    enable_ws_control = False,
    ws_control_policy = "ws-deadline",
    ws_control_deadline = 1.0,
    ws_size_fraction = 1.1,
    debug = False,
    mpl = None,

    enable_prefix_priority_queue = False,
    priority_queue_num_levels = 15,
    priority_queue_max_match_len = 200000,
    priority_queue_bucketing = "logarithmic",
    predictor_config = None,

    timeout = 120,
):
    try:
        retrify_log_file = "%s-%s-retrify-vllm-log.csv" % (exp_name, config_name)
        exp_path = os.path.join(results_path, exp_name)
        ensure_dir(exp_path)
        vllm_log_file = os.path.join(exp_path, retrify_log_file)

        output_log_file = vllm_log_file.split(".")[0] 
        # kill anything on port 
        # run_sync(['killport', str(port)])

        if debug:
            return

        vllm_args_list = [
                "python3", "-u", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model,
                "--tensor-parallel-size", str(tp_size),
                "--pipeline-parallel-size", str(pp_size),
                "--fr-policy", fr_policy,
                "--swap-strategy", swap_strategy,
                "--block-allocator", block_allocator,
                "--retrify-log-file", vllm_log_file,
                "--enable-chunked-prefill", str(enable_chunked_prefill),
                "--swap-space", str(swap_space),
                "--evict-token-thresh", str(evict_token_thresh), 
                "--evict-token-count", str(evict_token_count),
                "--enable-returning-queue", str(enable_returning_queue),
                "--returning-queue-sched-policy", str(returning_queue_sched_policy),
                "--returning-queue-sort-freq", str(returning_queue_sort_freq),
                "--enable-swap-budget", str(enable_swap_budget),
                "--swap-budget-type", str(swap_budget_type),
                "--swap-budget-frac", str(swap_budget_frac),
                "--enable-eager-evict", str(enable_eager_evict),

                "--enable-prefix-priority-queue", str(enable_prefix_priority_queue),
                "--priority-queue-num-levels", str(priority_queue_num_levels),
                "--priority-queue-max-match-len", str(priority_queue_max_match_len),
                "--priority-queue-bucketing", str(priority_queue_bucketing),

                "--cache-pin-ttl", str(cache_pin_ttl),
                "--pinned-memory-frac", str(pinned_memory_frac),
                "--preemption-mode", "recomputation",
                "--max-num-seqs", str(max_num_seqs),
                "--block-size", "128",
                '--enable-prefix-caching',
                '--enforce-eager',
                '--enable-cache-heirarchy', str(enable_cache_heirarchy),
                "--port", str(port),
                "--max-num-batched-tokens", str(max_num_batched_tokens),
                "--enable-ws-control", str(enable_ws_control),
                "--ws-control-policy", str(ws_control_policy),
                "--ws-control-deadline", str(ws_control_deadline),
                "--ws-size-fraction", str(ws_size_fraction),
            ]

        if predictor_config:
            vllm_args_list.append("--predictor")
            vllm_args_list.append(json.dumps(predictor_config))

        if mpl is not None:
            vllm_args_list.append("--mpl")
            vllm_args_list.append(str(mpl))

        if rope_scaling:
            vllm_args_list.append("--rope-scaling")
            vllm_args_list.append(str(rope_scaling))
        if max_model_len:
            vllm_args_list.append("--max-model-len")
            vllm_args_list.append(str(max_model_len))

        try:
            p, p_stdout, p_stderr, t_stdout, t_stderr = run_async(
                vllm_args_list,
                output_filename=output_log_file,
                block_until_output="Uvicorn running",
                timeout=900,
                env=env
            )
        except Exception as e:
            print(f"Error starting vLLM server: {e}")
            traceback.print_exc()
            raise

        t_stdout_loop = Thread(target=p_stdout.flush_to_file_loop, args=())
        t_stdout_loop.daemon = True # thread dies with the program
        t_stdout_loop.start()

        t_stderr_loop = Thread(target=p_stdout.flush_to_file_loop, args=())
        t_stderr_loop.daemon = True # thread dies with the program
        t_stderr_loop.start()
        print("Pipeline is ready")
        
        """
        execute workloads
        """
        def run_workload_in_subprocess():
            try:
                execute_workload_fn(port=port)
            except Exception as e:
                print(f"Error in workload execution: {e}")
                traceback.print_exc()
                raise
        
        print('Launching experiment with timeout %d' % timeout)
        process = multiprocessing.Process(target=run_workload_in_subprocess)
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            print("Workload process is hanging, force killing...")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                print("Process won't terminate, using SIGKILL...")
                process.kill()
                process.join()

        print('Done executing workloads, waiting for 60 seconds')
        time.sleep(60)
        

        p_stdout.terminate()
        p_stderr.terminate()
        p.terminate()
        p.wait()
        print("Pipeline is done")
    except Exception as e:
        print(f"Fatal error in run_experiment: {e}")
        traceback.print_exc()
        raise
    finally:
        # Ensure cleanup happens even if there's an error
        try:
            p_stdout.terminate()
            p_stderr.terminate()
            p.terminate()
            p.wait()
        except:
            pass
        print("Pipeline cleanup complete")

# def execute_workload(port=8000):
#     threads = []
#     request_size = 30000
#     interrupt_len = 20
#     num_interrupts = 5
#     rate = 0.30
#     t = 50
#     num_events = int(rate * t)
#     exp_times = rng.exponential(scale=1/rate, size=num_events)
    
#     print(np.cumsum(exp_times))
#     arrival_times = list(exp_times)

#     print(arrival_times)
#     # input()

#     print('--- test1 ---')
#     req1 = lorem.words(100)
#     print(req1)
    
#     print('--- test2 ---')
#     req2 = lorem.words(100)
#     print(req2)

#     # input()

#     for i in range(num_events):
#         workload = generate_workload(request_size, interrupt_len, num_interrupts, "test_%d" % i)
#         th = threading.Thread(target=workload.execute_workload, args=())
#         threads.append(th)

#     for thread in threads:
#         wait_time = arrival_times.pop(0)
#         time.sleep(wait_time)
#         thread.start()

#     for thread in threads:
#         thread.join()

# def main():
#     env = {
#             'CUDA_VISIBLE_DEVICES': '0,1,2,3',
#             'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1'
#         }
    
#     exp_name = "baselines8"
#     results_path = "/vllm/vllm/elasticswap/results"
#     abs_path = os.path.join("/vllm/vllm/elasticswap/results", exp_name)

#     ensure_dir(abs_path)
#     copied_script_name = "pipeline.py"

#     shutil.copy(__file__, os.path.join(abs_path, copied_script_name)) 

#     exps = [
#         { 
#             'results_path': results_path,
#             'exp_name':  exp_name,
#             'config_name': "swapall-le",
#             'env': env,
#             'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#             'tp_size': 1, 
#             'pp_size': 1, 
#             'swap_space': 100,
#             'evict_token_thresh': 1000000,
#             'evict_token_count': 10000,
#             'enable_chunked_prefill': False,
#             'fr_policy': "default",
#             'swap_strategy': "swap_all",
#             'block_allocator': "CpuOffloadingBlockAllocator",
#             'port': 8000,
#         },

#         # { 
#         #     'results_path': results_path,
#         #     'exp_name':  exp_name,
#         #     'config_name': "persist-le",
#         #     'env': env,
#         #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#         #     'tp_size': 1, 
#         #     'pp_size': 1, 
#         #     'swap_space': 100,
#         #     'evict_token_thresh': 1000000,
#         #     'evict_token_count': 10000,
#         #     'enable_chunked_prefill': False,
#         #     'fr_policy': "pause_recompute",
#         #     'swap_strategy': "persist",
#         #     'block_allocator': "CpuOffloadingBlockAllocator",
#         #     'port': 8000,
#         # },

#         # { 
#         #     'results_path': results_path,
#         #     'exp_name':  exp_name,
#         #     'config_name': "swapall-he",
#         #     'env': env,
#         #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#         #     'tp_size': 1, 
#         #     'pp_size': 1, 
#         #     'swap_space': 100,
#         #     'evict_token_thresh': 30000,
#         #     'evict_token_count': 10000,
#         #     'enable_chunked_prefill': False,
#         #     'fr_policy': "pause_recompute",
#         #     'swap_strategy': "swap_all",
#         #     'block_allocator': "CpuOffloadingBlockAllocator",
#         #     'port': 8000,
#         # },

#         # { 
#         #     'results_path': results_path,
#         #     'exp_name':  exp_name,
#         #     'config_name': "persist-he",
#         #     'env': env,
#         #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#         #     'tp_size': 1, 
#         #     'pp_size': 1, 
#         #     'swap_space': 100,
#         #     'evict_token_thresh': 30000,
#         #     'evict_token_count': 10000,
#         #     'enable_chunked_prefill': False,
#         #     'fr_policy': "pause_recompute",
#         #     'swap_strategy': "persist",
#         #     'block_allocator': "CpuOffloadingBlockAllocator",
#         #     'port': 8000,
#         # },

#         # { 
#         #     'results_path': results_path,
#         #     'exp_name':  exp_name,
#         #     'config_name': "default",
#         #     'env': env,
#         #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#         #     'tp_size': 1, 
#         #     'pp_size': 1, 
#         #     'swap_space': 100,
#         #     'evict_token_thresh': 1000000,
#         #     'evict_token_count': 10000,
#         #     'enable_chunked_prefill': False,
#         #     'fr_policy': "pause_recompute",
#         #     'swap_strategy': "swap_all",
#         #     'block_allocator': "CpuGpuBlockAllocator",
#         #     'port': 8000,
#         # },

#         # { 
#         #     'results_path': results_path,
#         #     'exp_name':  exp_name,
#         #     'config_name': "persist_he-pp2",
#         #     'env': env,
#         #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#         #     'tp_size': 1, 
#         #     'pp_size': 2, 
#         #     'swap_space': 100,
#         #     'evict_token_thresh': 30000,
#         #     'evict_token_count': 10000,
#         #     'enable_chunked_prefill': False,
#         #     'fr_policy': "pause_recompute",
#         #     'swap_strategy': "persist",
#         #     'block_allocator': "CpuOffloadingBlockAllocator",
#         #     'port': 8000,
#         # },

#         # { 
#         #     'results_path': results_path,
#         #     'exp_name':  exp_name,
#         #     'config_name': "persist_le-pp2",
#         #     'env': env,
#         #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#         #     'tp_size': 1, 
#         #     'pp_size': 2, 
#         #     'swap_space': 100,
#         #     'evict_token_thresh': 1000000,
#         #     'evict_token_count': 10000,
#         #     'enable_chunked_prefill': False,
#         #     'fr_policy': "pause_recompute",
#         #     'swap_strategy': "persist",
#         #     'block_allocator': "CpuOffloadingBlockAllocator",
#         #     'port': 8000,
#         # },


#         # { 
#         #     'results_path': results_path,
#         #     'exp_name':  exp_name,
#         #     'config_name': "persist_le-tp2pp2",
#         #     'env': env,
#         #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
#         #     'tp_size': 2, 
#         #     'pp_size': 2, 
#         #     'swap_space': 100,
#         #     'evict_token_thresh': 1000000,
#         #     'evict_token_count': 10000,
#         #     'enable_chunked_prefill': False,
#         #     'fr_policy': "pause_recompute",
#         #     'swap_strategy': "persist",
#         #     'block_allocator': "CpuOffloadingBlockAllocator",
#         #     'port': 8000,
#         # },

#     ]

#     for exp in exps:
#         print('Running experiment %s - %s' % (exp['exp_name'], exp['config_name']))
#         run_experiment(**exp)


