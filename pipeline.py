import os
import sys
from threading import Thread
import threading
import time
from queue import Queue, Empty
from os import environ
from subprocess import call, Popen, PIPE
from lorem_text import lorem

import numpy as np
from workload import workload0, workload1
from elasticswap.test_cyclic_workload_v2 import generate_workload

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
    results_path = "/vllm/vllm/elasticswap/results",
    exp_name = "test-all",
    config_name = "swapall-le",
    env = {
        'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1'
    },
    model = "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
    tp_size = 1,
    pp_size = 1,
    swap_space = 100,
    evict_token_thresh = 1000000,
    evict_token_count = 10000,
    enable_chunked_prefill = False,
    fr_policy = "pause_recompute",
    swap_strategy = "swap_all",
    block_allocator = "CpuOffloadingBlockAllocator",
    port = 8000,
):
    retrify_log_file = "%s-%s-retrify-vllm-log.csv" % (exp_name, config_name)
    exp_path = os.path.join(results_path, exp_name)
    ensure_dir(exp_path)
    vllm_log_file = os.path.join(exp_path, retrify_log_file)

    output_log_file = vllm_log_file.split(".")[0] 
    # kill anything on port 
    run_sync(['killport', str(port)])
    
    p, p_stdout, p_stderr, t_stdout, t_stderr = run_async(
        [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
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
            '--enable-prefix-caching',
            '--enforce-eager'
        ],
        output_filename=output_log_file,
        block_until_output="Uvicorn running",
        timeout=200,
        env=env
    )

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
    execute_workload(port=port)

    time.sleep(10)
    

    p_stdout.terminate()
    p_stderr.terminate()
    p.terminate()
    p.wait()
    print("Pipeline is done")

def execute_workload(port=8000):
    threads = []
    request_size = 30000
    interrupt_len = 5
    num_interrupts = 5
    rate = 0.3
    t = 50
    num_events = int(rate * t)
    exp_times = np.random.exponential(scale=1/rate, size=num_events)
    print(np.cumsum(exp_times))
    arrival_times = list(exp_times)

    print(arrival_times)
    # input()

    print('--- test1 ---')
    req1 = lorem.words(100)
    print(req1)
    
    print('--- test2 ---')
    req2 = lorem.words(100)
    print(req2)

    # input()

    for i in range(num_events):
        workload = generate_workload(request_size, interrupt_len, num_interrupts, "test_%d" % i)
        th = threading.Thread(target=workload.execute_workload, args=())
        threads.append(th)

    for thread in threads:
        wait_time = arrival_times.pop(0)
        time.sleep(wait_time)
        thread.start()

    for thread in threads:
        thread.join()

def main():
    env = {
            'CUDA_VISIBLE_DEVICES': '2',
            'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1'
        }
    
    exp_name = "baselines2"

    exps = [
        # { 
        #     'results_path': "/vllm/vllm/elasticswap/results",
        #     'exp_name':  exp_name,
        #     'config_name': "swapall-le",
        #     'env': env,
        #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
        #     'tp_size': 1, 
        #     'pp_size': 1, 
        #     'swap_space': 100,
        #     'evict_token_thresh': 1000000,
        #     'evict_token_count': 10000,
        #     'enable_chunked_prefill': False,
        #     'fr_policy': "pause_recompute",
        #     'swap_strategy': "swap_all",
        #     'block_allocator': "CpuOffloadingBlockAllocator",
        #     'port': 8000,
        # },

        # { 
        #     'results_path': "/vllm/vllm/elasticswap/results",
        #     'exp_name':  exp_name,
        #     'config_name': "persist-le",
        #     'env': env,
        #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
        #     'tp_size': 1, 
        #     'pp_size': 1, 
        #     'swap_space': 100,
        #     'evict_token_thresh': 1000000,
        #     'evict_token_count': 10000,
        #     'enable_chunked_prefill': False,
        #     'fr_policy': "pause_recompute",
        #     'swap_strategy': "persist",
        #     'block_allocator': "CpuOffloadingBlockAllocator",
        #     'port': 8000,
        # },

        # { 
        #     'results_path': "/vllm/vllm/elasticswap/results",
        #     'exp_name':  exp_name,
        #     'config_name': "swapall-he",
        #     'env': env,
        #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
        #     'tp_size': 1, 
        #     'pp_size': 1, 
        #     'swap_space': 100,
        #     'evict_token_thresh': 30000,
        #     'evict_token_count': 10000,
        #     'enable_chunked_prefill': False,
        #     'fr_policy': "pause_recompute",
        #     'swap_strategy': "swap_all",
        #     'block_allocator': "CpuOffloadingBlockAllocator",
        #     'port': 8000,
        # },

        # { 
        #     'results_path': "/vllm/vllm/elasticswap/results",
        #     'exp_name':  exp_name,
        #     'config_name': "persist-he",
        #     'env': env,
        #     'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
        #     'tp_size': 1, 
        #     'pp_size': 1, 
        #     'swap_space': 100,
        #     'evict_token_thresh': 30000,
        #     'evict_token_count': 10000,
        #     'enable_chunked_prefill': False,
        #     'fr_policy': "pause_recompute",
        #     'swap_strategy': "persist",
        #     'block_allocator': "CpuOffloadingBlockAllocator",
        #     'port': 8000,
        # },

        { 
            'results_path': "/vllm/vllm/elasticswap/results",
            'exp_name':  exp_name,
            'config_name': "default",
            'env': env,
            'model': "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct",
            'tp_size': 1, 
            'pp_size': 1, 
            'swap_space': 100,
            'evict_token_thresh': 1000000,
            'evict_token_count': 10000,
            'enable_chunked_prefill': False,
            'fr_policy': "pause_recompute",
            'swap_strategy': "swap_all",
            'block_allocator': "CpuGpuBlockAllocator",
            'port': 8000,
        },

    ]

    for exp in exps:
        print('Running experiment %s - %s' % (exp['exp_name'], exp['config_name']))
        run_experiment(**exp)


if __name__ == "__main__":
    main()