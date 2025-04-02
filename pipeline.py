import time
from os import environ
from subprocess import call, Popen, PIPE
from workload import workload0, workload1

def run_sync(cmd, env=None):
    abs_env = environ.copy()
    if env: 
        abs_env.update(env)

    p = call(cmd, env=abs_env)


def run_async(cmd, block_until_output=None, timeout=None, env=None):
    abs_env = environ.copy()
    if env:
        abs_env.update(env)

    print(' '.join(cmd))
    input()
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, env=abs_env)
    
    oom_line = "CUDA out of memory"

    if block_until_output:
        start_time = time.time()
    while True:
        stdout_line = str(p.stdout.readline())
        stderr_line = str(p.stderr.readline())
        print(stdout_line)
        print(stderr_line)
        if block_until_output in stdout_line\
            or block_until_output in stderr_line:
            return p
        elif oom_line in stderr_line or\
            oom_line in stdout_line:
            raise ValueError("CUDA OOM error was raised!")
        else:
            now = time.time()
            if timeout and now - start_time > timeout:
                raise ValueError(f"Timeout in laucnhing vLLM")


def execute_workload(port=8000):
    # workload0(port)
    workload1(port)


def main():

    exp_name = "test-swapall"
    env = {'CUDA_VISIBLE_DEVICES': '0'}
    model = "meta-llama/Llama-2-7b-chat-hf"
    tp_size = 1 
    pp_size = 1 
    fr_policy = "pause_recompute"
    swap_strategy = "swap_all"
    block_allocator = "CpuOffloadingBlockAllocator"
    retrify_log_file = "%s-retrify-vllm-log.csv" % exp_name
    port = 8000

    # kill anything on port 
    run_sync(['killport', str(port)])

    p = run_async(
        [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--tensor-parallel-size", str(tp_size),
            "--pipeline-parallel-size", str(pp_size),
            "--fr-policy", fr_policy,
            "--swap-strategy", swap_strategy,
            "--block-allocator", block_allocator,
            "--retrify-log-file", retrify_log_file,
            '--enable-prefix-caching',
            '--enforce-eager'
        ],
        block_until_output="Uvicorn running",
        timeout=100,
        env=env
    )

    print("Pipeline is ready")
    
    """
    execute workloads
    """
    execute_workload(port=port)

    time.sleep(10)
    
    p.terminate()
    p.wait()
    print("Pipeline is done")

if __name__ == "__main__":
    main()