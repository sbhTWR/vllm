import time
from os import environ
from subprocess import Popen, PIPE


def run_async(cmd, block_until_output=None, timeout=None, env=None):
    abs_env = environ.copy()
    abs_env.update(env)
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

def fetch_vllm_metrics(port=8000):
    """
    Fetch vllm metrics
    """


def main():

    env = {'CUDA_VISIBLE_DEVICES': '1'}
    model = "meta-llama/Llama-2-7b-chat-hf"
    tp_size = 1 
    pp_size = 1 
    fr_policy = "pause_swap"

    p = run_async(
        ["python3", "-m", "vllm.entrypoints.openai.api_server",
         "--model", model,
         "--tensor-parallel-size", str(tp_size),
         "--pipeline-parallel-size", str(pp_size),
         "--fr-policy", fr_policy],
        block_until_output="Uvicorn running",
        timeout=100,
        env=env
    )

    print("Pipeline is ready")
    # p.terminate()
    p.wait()
    print("Pipeline is done")

if __name__ == "__main__":
    main()