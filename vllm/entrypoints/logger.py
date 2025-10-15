# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class RequestLogger:

    def __init__(self, *, max_log_len: Optional[int]) -> None:
        super().__init__()

        self.max_log_len = max_log_len

    def log_inputs(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        params: Optional[Union[SamplingParams, PoolingParams,
                               BeamSearchParams]],
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
    ) -> None:
        max_log_len = self.max_log_len
        
        # For logging, truncate or summarize
        prompt_summary = prompt
        token_ids_summary = prompt_token_ids
        
        if max_log_len is not None:
            if prompt is not None:
                prompt_summary = prompt[:max_log_len]

            if prompt_token_ids is not None:
                token_ids_summary = prompt_token_ids[:max_log_len]
        
        # If prompt_token_ids are provided, don't log the decoded text
        # (it's often gibberish from random token IDs)
        if prompt_token_ids is not None and len(prompt_token_ids) > 100:
            # Just log the length and first few tokens
            prompt_summary = f"<{len(prompt_token_ids)} tokens>"
            token_ids_summary = f"{prompt_token_ids[:10]}...{prompt_token_ids[-5:]}"

        logger.info(
            "Received request %s: prompt: %r, "
            "params: %s, prompt_token_ids: %s, "
            "lora_request: %s, prompt_adapter_request: %s.", request_id,
            prompt_summary, params, token_ids_summary, lora_request,
            prompt_adapter_request)
