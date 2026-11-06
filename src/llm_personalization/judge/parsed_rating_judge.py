from vllm.outputs import RequestOutput
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import re
import torch
from llm_personalization.llm.llm_helper import suppress_vllm_logs
from .judge import AttributeJudge
from typing import Any
from .prompt_templates import (
    JUDGE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT_THINKING,
    JUDGE_USER_TEMPLATE_RESPONSE_ATTRIBUTE, JUDGE_USER_TEMPLATE_PROMPT_ATTRIBUTE,
)

class ParsedRatingJudge(AttributeJudge):
    llm: None | LLM = None
    tokenizer: None | AutoTokenizer = None

    def __init__(self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int | None = None,
        enable_thinking: bool = False,
        force_rating_on_thinking_timeout: bool = True,
        sampling_params: dict[str, Any] | None = None,
        vllm_kwargs: dict[str, Any] | None = None,
        retries: int = 3,
        thinking_close_token: str = "</think>",
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.force_rating_on_thinking_timeout = force_rating_on_thinking_timeout

        self.sampling_params = sampling_params or {}
        self.vllm_kwargs = vllm_kwargs or {}
        self.retries = retries
        # appended to a truncated reasoning trace to force a rating
        # (e.g. "</think>" for Qwen/DeepSeek, "<channel|>" for Gemma 4)
        self.thinking_close_token = thinking_close_token

    def load(self):
        suppress_vllm_logs()
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            **self.vllm_kwargs,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def unload(self) -> None:
        if self.llm is None:
            return
        
        del self.llm
        del self.tokenizer
        self.llm = None
        self.tokenizer = None

        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[RatingJudge] Judge model unloaded")


    def _parse_score(self, output: RequestOutput) -> float | None:
        text = output.outputs[0].text.strip()
        matches = re.findall(r'\b(100|[1-9]\d?)\b', text)
        if matches:
            score = int(matches[-1])
            if 1 <= score <= 100:
                return score / 10.0
        return None

    def judge_manual(self, judge_prompts: list[str]) -> list[float | None]:
        scores: list[float | None] = [None] * len(judge_prompts)
        original_prompts = list(judge_prompts)
        pending_indices = list(range(len(judge_prompts)))

        for attempt_idx in range(self.retries):
            pending_prompts = [original_prompts[i] for i in pending_indices]
            outputs = self.llm.generate(pending_prompts, sampling_params=SamplingParams(**self.sampling_params))

            timeout_local_indices = []
            for local_i, (score_idx, output) in enumerate(zip(pending_indices, outputs)):
                if output.outputs[0].finish_reason == "stop":
                    scores[score_idx] = self._parse_score(output)
                else:
                    timeout_local_indices.append(local_i)

            if self.force_rating_on_thinking_timeout and timeout_local_indices:
                print(f"[ParsedRatingJudge] {len(timeout_local_indices)}/{len(pending_prompts)} prompts hit thinking timeout, forcing rating...")
                retry_prompts = [
                    pending_prompts[local_i] + outputs[local_i].outputs[0].text + f"{self.thinking_close_token}\n\nRating: "
                    for local_i in timeout_local_indices
                ]
                retry_outputs = self.llm.generate(retry_prompts, sampling_params=SamplingParams(**{**self.sampling_params, "max_tokens": 4}))
                for local_i, o in zip(timeout_local_indices, retry_outputs):
                    scores[pending_indices[local_i]] = self._parse_score(o)

            valid_score_count = sum(1 for score in scores if score is not None)

            if valid_score_count == len(scores):
                print(f"[ParsedRatingJudge] All {len(scores)} scores valid after {attempt_idx + 1} retries")
                break
            else:
                print(f"[ParsedRatingJudge] {valid_score_count}/{len(scores)} scores valid, retrying...")
                pending_indices = [i for i in range(len(scores)) if scores[i] is None]

        return scores

    def judge_response_attribute(self, conversations: list[list[dict[str, str]]], attributes: list[str]) -> list[float | None]:
        judge_prompts = []

        for messages, attribute in zip(conversations, attributes):
            if len(messages) < 2:
                raise ValueError(f"Conversation has less than 2 messages: {messages}")
            if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
                raise ValueError(f"Last message must be an assistant response and the second to last message must be a user prompt.")
            message_string = ""
            for message in messages[:-1]:
                message_string += f"<message role='{message['role']}'>{message['content']}</message>\n"
            system_prompt = JUDGE_SYSTEM_PROMPT_THINKING if self.enable_thinking else JUDGE_SYSTEM_PROMPT
            user_prompt = JUDGE_USER_TEMPLATE_RESPONSE_ATTRIBUTE.format(conversation=message_string, response=messages[-1]["content"], attribute=attribute)
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            judge_prompts.append(full_prompt)
            
        return self.judge_manual(judge_prompts)

    def judge_user_prompt_attribute(self, conversations: list[list[dict[str, str]]], attributes: list[str]) -> list[float | None]:
        judge_prompts = []

        for messages, attribute in zip(conversations, attributes):
            if messages[-1]["role"] != "user":
                raise ValueError(f"Last message must be a user prompt.")
            message_string = ""
            for message in messages[:-1]:
                message_string += f"<message role='{message['role']}'>{message['content']}</message>\n"
            system_prompt = JUDGE_SYSTEM_PROMPT_THINKING if self.enable_thinking else JUDGE_SYSTEM_PROMPT
            user_prompt = JUDGE_USER_TEMPLATE_PROMPT_ATTRIBUTE.format(conversation=message_string, user_prompt=messages[-1]["content"], attribute=attribute)
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            judge_prompts.append(full_prompt)

        return self.judge_manual(judge_prompts)
