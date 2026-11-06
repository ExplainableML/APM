from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import gc
import torch
import logging
from typing import Any
from dataclasses import dataclass


def suppress_vllm_logs():
    for name in logging.Logger.manager.loggerDict:
        if "vllm" in name:
            logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("vllm").setLevel(logging.WARNING)

@dataclass
class ModelResponse:
    content: str
    reasoning: str | None
    raw_text: str
    finish_reason_stop: bool

class LLMHelper:
    llm: None | LLM = None
    tokenizer: None | AutoTokenizer = None

    def __init__(self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int | None = None,
        enable_thinking: bool = False,
        sampling_params: dict[str, Any] | None = None,
        vllm_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.sampling_params = sampling_params or {}
        self.vllm_kwargs = vllm_kwargs or {}

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
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"[LLMHelper] GPU {i}: {free/1024**3:.1f}/{total/1024**3:.1f} GiB free after unload")

        print("[LLMHelper] Model unloaded")

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[ModelResponse]:
        
        def _tokenize(messages):
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            return result["input_ids"] if hasattr(result, "input_ids") else result
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            all_token_ids = list(tqdm(
                executor.map(_tokenize, conversations),
                total=len(conversations),
                desc="Tokenizing",
            ))
        inputs = [{"prompt_token_ids": ids} for ids in all_token_ids]

        outputs = self.llm.generate(inputs, sampling_params=SamplingParams(**self.sampling_params))

        model_responses = []
        for output in outputs:
            raw_text = output.outputs[0].text
            if self.enable_thinking:
                assistant_final_sep = "assistantfinal"
                think_start = raw_text.find("<think>")
                think_end = raw_text.find("</think>")
                af_pos = raw_text.find(assistant_final_sep)
                if af_pos != -1:  # gpt-oss harmony output decoded without special tokens
                    reasoning = raw_text[:af_pos].removeprefix("analysis").strip()
                    content = raw_text[af_pos + len(assistant_final_sep):].strip()
                elif think_start != -1 and think_end != -1:
                    reasoning = raw_text[think_start + len("<think>"):think_end].strip()
                    content = raw_text[think_end + len("</think>"):].strip()
                else:
                    reasoning = None
                    content = raw_text
                    print(f"[LLMHelper] Warning: No <think> tag or assistantfinal found in output: {raw_text!r}")
            else:
                reasoning = None
                content = raw_text

            model_responses.append(ModelResponse(
                content=content,
                reasoning=reasoning,
                raw_text=raw_text,
                finish_reason_stop=output.outputs[0].finish_reason == "stop",
            ))

        return model_responses
