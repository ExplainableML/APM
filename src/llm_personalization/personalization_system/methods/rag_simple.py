"""RAG-simple personalization: retrieve the top-k most history-similar train
users and prepend their (prompt, best response, worst response) exemplars to
the system prompt."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..prompts import format_history
from .base import PersonalizationSystem
from .common import encode, first_user_prompt, load_embedder, pick_best_worst, release_embedder, top_k_neighbors


RAG_SYSTEM_PROMPT = """\
Below are examples of prompts and responses showing the response styles \
that users similar to the current user prefer ("best") and dislike ("worst"). \
Use these examples to infer the kind of response style this user wants, then \
respond accordingly.

<examples>
{examples}
</examples>

Now respond to the user's prompt in a style consistent with what they appear \
to prefer based on the examples above. Do not mention or refer to the examples.
"""

EXAMPLE_TEMPLATE = """\
<example>
<prompt>{prompt}</prompt>
<best_response>{best}</best_response>
<worst_response>{worst}</worst_response>
</example>"""


class RAGSimpleSystem(PersonalizationSystem):
    def __init__(
        self,
        embedder_model: str,
        llm_helper_config: dict,
        top_k: int = 3,
        embedder_batch_size: int = 32,
        embedder_max_seq_length: int | None = None,
    ):
        self.embedder_model = embedder_model
        self.llm_helper = LLMHelper(**llm_helper_config)
        self.top_k = top_k
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_seq_length = embedder_max_seq_length

    # ------------------------- train -------------------------

    def train(
        self,
        dataset,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]],
        save_path: Path,
        val_dataset=None,
        val_user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug=None,
        val_gt_for_debug=None,
    ) -> None:
        # no trainable parameters; val/gt_for_debug are ignored
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print("[RAGSimpleSystem] 1. Selecting best/worst response per train user...")
        index_users = []
        bests: list[str] = []
        worsts: list[str] = []
        skipped = 0
        for user in dataset:
            scores = user_id_to_candidate_scores.get(user.user_id)
            if not scores:
                skipped += 1
                continue
            picked = pick_best_worst(user, scores)
            if picked is None:
                skipped += 1
                continue
            best, worst = picked
            index_users.append(user)
            bests.append(best)
            worsts.append(worst)
        print(f"[RAGSimpleSystem]    indexed {len(index_users)} users (skipped {skipped})")

        print("[RAGSimpleSystem] 2. Encoding histories...")
        log_gpu_usage("Before embedder load")
        embedder = load_embedder(self.embedder_model, self.embedder_max_seq_length)
        log_gpu_usage("After embedder load")
        history_texts = [format_history(u.history) for u in index_users]
        embeddings = encode(embedder, history_texts, self.embedder_batch_size)
        release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        print(f"[RAGSimpleSystem] 3. Saving index ({embeddings.shape}) to {save_path}")
        np.save(save_path / "embeddings.npy", embeddings)
        with open(save_path / "meta.jsonl", "w") as f:
            for u, best, worst in zip(index_users, bests, worsts):
                f.write(json.dumps({
                    "user_id": u.user_id,
                    "current_prompt": first_user_prompt(u.current_messages),
                    "best_response": best,
                    "worst_response": worst,
                }) + "\n")
        with open(save_path / "config.json", "w") as f:
            json.dump({
                "embedder_model": self.embedder_model,
                "top_k": self.top_k,
                "n_index_users": len(index_users),
            }, f, indent=2)

    # ------------------------- evaluate -------------------------

    def evaluate(
        self,
        dataset,
        load_path: Path,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug=None,
    ) -> list[str]:
        load_path = Path(load_path)
        embeddings = np.load(load_path / "embeddings.npy")  # L2-normalized
        meta = []
        with open(load_path / "meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        print(f"[RAGSimpleSystem] loaded index: {embeddings.shape}, {len(meta)} entries")

        print("[RAGSimpleSystem] 1. Encoding test histories...")
        log_gpu_usage("Before embedder load")
        embedder = load_embedder(self.embedder_model, self.embedder_max_seq_length)
        test_emb = encode(embedder, [format_history(u.history) for u in dataset], self.embedder_batch_size)
        release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        print("[RAGSimpleSystem] 2. Top-k retrieval (cosine, normalized)...")
        topk_idx = top_k_neighbors(test_emb, embeddings, self.top_k)

        print("[RAGSimpleSystem] 3. Building augmented prompts...")
        prompts = []
        for i, user in enumerate(dataset):
            examples = "\n".join(
                EXAMPLE_TEMPLATE.format(
                    prompt=meta[j]["current_prompt"],
                    best=meta[j]["best_response"],
                    worst=meta[j]["worst_response"],
                )
                for j in topk_idx[i]
            )
            sys_prompt = RAG_SYSTEM_PROMPT.format(examples=examples)
            prompts.append([{"role": "system", "content": sys_prompt}] + user.current_messages)

        print("[RAGSimpleSystem] 4. Generating responses (single batched vLLM call)...")
        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        responses = self.llm_helper.generate(prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")
        return [r.content for r in responses]
