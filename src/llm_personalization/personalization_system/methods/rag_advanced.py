"""RAG-advanced personalization: LLM-summarized user "vibes" are the retrieval
keys and LLM-summarized style preferences (from each train user's best/worst
responses) are the values prepended to the system prompt."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..prompts import format_history
from .base import PersonalizationSystem
from .common import encode, first_user_prompt, load_embedder, pick_best_worst, release_embedder, top_k_neighbors


VIBE_SYSTEM_PROMPT = """\
You analyze users by their writing style. You will be shown a few prompts a \
user has written. In 2-3 sentences, describe this user's vibe -- their tone, \
interests, level of expertise, and the kind of response style they would \
likely appreciate. Be specific and concrete; avoid generic statements.\
"""

VIBE_USER_TEMPLATE = """\
User's prompts:

{history}

Now write a 2-3 sentence vibe description for this user.\
"""

PREF_SYSTEM_PROMPT = """\
You analyze user preferences. You will be shown a prompt and two responses \
to it: one this user strongly liked, one they strongly disliked. In 2-3 \
sentences, describe what response style this user prefers and what they \
dislike, focusing on the concrete stylistic differences between the two \
responses (tone, length, structure, level of detail, phrasing). Do not \
mention "Response A" or "Response B" -- describe the user's preferences \
in general terms.\
"""

PREF_USER_TEMPLATE = """\
Prompt:
{prompt}

Response the user LIKED:
{best}

Response the user DISLIKED:
{worst}

Now write a 2-3 sentence preference description for this user.\
"""

RAG_SYSTEM_PROMPT = """\
The current user has the following style preferences (inferred from similar users):

{preferences}

Respond to the user's prompt in a style that matches these preferences. Do \
not mention or refer to the preference descriptions above.\
"""


class RAGAdvancedSystem(PersonalizationSystem):
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

    def _build_vibe_prompts(self, histories: list[list[str]]) -> list[list[dict]]:
        return [
            [
                {"role": "system", "content": VIBE_SYSTEM_PROMPT},
                {"role": "user", "content": VIBE_USER_TEMPLATE.format(history=format_history(h))},
            ]
            for h in histories
        ]

    def _build_pref_prompts(
        self, prompts: list[str], bests: list[str], worsts: list[str]
    ) -> list[list[dict]]:
        return [
            [
                {"role": "system", "content": PREF_SYSTEM_PROMPT},
                {"role": "user", "content": PREF_USER_TEMPLATE.format(prompt=p, best=b, worst=w)},
            ]
            for p, b, w in zip(prompts, bests, worsts)
        ]

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

        print("[RAGAdvancedSystem] 1. Selecting best/worst response per train user...")
        index_users = []
        bests, worsts, prompts_text, histories = [], [], [], []
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
            prompts_text.append(first_user_prompt(user.current_messages))
            histories.append(user.history)
        print(f"[RAGAdvancedSystem]    indexed {len(index_users)} users (skipped {skipped})")

        print("[RAGAdvancedSystem] 2. Generating vibe + preference summaries via LLM (batched)...")
        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        vibe_prompts = self._build_vibe_prompts(histories)
        pref_prompts = self._build_pref_prompts(prompts_text, bests, worsts)
        all_prompts = vibe_prompts + pref_prompts
        all_responses = self.llm_helper.generate(all_prompts)
        n = len(vibe_prompts)
        vibe_texts = [r.content for r in all_responses[:n]]
        pref_texts = [r.content for r in all_responses[n:]]
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")

        print("[RAGAdvancedSystem] 3. Encoding vibe descriptions...")
        log_gpu_usage("Before embedder load")
        embedder = load_embedder(self.embedder_model, self.embedder_max_seq_length)
        embeddings = encode(embedder, vibe_texts, self.embedder_batch_size)
        release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        print(f"[RAGAdvancedSystem] 4. Saving index ({embeddings.shape}) to {save_path}")
        np.save(save_path / "embeddings.npy", embeddings)
        with open(save_path / "meta.jsonl", "w") as f:
            for u, vibe, pref in zip(index_users, vibe_texts, pref_texts):
                f.write(json.dumps({
                    "user_id": u.user_id,
                    "vibe": vibe,
                    "preference": pref,
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
        embeddings = np.load(load_path / "embeddings.npy")
        meta = []
        with open(load_path / "meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        print(f"[RAGAdvancedSystem] loaded index: {embeddings.shape}, {len(meta)} entries")

        print("[RAGAdvancedSystem] 1. Generating test vibe descriptions via LLM...")
        log_gpu_usage("Before LLM load (vibe)")
        self.llm_helper.load()
        vibe_prompts = self._build_vibe_prompts([u.history for u in dataset])
        test_vibes = [r.content for r in self.llm_helper.generate(vibe_prompts)]
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload (vibe)")

        print("[RAGAdvancedSystem] 2. Encoding + retrieving top-k...")
        log_gpu_usage("Before embedder load")
        embedder = load_embedder(self.embedder_model, self.embedder_max_seq_length)
        test_emb = encode(embedder, test_vibes, self.embedder_batch_size)
        release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        topk_idx = top_k_neighbors(test_emb, embeddings, self.top_k)

        print("[RAGAdvancedSystem] 3. Building augmented prompts + generating responses...")
        gen_prompts = []
        for i, user in enumerate(dataset):
            preferences = "\n".join(
                f"- {meta[j]['preference']}" for j in topk_idx[i]
            )
            sys_prompt = RAG_SYSTEM_PROMPT.format(preferences=preferences)
            gen_prompts.append([{"role": "system", "content": sys_prompt}] + user.current_messages)

        log_gpu_usage("Before LLM load (gen)")
        self.llm_helper.load()
        responses = self.llm_helper.generate(gen_prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload (gen)")
        return [r.content for r in responses]
