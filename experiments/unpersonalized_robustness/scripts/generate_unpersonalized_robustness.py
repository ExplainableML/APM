"""Generate the unpersonalized baseline responses for the robustness benchmark,
cached once per (gen_model, robustness config) at
`robustness.unpers_cache_path(gen_model_tag, robustness)`."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from llm_personalization.data.robustness_dataset import format_mc_prompt, parse_answer_letter
from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.personalization_system.robustness import (
    accuracy_summary,
    load_questions,
    unpers_cache_path,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_path = Path(get_original_cwd()) / unpers_cache_path(cfg.gen_model_tag, cfg.robustness)
    if out_path.exists() and not cfg.get("force", False):
        print(f"[unpers_robustness] {out_path} already exists; skipping (pass force=true to regenerate)")
        return

    questions = load_questions(cfg.robustness)
    print(f"[unpers_robustness] loaded {len(questions)} questions")

    prompts = [[{"role": "user", "content": format_mc_prompt(q)}] for q in questions]
    print(f"[unpers_robustness] generating {len(prompts)} responses with {cfg.llm.model}")
    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()
    try:
        responses = [r.content for r in llm.generate(prompts)]
    finally:
        llm.unload()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for q, r in zip(questions, responses):
            f.write(json.dumps({
                "question_id": q.question_id,
                "source": q.source,
                "category": (q.metadata or {}).get("category"),
                "response": r,
                "parsed_letter": parse_answer_letter(r),
                "correct_letter": q.correct_letter,
            }) + "\n")
    print(f"[unpers_robustness] wrote {len(responses)} records to {out_path}")
    print(json.dumps(accuracy_summary(responses, questions), indent=2))


if __name__ == "__main__":
    main()
