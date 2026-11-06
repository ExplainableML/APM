<div align="center">

# APM: Evaluating Style Personalization in LLMs with Arbitrary Preference Mappings

**Philipp Spohn, Leander Girrbach, Zeynep Akata**

NeurIPS 2026 Evaluations & Datasets Track

<!-- TODO: add the paper URL -->
[📄 Paper](https://arxiv.org/abs/2605.21063) · [📝 BibTeX](#citation)

</div>

---

This repository contains the code for **APM**, a benchmark for style personalization in LLMs. It covers the full pipeline:

- generating synthetic users whose prompts reflect a latent style attribute,
- generating and judging candidate responses that give methods a shared training signal,
- mapping user attributes to preferred response styles through arbitrary preference mappings,
- training and evaluating personalization methods with an LLM judge,
- checking robustness on MMLU-Pro, TruthfulQA and BBQ.

## Installation

```bash
conda create -n llm-pers python=3.12 -y
conda activate llm-pers
pip install -r requirements.txt
pip install -e .
```

All models are served locally with [vLLM](https://github.com/vllm-project/vllm). With the default configs, the judge (`gpt-oss-120b`) and generators run on a single 80 GB GPU, and candidate generation with Nemotron-3-Super-120B (FP8) uses 4 GPUs. Change `tensor_parallel_size` in the `llm/`, `judge/` and `gen_model/` configs to fit your hardware.

We recommend pointing the model caches at a large volume:

```bash
export HF_HOME=/path/to/cache/huggingface
export VLLM_CACHE_ROOT=/path/to/cache/vllm
```

## Repository structure

```
src/llm_personalization/
├── personalization_system/
│   ├── benchmark.py            # main benchmark loop (worlds → train → generate → judge)
│   ├── preference_mapping.py   # world matrices: user attrs → response attrs
│   ├── cache.py                # joins conversations + candidate responses + ratings
│   ├── judging.py              # weighted attribute scoring
│   ├── robustness.py           # MMLU-Pro / TruthfulQA / BBQ evaluation
│   └── methods/                # unpersonalized, rag_*, routing*, user_embedding
├── judge/                      # LLM-as-judge (rating parsing, prompts)
├── llm/                        # vLLM wrapper
├── classification_model/       # ModernBERT text classifier for routing
└── data/                       # UltraChat + robustness dataset loaders

experiments/                    # one folder per pipeline stage (configs/ + scripts/)
```

| Stage (`experiments/…`) | What it does |
|---|---|
| `attribute_selection` | Has the judge rate a large pool of candidate user-prompt and response attributes, drops attributes whose score entropy is low (< 1.5), then runs PCA with parallel analysis and varimax rotation to pick near-independent attributes. Writes score histograms, a scree plot, rotated-component loadings and a UMAP plot of the attributes to `data/attribute_selection/`. |
| `synthetic_conversations` | Rewrites UltraChat prompts (`gpt-oss-120b`) so they show each synthetic user's attribute. |
| `attribute_responses` | Generates the 20 candidate responses per prompt (`Nemotron-3-Super-120B`), using the same system prompt the routing methods use. `merge_response_shards.py` checks the array shards and merges them. |
| `attribute_response_judging` | Judges each candidate on every response attribute. The `avoid` score is derived as $11 - s_\text{follow}$ instead of being judged separately. `validate_rating_shards.py` checks shard completeness. |
| `unpersonalized_responses` | Generates the baseline responses (no system prompt, no history) with each generator. |
| `unpersonalized_response_judging` | Judges the baseline responses on every response attribute, so the benchmark can look them up instead of re-judging. |
| `unpersonalized_robustness` | Caches the baseline answers on MMLU-Pro / TruthfulQA / BBQ for each generator. |
| `run_benchmark` | The main benchmark (`run_benchmark.py`), robustness evaluation split into shards, and `aggregate_results.py` to aggregate the results. |

Every stage is a [Hydra](https://hydra.cc) script: run it from the repository root and override any config value on the command line. All outputs go to `data/`. Generation and judging scripts can be split across a Slurm array: they read `SLURM_ARRAY_TASK_ID` / `SLURM_ARRAY_TASK_COUNT`, or `array_task_id=… array_task_count=…` on the command line, and each task writes its own `*_part<i>_of<n>.jsonl` shard.

## Running the evaluation

The default configs reproduce the main setting of the paper: 10 attributes, 1 attribute per user, 4,000 train / 500 val / 1,000 test users.

### 0. Attribute selection (optional)

The selected attributes are already set in the configs. To re-run the selection:

```bash
python experiments/attribute_selection/scripts/generate_ratings.py                                  # response attributes
python experiments/attribute_selection/scripts/generate_ratings.py --config-name config_user_prompts  # user attributes
python experiments/attribute_selection/scripts/analyze_ratings.py
python experiments/attribute_selection/scripts/analyze_ratings.py --config-name config_user_prompts
```

### 1. Synthetic users

```bash
python experiments/synthetic_conversations/scripts/generate_conversations.py
```

### 2. Candidate responses and judge cache

```bash
# 20 candidate responses per prompt (run once per split)
python experiments/attribute_responses/scripts/generate_attribute_responses.py split_name=train
python experiments/attribute_responses/scripts/generate_attribute_responses.py split_name=test

# merge shards into train.jsonl / test.jsonl
python experiments/attribute_responses/scripts/merge_response_shards.py \
    --input-dir data/attribute_responses/<run> --split train --num-shards <n> \
    --expected-records 4500 --expected-attributes concise formal structured warm lively technical cautious opinionated direct nuanced

# judge every candidate on every response attribute
python experiments/attribute_response_judging/scripts/judge_attribute_responses.py
```

### 3. Unpersonalized baseline cache

Run this once per generator (`llm=qwen3p5_27b` or `llm=llama31_8b`).

```bash
python experiments/unpersonalized_responses/scripts/generate_unpersonalized_responses.py llm=qwen3p5_27b
python experiments/unpersonalized_response_judging/scripts/judge_unpersonalized_responses.py \
    dataset=qwen3p5_27b_natural_k10_1attrs_test
python experiments/unpersonalized_robustness/scripts/generate_unpersonalized_robustness.py \
    llm=qwen3p5_27b robustness=full
```

### 4. Benchmark

```bash
# one method, one world
python experiments/run_benchmark/scripts/run_benchmark.py method=routing_abs gen_model=qwen3p5_27b

# several worlds, plus robustness evaluation
python experiments/run_benchmark/scripts/run_benchmark.py method=user_embedding \
    gen_model=llama31_8b +world_seeds=[0,1,2] +robustness=full
```

Each run writes to `data/run_benchmark/<timestamp>_<gen_model>_<method>_<seeds>_<pool>/`:

- `config.yaml`: the fully resolved config of the run.
- `trained_systems/world_<i>/`: the trained system (checkpoints, classifier, retrieval index).
- `results/results.json`: the same config, plus per-world personalized/unpersonalized means, win/tie/loss rates and robustness accuracies. It is rewritten after every world.
- `results/{attribute,robustness}_world_<i>_test_samples.jsonl`: per-user responses, targets and scores.

Robustness evaluation of a trained run can also be split into shards. Each shard reads the run's `config.yaml`, so only the shard needs to be specified. Merging writes the result into the run's `results.json`, exactly as an unsharded run would:

```bash
python experiments/run_benchmark/scripts/run_robustness_shard.py --run-dir data/run_benchmark/<run> \
    --world 0 --robustness full --shard-index 0 --num-shards 8
python experiments/run_benchmark/scripts/merge_robustness_shards.py --run-dir data/run_benchmark/<run> --world 0
```

### 5. Aggregate results

```bash
python experiments/run_benchmark/scripts/aggregate_results.py
```

This reads every `data/run_benchmark/*/results/results.json`, takes each run's metadata from its saved config, and writes per-world and mean ± std (over worlds) CSV tables to `data/run_benchmark/aggregated/`. If the same setting and world was run more than once, the latest run is used.

## Citation

```bibtex
@inproceedings{spohn2026apm,
  title     = {{APM}: Evaluating Style Personalization in {LLMs} with Arbitrary Preference Mappings},
  author    = {Spohn, Philipp and Girrbach, Leander and Akata, Zeynep},
  booktitle = {NeurIPS 2026 Evaluations \& Datasets Track},
  year      = {2026}
}
```
