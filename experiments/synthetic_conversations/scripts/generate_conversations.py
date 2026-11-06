import hydra
from omegaconf import DictConfig
from llm_personalization.llm.llm_helper import LLMHelper
from hydra.utils import instantiate, get_original_cwd
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations_with_ids

from pathlib import Path
import random
import json
from tqdm import tqdm

SYSTEM_PROMPT = """
Your task: Rewrite the highlighted user prompt to match a described persona. Output only the rewritten prompt, nothing else.

Rules:
- Preserve the core meaning, intent, and information content of the original prompt.
- Only rewrite and output the "prompt_to_rewrite" message, the preceding conversation is just for context.
- Change the way it is expressed — word choice, sentence structure, tone, and register — to match the target style. Apply the style NATURALLY — the rewrite should still sound like a real person, not a caricature. Less is more!
- Do not mention any style attribute explicitly in the rewritten prompt.
- Keep roughly the same length unless the style naturally calls for shorter or longer phrasing (e.g., a terse style should be shorter, a verbose style should be longer).
"""

USER_PROMPT_TEMPLATE = """
Rewrite the highlighted user prompt to match a described persona. Output only the rewritten prompt, nothing else.

<conversation>
{conversation}
</conversation>

<prompt_to_rewrite>
{prompt_to_rewrite}
</prompt_to_rewrite>

<persona>
The simulated user is...
{formatted_attributes}
</persona>
"""

def _format_conversation(conversation: list[dict[str, str]]) -> str:
    return "\n".join([f"<message role='{message['role']}'>{message['content']}</message>" for message in conversation])

def _format_rewrite_style(attributes: list[dict[str, str]]) -> str:
    follow = [a["attribute"] for a in attributes if a["side"] == "follow"]
    avoid = [a["attribute"] for a in attributes if a["side"] == "avoid"]
    parts = []
    if follow:
        parts.append("VERY: " + ", ".join(follow))
    if avoid:
        parts.append("NOT AT ALL: " + ", ".join(avoid))
    return "\n".join(parts)

def _build_requests_for_user(
    user_spec: dict,
    conversations_with_ids: list,
    first_user_turn_only: bool = False,
) -> tuple[list, list]:
    """Returns (requests, conv_data) where conv_data holds
    (conversation_id, original_conversation, request_indices) with indices local to `requests`."""
    requests = []
    conv_data = []
    for conv_idx in user_spec["conv_indices"]:
        conversation_id, conversation = conversations_with_ids[conv_idx]
        request_indices = []
        user_turn_indices = range(0, len(conversation), 2)
        if first_user_turn_only:
            # The benchmark only consumes each conversation's first user
            # message, so with first_user_turn_only the later turns are not rewritten.
            user_turn_indices = range(0, min(len(conversation), 1), 2)
        for i in user_turn_indices:
            preceding_context = conversation[:i]
            local_req_idx = len(requests)
            requests.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        conversation=_format_conversation(preceding_context),
                        prompt_to_rewrite=conversation[i]["content"],
                        formatted_attributes=_format_rewrite_style(user_spec["rewrite_style_attributes"]),
                    ),
                },
            ])
            request_indices.append((local_req_idx, i))
        conv_data.append((conversation_id, conversation, request_indices))
    return requests, conv_data

def _reconstruct_user(user_spec: dict, conv_data: list, responses: list, offset: int) -> dict | None:
    """Reconstruct a user's conversations from responses starting at `offset`;
    returns None if any generation failed."""
    all_local_indices = [
        local_req_idx
        for _, _, request_indices in conv_data
        for local_req_idx, _ in request_indices
    ]
    if any(
        not responses[offset + idx].content or not responses[offset + idx].finish_reason_stop
        for idx in all_local_indices
    ):
        return None

    rewritten_conversations = []
    for conversation_id, original_conversation, request_indices in conv_data:
        rewritten = [msg.copy() for msg in original_conversation]
        for local_req_idx, turn_idx in request_indices:
            rewritten[turn_idx] = {
                "role": "user",
                "content": responses[offset + local_req_idx].content,
            }
        rewritten_conversations.append({
            "conversation_id": conversation_id,
            "messages": rewritten,
        })

    return {
        "user_idx": user_spec["user_idx"],
        "rewrite_style_attributes": user_spec["rewrite_style_attributes"],
        "conversations": rewritten_conversations,
    }

def _process_chunk(
    chunk_user_specs: list,
    chunk_conv_data: list,
    chunk_request_offsets: list,
    chunk_requests: list,
    llm: LLMHelper,
    out_files: dict,
    num_written: dict,
    num_target: dict,
    num_discarded: dict,
) -> None:
    """Generate one chunk, reconstruct users, append valid ones to out_files."""
    print(f"  Generating {len(chunk_requests)} requests for {len(chunk_user_specs)} users...", flush=True)
    responses = llm.generate(chunk_requests)

    for user_spec, conv_data, offset in zip(chunk_user_specs, chunk_conv_data, chunk_request_offsets):
        split = user_spec["split"]
        if num_written[split] >= num_target[split]:
            continue  # Already have enough valid users for this split

        result = _reconstruct_user(user_spec, conv_data, responses, offset)
        if result is None:
            num_discarded[split] += 1
        else:
            out_files[split].write(json.dumps(result) + "\n")
            out_files[split].flush()
            num_written[split] += 1

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    oversample_factor = 1.05
    num_train_users = cfg.num_train_users
    num_sampled_train_users = int(num_train_users * oversample_factor)
    num_test_users = cfg.num_test_users
    num_sampled_test_users = int(num_test_users * oversample_factor)
    num_conversations_per_user = cfg.num_conversations_per_user
    num_attributes_per_user = cfg.num_attributes_per_user
    attributes = list(cfg.attributes)
    seed = cfg.get("seed", 42)
    chunk_size = cfg.get("chunk_size", 100_000)
    first_user_turn_only = bool(cfg.get("first_user_turn_only", False))

    train_conversations_with_ids = load_ultrachat_conversations_with_ids(
        split="train_sft",
        seed=seed,
    )
    test_conversations_with_ids = load_ultrachat_conversations_with_ids(
        split="test_sft",
        seed=seed,
    )

    llm: LLMHelper = instantiate(cfg.llm)
    llm.load()

    # User specs are lightweight: attributes + conversation indices only.
    user_specs = []
    rng = random.Random(seed)
    global_conv_idx = 0
    for split in ["train", "test"]:
        conversations_with_ids = train_conversations_with_ids if split == "train" else test_conversations_with_ids
        num_sampled_users = num_sampled_train_users if split == "train" else num_sampled_test_users
        print(f"[{split}] Building specs for {num_sampled_users} users")
        for usr_idx in tqdm(range(num_sampled_users), desc=f"[{split}] Building user specs"):
            sampled_attrs = rng.sample(attributes, num_attributes_per_user)
            rewrite_style_attributes = [
                {"attribute": attr, "side": rng.choice(["follow", "avoid"])}
                for attr in sampled_attrs
            ]
            conv_indices = []
            for _ in range(num_conversations_per_user):
                conv_indices.append(global_conv_idx % len(conversations_with_ids))
                global_conv_idx += 1
            user_specs.append({
                "split": split,
                "user_idx": usr_idx,
                "rewrite_style_attributes": rewrite_style_attributes,
                "conv_indices": conv_indices,
            })

    train_out_path = Path(get_original_cwd()) / cfg["train_output_path"]
    test_out_path = Path(get_original_cwd()) / cfg["test_output_path"]
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    test_out_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = {"train": 0, "test": 0}
    num_discarded = {"train": 0, "test": 0}
    num_target = {"train": num_train_users, "test": num_test_users}

    with open(train_out_path, "w") as train_f, open(test_out_path, "w") as test_f:
        out_files = {"train": train_f, "test": test_f}

        chunk_user_specs = []
        chunk_conv_data = []  # parallel to chunk_user_specs
        chunk_request_offsets = []  # parallel: start index in chunk_requests
        chunk_requests = []

        chunk_num = 0
        for spec in user_specs:
            split = spec["split"]
            conversations_with_ids = (
                train_conversations_with_ids if split == "train" else test_conversations_with_ids
            )

            user_requests, conv_data = _build_requests_for_user(
                spec,
                conversations_with_ids,
                first_user_turn_only=first_user_turn_only,
            )

            chunk_request_offsets.append(len(chunk_requests))
            chunk_requests.extend(user_requests)
            chunk_user_specs.append(spec)
            chunk_conv_data.append(conv_data)

            if len(chunk_requests) >= chunk_size:
                chunk_num += 1
                print(f"\n[chunk {chunk_num}] {len(chunk_requests)} requests, {len(chunk_user_specs)} users "
                      f"(train: {num_written['train']}/{num_target['train']}, "
                      f"test: {num_written['test']}/{num_target['test']})", flush=True)
                _process_chunk(
                    chunk_user_specs, chunk_conv_data, chunk_request_offsets,
                    chunk_requests, llm, out_files, num_written, num_target, num_discarded,
                )
                chunk_user_specs = []
                chunk_conv_data = []
                chunk_request_offsets = []
                chunk_requests = []

        if chunk_requests:
            chunk_num += 1
            print(f"\n[chunk {chunk_num}] {len(chunk_requests)} requests, {len(chunk_user_specs)} users "
                  f"(train: {num_written['train']}/{num_target['train']}, "
                  f"test: {num_written['test']}/{num_target['test']})", flush=True)
            _process_chunk(
                chunk_user_specs, chunk_conv_data, chunk_request_offsets,
                chunk_requests, llm, out_files, num_written, num_target, num_discarded,
            )

    for split in ["train", "test"]:
        path = train_out_path if split == "train" else test_out_path
        print(f"[{split}] Kept {num_written[split]}/{num_target[split]} users, "
              f"discarded {num_discarded[split]} due to failed generations")
        print(f"[{split}] Saved to {path}")

    llm.unload()

    if bool(cfg.get("require_exact_user_counts", False)):
        shortfalls = {
            split: num_target[split] - num_written[split]
            for split in ["train", "test"]
            if num_written[split] != num_target[split]
        }
        if shortfalls:
            raise RuntimeError(
                "Synthetic generation did not produce the requested number "
                f"of users: {shortfalls}"
            )

if __name__ == "__main__":
    main()
