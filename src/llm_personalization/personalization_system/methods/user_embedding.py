"""User-embedding personalization: history -> frozen embedder -> projector ->
one soft prompt token prepended to the (LoRA-tuned) LLM, trained with SimPO or
DPO on best/worst cached responses."""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..prompts import format_history
from .base import PersonalizationSystem
from .common import encode, load_embedder, pick_best_worst, release_embedder


# ----------------------------- preference data -----------------------------


def _build_pairs(
    dataset,
    user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]],
) -> list[dict]:
    """Returns list of {history_text, current_messages, chosen, rejected} per user."""
    pairs = []
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
        chosen, rejected = picked
        pairs.append({
            "user_id": user.user_id,
            "history_text": format_history(user.history),
            "current_messages": user.current_messages,
            "chosen": chosen,
            "rejected": rejected,
        })
    if skipped:
        print(f"[UserEmbedding] skipped {skipped} users without usable cached scores")
    return pairs


# ----------------------------- model -----------------------------


class Projector(nn.Module):
    """Small MLP from embedder dim -> LLM hidden size, producing one soft token."""
    def __init__(self, in_dim: int, hidden_size: int, mlp_hidden: int | None = None):
        super().__init__()
        h = mlp_hidden or hidden_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) -> (B, hidden)
        return self.net(x)


def _encode_histories_grad(embedder, histories: list[str], device) -> torch.Tensor:
    """Differentiable (B, D) encoding, used when train_embedder=True."""
    features = embedder.tokenize(histories)
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    out = embedder(features)
    return out["sentence_embedding"]


# ----------------------------- training utils -----------------------------


def _as_ids(out) -> list[int]:
    # transformers may return BatchEncoding/Encoding; coerce to list[int]
    if hasattr(out, "ids"):
        out = out.ids
    elif hasattr(out, "input_ids"):
        out = out.input_ids
    return list(out)


def _tokenize_prompt(tokenizer, current_messages: list[dict]) -> list[int]:
    return _as_ids(tokenizer.apply_chat_template(
        current_messages, add_generation_prompt=True, tokenize=True
    ))


def _tokenize_response(tokenizer, response: str) -> list[int]:
    ids = _as_ids(tokenizer.encode(response, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return ids


def _logp_response(
    model,
    embed_layer: nn.Embedding,
    soft_tokens: torch.Tensor,        # (B, hidden)
    prompt_ids: list[list[int]],      # B prompts
    response_ids: list[list[int]],    # B responses
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sum_logp, num_response_tokens) per item for
    inputs_embeds = [soft, prompt_embeds, response_embeds]."""
    B = len(prompt_ids)
    seq_lens = [1 + len(p) + len(r) for p, r in zip(prompt_ids, response_ids)]
    max_len = max(seq_lens)

    token_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)

    for i, (p, r) in enumerate(zip(prompt_ids, response_ids)):
        # positions: [0]=soft, [1..1+len(p))=prompt, then response
        token_ids[i, 1 : 1 + len(p)] = torch.tensor(p, device=device)
        token_ids[i, 1 + len(p) : 1 + len(p) + len(r)] = torch.tensor(r, device=device)
        attention_mask[i, : 1 + len(p) + len(r)] = 1
        response_mask[i, 1 + len(p) : 1 + len(p) + len(r)] = True

    # concat instead of in-place write: position 0 is a view of the embedding
    # weight, a leaf with requires_grad
    word_embeds_rest = embed_layer(token_ids[:, 1:]).to(dtype)
    soft = soft_tokens.to(dtype).unsqueeze(1)
    word_embeds = torch.cat([soft, word_embeds_rest], dim=1)

    out = model(
        inputs_embeds=word_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = out.logits
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = token_ids[:, 1:]
    gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    loss_mask = response_mask[:, 1:]

    sum_logp = (gathered * loss_mask).sum(dim=1)
    n_tokens = loss_mask.sum(dim=1).clamp(min=1)
    return sum_logp, n_tokens


def _logp_response_no_soft(
    model,
    embed_layer: nn.Embedding,
    prompt_ids: list[list[int]],
    response_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like `_logp_response` but without the soft-token slot; scores under the
    reference (base) model."""
    B = len(prompt_ids)
    seq_lens = [len(p) + len(r) for p, r in zip(prompt_ids, response_ids)]
    max_len = max(seq_lens)

    token_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    for i, (p, r) in enumerate(zip(prompt_ids, response_ids)):
        token_ids[i, : len(p)] = torch.tensor(p, device=device)
        token_ids[i, len(p) : len(p) + len(r)] = torch.tensor(r, device=device)
        attention_mask[i, : len(p) + len(r)] = 1
        response_mask[i, len(p) : len(p) + len(r)] = True

    word_embeds = embed_layer(token_ids).to(dtype)
    out = model(inputs_embeds=word_embeds, attention_mask=attention_mask, use_cache=False)
    logits = out.logits
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = token_ids[:, 1:]
    gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    loss_mask = response_mask[:, 1:]
    sum_logp = (gathered * loss_mask).sum(dim=1)
    n_tokens = loss_mask.sum(dim=1).clamp(min=1)
    return sum_logp, n_tokens


def _simpo_loss(
    sum_logp_chosen, n_chosen,
    sum_logp_rejected, n_rejected,
    beta: float, gamma: float,
) -> torch.Tensor:
    avg_chosen = sum_logp_chosen / n_chosen
    avg_rejected = sum_logp_rejected / n_rejected
    margin = beta * (avg_chosen - avg_rejected) - gamma
    loss = -F.logsigmoid(margin)
    return loss.mean(), avg_chosen.detach().mean(), avg_rejected.detach().mean()


def _dpo_loss(
    sum_logp_chosen, sum_logp_rejected,
    ref_sum_logp_chosen, ref_sum_logp_rejected,
    beta: float,
) -> torch.Tensor:
    """Standard DPO with the base LLM (LoRA-off, no soft token) as reference;
    sum-of-log-probs, not length-normalized."""
    pol_diff = sum_logp_chosen - sum_logp_rejected
    ref_diff = ref_sum_logp_chosen - ref_sum_logp_rejected
    margin = beta * (pol_diff - ref_diff)
    loss = -F.logsigmoid(margin)
    avg_c = (sum_logp_chosen - ref_sum_logp_chosen).detach().mean()
    avg_r = (sum_logp_rejected - ref_sum_logp_rejected).detach().mean()
    return loss.mean(), avg_c, avg_r


# ----------------------------- main class -----------------------------


class UserEmbeddingSystem(PersonalizationSystem):
    def __init__(
        self,
        gen_model: str,
        embedder_model: str,
        projector_mlp_hidden: int | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        batch_size: int = 1,
        grad_accum_steps: int = 8,
        epochs: int = 1,
        lr: float = 5e-5,
        max_prompt_tokens: int = 1024,
        max_response_tokens: int = 1024,
        loss_type: str = "simpo",   # "simpo" | "dpo"
        beta: float = 2.0,          # SimPO beta
        gamma: float = 1.0,         # SimPO margin
        dpo_beta: float = 0.1,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        infer_batch_size: int = 4,
        max_new_tokens: int = 1024,
        infer_dtype: str = "bfloat16",
        embedder_batch_size: int = 32,
        embedder_max_seq_length: int | None = 512,
        train_embedder: bool = False,
        embedder_lr: float | None = None,  # default = lr
    ):
        self.gen_model = gen_model
        self.embedder_model = embedder_model

        self.projector_mlp_hidden = projector_mlp_hidden
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # cast away Hydra's ListConfig so peft's JSON serializer can save it
        self.lora_target_modules = list(lora_target_modules) if lora_target_modules else ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.lr = lr
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        if loss_type not in ("simpo", "dpo"):
            raise ValueError(f"loss_type must be 'simpo' or 'dpo', got {loss_type!r}")
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.dpo_beta = dpo_beta
        self.seed = seed
        self.gradient_checkpointing = gradient_checkpointing

        self.infer_batch_size = infer_batch_size
        self.max_new_tokens = max_new_tokens
        self.infer_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[infer_dtype]

        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_seq_length = embedder_max_seq_length
        self.train_embedder = train_embedder
        self.embedder_lr = embedder_lr if embedder_lr is not None else lr

    # --------------- helpers ---------------

    def _load_llm(self, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(self.gen_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.gen_model, torch_dtype=dtype, trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        return tokenizer, model

    def _make_lora(self, model):
        cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        return model

    # --------------- train ---------------

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
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        print("[UserEmbedding] 1. Building preference pairs...")
        train_pairs = _build_pairs(dataset, user_id_to_candidate_scores)
        val_pairs = (
            _build_pairs(val_dataset, val_user_id_to_candidate_scores)
            if (val_dataset is not None and val_user_id_to_candidate_scores is not None) else []
        )
        print(f"[UserEmbedding]    train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

        embedder = load_embedder(self.embedder_model, self.embedder_max_seq_length)
        log_gpu_usage("After embedder load")
        if not self.train_embedder:
            print("[UserEmbedding] 2. Encoding histories with frozen embedder (one-shot)...")
            train_emb = encode(embedder, [p["history_text"] for p in train_pairs], self.embedder_batch_size, normalize=False)
            val_emb = (
                encode(embedder, [p["history_text"] for p in val_pairs], self.embedder_batch_size, normalize=False)
                if val_pairs else np.zeros((0, train_emb.shape[1]), dtype=np.float32)
            )
            emb_dim = train_emb.shape[1]
            release_embedder(embedder)
            embedder = None
            log_gpu_usage("After embedder unload")
        else:
            print("[UserEmbedding] 2. train_embedder=True — embedder stays loaded, encoded per batch with grad.")
            with torch.no_grad():
                probe = _encode_histories_grad(embedder, [train_pairs[0]["history_text"]], device="cuda")
            emb_dim = int(probe.shape[1])
            train_emb = None
            val_emb = None
            for p in embedder.parameters():
                p.requires_grad = True
            log_gpu_usage("After embedder configure")

        print(f"[UserEmbedding] 3. Loading LLM ({self.gen_model}) + LoRA + projector...")
        log_gpu_usage("Before LLM load")
        tokenizer, model = self._load_llm(self.infer_dtype)
        for p in model.parameters():
            p.requires_grad = False
        model = self._make_lora(model)
        model.print_trainable_parameters()
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        hidden_size = model.config.hidden_size
        # projector stays fp32; its output is cast to the LLM dtype at injection
        projector = Projector(emb_dim, hidden_size, self.projector_mlp_hidden).cuda()
        for p in projector.parameters():
            p.requires_grad = True
        log_gpu_usage("After LLM+LoRA+projector load")

        print("[UserEmbedding] 4. Tokenizing prompts and responses...")
        def _prep(pairs):
            prompts, chosens, rejecteds = [], [], []
            for p in pairs:
                pid = _tokenize_prompt(tokenizer, p["current_messages"])[: self.max_prompt_tokens]
                cid = _tokenize_response(tokenizer, p["chosen"])[: self.max_response_tokens]
                rid = _tokenize_response(tokenizer, p["rejected"])[: self.max_response_tokens]
                prompts.append(pid); chosens.append(cid); rejecteds.append(rid)
            return prompts, chosens, rejecteds
        tr_p, tr_c, tr_r = _prep(train_pairs)
        va_p, va_c, va_r = _prep(val_pairs)

        embed_layer = model.get_input_embeddings()
        main_params = list(projector.parameters()) + [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": main_params, "lr": self.lr}]
        if self.train_embedder:
            param_groups.append({"params": list(embedder.parameters()), "lr": self.embedder_lr})
        optimizer = torch.optim.AdamW(param_groups)
        all_params = main_params + (list(embedder.parameters()) if self.train_embedder else [])
        device = next(model.parameters()).device

        def _ref_logps(pids, ys):
            """Score responses under the reference (adapter off, no soft token)."""
            with torch.no_grad():
                ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else None
                if ctx is not None:
                    with ctx:
                        sl, _ = _logp_response_no_soft(
                            model, embed_layer, pids, ys, tokenizer.pad_token_id, device, self.infer_dtype
                        )
                else:
                    sl, _ = _logp_response_no_soft(
                        model, embed_layer, pids, ys, tokenizer.pad_token_id, device, self.infer_dtype
                    )
            return sl.detach()

        def _step_loss(idx, split):
            if split == "train":
                pids, cids, rids = [tr_p[i] for i in idx], [tr_c[i] for i in idx], [tr_r[i] for i in idx]
                hist_texts = [train_pairs[i]["history_text"] for i in idx]
                emb_arr = train_emb
            else:
                pids, cids, rids = [va_p[i] for i in idx], [va_c[i] for i in idx], [va_r[i] for i in idx]
                hist_texts = [val_pairs[i]["history_text"] for i in idx]
                emb_arr = val_emb
            if self.train_embedder:
                emb = _encode_histories_grad(embedder, hist_texts, device).to(torch.float32)
            else:
                emb = torch.from_numpy(emb_arr[idx]).to(device, dtype=torch.float32)
            soft = projector(emb)
            sl_c, n_c = _logp_response(model, embed_layer, soft, pids, cids, tokenizer.pad_token_id, device, self.infer_dtype)
            sl_r, n_r = _logp_response(model, embed_layer, soft, pids, rids, tokenizer.pad_token_id, device, self.infer_dtype)
            if self.loss_type == "simpo":
                loss, avg_c, avg_r = _simpo_loss(sl_c, n_c, sl_r, n_r, self.beta, self.gamma)
            else:  # dpo
                ref_sl_c = _ref_logps(pids, cids)
                ref_sl_r = _ref_logps(pids, rids)
                loss, avg_c, avg_r = _dpo_loss(sl_c, sl_r, ref_sl_c, ref_sl_r, self.dpo_beta)
            with torch.no_grad():
                pref_acc = ((sl_c / n_c) > (sl_r / n_r)).float().sum().item()
            return loss, avg_c, avg_r, pref_acc

        def _eval_val_loss():
            if not val_pairs:
                return None
            model.eval(); projector.eval()
            if self.train_embedder: embedder.eval()
            with torch.no_grad():
                total, n = 0.0, 0
                for s in range(0, len(val_pairs), self.batch_size):
                    e = min(s + self.batch_size, len(val_pairs))
                    idx = list(range(s, e))
                    loss, _, _, _ = _step_loss(idx, "val")
                    total += loss.item() * (e - s); n += (e - s)
            model.train(); projector.train()
            if self.train_embedder: embedder.train()
            return total / max(1, n)

        loss_descr = (f"SimPO (β={self.beta}, γ={self.gamma})" if self.loss_type == "simpo"
                      else f"DPO (β={self.dpo_beta})")
        print(f"[UserEmbedding] 5. Training loss={loss_descr}: epochs={self.epochs}, "
              f"batch={self.batch_size}, grad_accum={self.grad_accum_steps}")
        model.train(); projector.train()
        if self.train_embedder: embedder.train()
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            order = rng.permutation(len(train_pairs))
            running, running_acc, running_n = 0.0, 0.0, 0
            optimizer.zero_grad()
            for step, idx_start in enumerate(tqdm(range(0, len(order), self.batch_size), desc=f"epoch {epoch}")):
                idx = order[idx_start : idx_start + self.batch_size].tolist()
                loss, avg_c, avg_r, pref_acc = _step_loss(idx, "train")
                (loss / self.grad_accum_steps).backward()
                running += loss.item() * len(idx); running_n += len(idx)
                running_acc += pref_acc
                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step(); optimizer.zero_grad()
                if (step + 1) % 50 == 0:
                    print(f"    step {step+1}: loss={running/running_n:.4f} pref_acc={running_acc/running_n:.3f} "
                          f"avg_c={avg_c.item():.2f} avg_r={avg_r.item():.2f}")
            optimizer.step(); optimizer.zero_grad()
            val_loss = _eval_val_loss()
            tag = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"  Epoch {epoch}: train_loss={running/running_n:.4f} pref_acc={running_acc/running_n:.3f}{tag}")

        print(f"[UserEmbedding] 6. Saving LoRA + projector + emb_dim to {save_path}")
        model.save_pretrained(save_path / "lora")
        tokenizer.save_pretrained(save_path / "lora")
        torch.save(projector.state_dict(), save_path / "projector.pt")
        if self.train_embedder:
            embedder.save(str(save_path / "embedder"))
            release_embedder(embedder)
            embedder = None
        with open(save_path / "config.json", "w") as f:
            json.dump({
                "gen_model": self.gen_model,
                "embedder_model": self.embedder_model,
                "train_embedder": self.train_embedder,
                "emb_dim": int(emb_dim),
                "hidden_size": int(hidden_size),
                "projector_mlp_hidden": self.projector_mlp_hidden,
                "lora_r": self.lora_r,
                "loss_type": self.loss_type,
                "beta": self.beta,
                "gamma": self.gamma,
                "dpo_beta": self.dpo_beta,
            }, f, indent=2)

        del model, projector, embed_layer, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------- evaluate ---------------

    def evaluate(
        self,
        dataset,
        load_path: Path,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug=None,
    ) -> list[str]:
        load_path = Path(load_path)
        with open(load_path / "config.json") as f:
            cfg = json.load(f)
        emb_dim = cfg["emb_dim"]
        hidden_size = cfg["hidden_size"]

        print("[UserEmbedding] 1. Encoding test histories...")
        log_gpu_usage("Before embedder load")
        embedder_path = load_path / "embedder"
        if cfg.get("train_embedder", False) and embedder_path.exists():
            print(f"[UserEmbedding]    using fine-tuned embedder at {embedder_path}")
            embedder = load_embedder(str(embedder_path), self.embedder_max_seq_length)
        else:
            embedder = load_embedder(self.embedder_model, self.embedder_max_seq_length)
        test_emb = encode(embedder, [format_history(u.history) for u in dataset], self.embedder_batch_size, normalize=False)
        release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        print("[UserEmbedding] 2. Loading LLM + LoRA adapter + projector...")
        log_gpu_usage("Before LLM load")
        tokenizer, base = self._load_llm(self.infer_dtype)
        model = PeftModel.from_pretrained(base, str(load_path / "lora"))
        model.eval()
        projector = Projector(emb_dim, hidden_size, self.projector_mlp_hidden).cuda()
        projector.load_state_dict(torch.load(load_path / "projector.pt", map_location="cuda"))
        projector.eval()
        embed_layer = model.get_input_embeddings()
        device = next(model.parameters()).device
        log_gpu_usage("After LLM+LoRA+projector load")

        print("[UserEmbedding] 3. Generating (HF model.generate with inputs_embeds)...")
        responses: list[str] = [""] * len(dataset)
        prompts_ids = [_tokenize_prompt(tokenizer, u.current_messages)[: self.max_prompt_tokens] for u in dataset]
        all_indices = list(range(len(dataset)))

        for s in tqdm(range(0, len(all_indices), self.infer_batch_size), desc="generate"):
            idx = all_indices[s : s + self.infer_batch_size]
            B = len(idx)
            emb = torch.from_numpy(test_emb[idx]).to(device, dtype=torch.float32)
            with torch.no_grad():
                soft = projector(emb).to(self.infer_dtype)  # (B, H)
            batch_prompts = [prompts_ids[i] for i in idx]
            max_p = max(len(p) for p in batch_prompts)
            # per row, right-aligned: [left pad..., soft, prompt tokens]
            max_len = 1 + max_p
            input_ids = torch.full((B, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            soft_pos = torch.empty(B, dtype=torch.long, device=device)
            for j, p in enumerate(batch_prompts):
                offset = max_len - len(p)
                soft_pos[j] = offset - 1
                input_ids[j, offset:] = torch.tensor(p, device=device)
                attention_mask[j, offset - 1 :] = 1
            # combine via mask instead of in-place writes on leaf views
            base_embeds = embed_layer(input_ids).to(self.infer_dtype)
            soft_full = torch.zeros_like(base_embeds)
            soft_full[torch.arange(B, device=device), soft_pos] = soft
            mask = torch.zeros((B, max_len, 1), dtype=base_embeds.dtype, device=device)
            mask[torch.arange(B, device=device), soft_pos] = 1.0
            word_embeds = base_embeds * (1.0 - mask) + soft_full * mask
            with torch.no_grad():
                out_ids = model.generate(
                    inputs_embeds=word_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # with inputs_embeds, generate returns only the new tokens
            for j, ids in zip(idx, out_ids):
                responses[j] = tokenizer.decode(ids, skip_special_tokens=True)

        del model, base, projector, embed_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_gpu_usage("After eval unload")
        return responses
