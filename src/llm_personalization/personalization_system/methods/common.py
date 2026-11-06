"""Helpers shared by the retrieval- and embedding-based methods."""
from __future__ import annotations

import gc
import math

import numpy as np
import torch


def pick_best_worst(user, candidate_scores: dict[tuple[str, str], float]) -> tuple[str, str] | None:
    """(best_response, worst_response) among the user's cached candidates by
    user-fit score, or None if there are fewer than two distinct candidates."""
    if not candidate_scores:
        return None
    best_attr_side, best_score = None, -math.inf
    worst_attr_side, worst_score = None, math.inf
    for (attr, side), s in candidate_scores.items():
        if (attr, side) not in user.responses:
            continue
        if s > best_score:
            best_attr_side, best_score = (attr, side), s
        if s < worst_score:
            worst_attr_side, worst_score = (attr, side), s
    if best_attr_side is None or worst_attr_side is None or best_attr_side == worst_attr_side:
        return None
    return user.responses[best_attr_side], user.responses[worst_attr_side]


def first_user_prompt(current_messages: list[dict]) -> str:
    return current_messages[0]["content"] if current_messages else ""


def load_embedder(model_name: str, max_seq_length: int | None = None):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    if max_seq_length is not None:
        embedder.max_seq_length = max_seq_length
    return embedder


def release_embedder(embedder) -> None:
    """Moves the embedder off the GPU before dropping it: `del + empty_cache`
    alone leaves ~3 GiB pinned, enough to OOM a 27B model load right after."""
    try:
        embedder.to("cpu")
    except Exception:
        pass
    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def encode(embedder, texts: list[str], batch_size: int, normalize: bool = True) -> np.ndarray:
    return embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    ).astype(np.float32)


def top_k_neighbors(queries: np.ndarray, index: np.ndarray, k: int) -> np.ndarray:
    """(n_queries, k) indices into `index` by descending dot product
    (cosine similarity for L2-normalized embeddings)."""
    sims = queries @ index.T
    k = min(k, sims.shape[1])
    top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)
