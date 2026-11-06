"""Routing personalization: train a text classifier on user history to predict
the (attribute, side) pair that maximizes the user simulator's score on
cached candidate responses. At eval time, the predicted (attr, side) is
inserted into the system prompt and the response is generated.

Label-selection variants:
  - abs:           per attr, score = candidate_scores[(attr, "follow")];
                    pick attr with max |score - 5.5|; side = sign of centered.
  - abs_two_sided: pick (attr, side) with max raw candidate score.
  - margin:        per attr, margin = score(follow) - score(avoid);
                    pick attr with max |margin|; side = sign(margin).
  - regression:    soft label = softmax over candidate scores of all 2K
                    (attr, side) candidates; trained with soft-label CE.
  - mse:           true regression: target = raw candidate scores of all 2K
                    (attr, side) candidates (NaN where uncached); trained with
                    masked MSE; predicted (attr, side) = argmax of regressed scores.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import torch

from llm_personalization.classification_model.text_classification_model import (
    TextClassificationModel,
)
from llm_personalization.judge.judge import SCORE_MIDPOINT
from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..prompts import format_history, format_system_prompt
from .base import PersonalizationSystem


def _label_idx(attr_to_idx: dict[str, int], n_attrs: int, attr: str, side: str) -> int:
    return attr_to_idx[attr] + (n_attrs if side == "follow" else 0)


def _decode_label(label: int, attributes: list[str]) -> tuple[str, str]:
    n = len(attributes)
    return attributes[label % n], ("follow" if label // n == 1 else "avoid")


class RoutingSystem(PersonalizationSystem):
    def __init__(
        self,
        text_classification_model_config: dict,
        llm_helper_config: dict,
        attributes: list[str],
        label_selection: Literal["abs", "abs_two_sided", "margin", "regression", "mse"] = "abs",
        text_classification_model_train_kwargs: dict | None = None,
        predict_batch_size: int | None = None,
    ):
        self.text_classification_model_config = text_classification_model_config
        self.llm_helper_config = llm_helper_config
        self.attributes = list(attributes)
        self.label_selection = label_selection
        self.text_classification_model_train_kwargs = text_classification_model_train_kwargs or {}
        self.predict_batch_size = predict_batch_size

        self.text_classification_model = TextClassificationModel(
            **self.text_classification_model_config,
            num_classes=len(self.attributes) * 2,
        )
        self.llm_helper = LLMHelper(**self.llm_helper_config)

    # ------- label derivation (no LLM/judge calls; pure cache lookup) -------

    def _derive_labels(
        self,
        dataset,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]],
    ) -> tuple[dict[str, int], dict[str, list[float]] | None]:
        attr_to_idx = {a: i for i, a in enumerate(self.attributes)}
        n = len(self.attributes)
        labels: dict[str, int] = {}
        # Per-class float targets: softmax distribution for "regression",
        # raw scores (NaN = uncached candidate) for "mse".
        soft_labels: dict[str, list[float]] | None = (
            {} if self.label_selection in ("regression", "mse") else None
        )

        skipped = 0
        for user in dataset:
            scores = user_id_to_candidate_scores.get(user.user_id)
            if not scores:
                skipped += 1
                continue

            if self.label_selection == "abs":
                best_attr, best_centered = None, -1.0
                for attr in self.attributes:
                    s = scores.get((attr, "follow"))
                    if s is None:
                        continue
                    c = s - SCORE_MIDPOINT
                    if abs(c) > abs(best_centered) or best_attr is None:
                        best_attr, best_centered = attr, c
                if best_attr is None:
                    skipped += 1
                    continue
                labels[user.user_id] = _label_idx(
                    attr_to_idx, n, best_attr, "avoid" if best_centered < 0 else "follow"
                )

            elif self.label_selection == "abs_two_sided":
                best_attr, best_side, best_score = None, None, -math.inf
                for attr in self.attributes:
                    for side in ("follow", "avoid"):
                        s = scores.get((attr, side))
                        if s is None:
                            continue
                        if s > best_score:
                            best_attr, best_side, best_score = attr, side, s
                if best_attr is None:
                    skipped += 1
                    continue
                labels[user.user_id] = _label_idx(attr_to_idx, n, best_attr, best_side)

            elif self.label_selection == "margin":
                best_attr, best_margin = None, 0.0
                for attr in self.attributes:
                    f = scores.get((attr, "follow"))
                    a = scores.get((attr, "avoid"))
                    if f is None or a is None:
                        continue
                    m = f - a
                    if best_attr is None or abs(m) > abs(best_margin):
                        best_attr, best_margin = attr, m
                if best_attr is None:
                    skipped += 1
                    continue
                labels[user.user_id] = _label_idx(
                    attr_to_idx, n, best_attr, "follow" if best_margin > 0 else "avoid"
                )

            elif self.label_selection == "regression":
                vec = torch.full((2 * n,), float("-inf"))
                any_score = False
                for attr in self.attributes:
                    for side in ("follow", "avoid"):
                        s = scores.get((attr, side))
                        if s is None:
                            continue
                        any_score = True
                        vec[_label_idx(attr_to_idx, n, attr, side)] = s
                if not any_score:
                    skipped += 1
                    continue
                soft = torch.softmax(vec, dim=0)
                soft_labels[user.user_id] = soft.tolist()
                labels[user.user_id] = int(soft.argmax().item())

            elif self.label_selection == "mse":
                vec = torch.full((2 * n,), float("nan"))
                any_score = False
                for attr in self.attributes:
                    for side in ("follow", "avoid"):
                        s = scores.get((attr, side))
                        if s is None:
                            continue
                        any_score = True
                        vec[_label_idx(attr_to_idx, n, attr, side)] = s
                if not any_score:
                    skipped += 1
                    continue
                soft_labels[user.user_id] = vec.tolist()
                labels[user.user_id] = int(
                    torch.nan_to_num(vec, nan=float("-inf")).argmax().item()
                )

            else:
                raise ValueError(f"Unknown label_selection: {self.label_selection}")

        if skipped:
            print(f"[RoutingSystem] skipped {skipped} users with insufficient cached scores")

        if labels:
            dist: dict[int, int] = {}
            for v in labels.values():
                dist[v] = dist.get(v, 0) + 1
            print(f"[RoutingSystem] label distribution ({len(labels)} users, {len(dist)} classes used):")
            for label in sorted(dist):
                attr, side = _decode_label(label, self.attributes)
                print(f"    {label}: {attr} ({side}) = {dist[label]}")

        return labels, soft_labels

    # ----------------- label accuracy vs. ground-truth targets (logging only) -----------------

    def _log_judge_accuracy(
        self,
        labels: dict[str, int],
        gt: dict[str, list[dict]],
        tag: str,
    ) -> None:
        n = len(self.attributes)
        attr_to_idx = {a: i for i, a in enumerate(self.attributes)}
        correct = total = attr_only = 0
        for uid, lbl in labels.items():
            for g in gt.get(uid, []):
                if g["attribute"] not in attr_to_idx:
                    continue
                gt_lbl = _label_idx(attr_to_idx, n, g["attribute"], g["side"])
                total += 1
                if lbl == gt_lbl:
                    correct += 1
                if self.attributes[lbl % n] == g["attribute"]:
                    attr_only += 1
        if total:
            print(
                f"[RoutingSystem] {tag} ACC label={correct}/{total}={correct/total:.4f} "
                f"attr={attr_only}/{total}={attr_only/total:.4f}"
            )

    # -------------------- PersonalizationSystem --------------------

    def train(
        self,
        dataset,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]],
        save_path: Path,
        val_dataset=None,
        val_user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug: dict[str, list[dict]] | None = None,
        val_gt_for_debug: dict[str, list[dict]] | None = None,
    ) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"[RoutingSystem] 1. Deriving labels (selection={self.label_selection})...")
        labels, soft_labels = self._derive_labels(dataset, user_id_to_candidate_scores)
        if gt_for_debug:
            self._log_judge_accuracy(labels, gt_for_debug, tag="JUDGE")

        val_texts: list[str] | None = None
        val_targets: list | None = None
        if val_dataset is not None and val_user_id_to_candidate_scores is not None:
            val_labels, val_soft = self._derive_labels(val_dataset, val_user_id_to_candidate_scores)
            if val_gt_for_debug:
                self._log_judge_accuracy(val_labels, val_gt_for_debug, tag="JUDGE-VAL")
            val_texts, val_targets = [], []
            for user in val_dataset:
                if user.user_id not in val_labels:
                    continue
                val_texts.append(format_history(user.history))
                val_targets.append(
                    val_soft[user.user_id]
                    if self.label_selection in ("regression", "mse")
                    else val_labels[user.user_id]
                )

        print("[RoutingSystem] 2. Training text classifier...")
        texts: list[str] = []
        targets: list = []
        for user in dataset:
            if user.user_id not in labels:
                continue
            texts.append(format_history(user.history))
            if self.label_selection in ("regression", "mse"):
                targets.append(soft_labels[user.user_id])
            else:
                targets.append(labels[user.user_id])

        log_gpu_usage("Before classifier load")
        self.text_classification_model.load_untrained()
        self.text_classification_model.train(
            texts, targets,
            val_texts=val_texts, val_labels=val_targets,
            loss_fn="mse" if self.label_selection == "mse" else "ce",
            **self.text_classification_model_train_kwargs,
        )

        print("[RoutingSystem] 3. Saving classifier + attributes...")
        self.text_classification_model.save_to_file(str(save_path / "text_classification_model"))
        with open(save_path / "attributes.json", "w") as f:
            json.dump(self.attributes, f)

    def evaluate(
        self,
        dataset,
        load_path: Path,
        user_id_to_candidate_scores: dict[str, dict[tuple[str, str], float]] | None = None,
        gt_for_debug: dict[str, list[dict]] | None = None,
    ) -> list[str]:
        load_path = Path(load_path)
        with open(load_path / "attributes.json") as f:
            attributes: list[str] = json.load(f)

        print("[RoutingSystem] 1. Predicting labels...")
        self.text_classification_model.load_from_file(str(load_path / "text_classification_model"))
        texts = [format_history(u.history) for u in dataset]
        predictions = self.text_classification_model.predict(
            texts, batch_size=self.predict_batch_size
        )
        self.text_classification_model.unload()

        if gt_for_debug is not None:
            pred_labels = {u.user_id: int(p) for u, p in zip(dataset, predictions)}
            self._log_judge_accuracy(pred_labels, gt_for_debug, tag="TEST")

        print("[RoutingSystem] 2. Generating responses (single batched vLLM call)...")
        prompts = []
        for user, pred in zip(dataset, predictions):
            attr, side = _decode_label(int(pred), attributes)
            prompts.append(
                [{"role": "system", "content": format_system_prompt(attr, side)}]
                + user.current_messages
            )

        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        responses = self.llm_helper.generate(prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")
        return [r.content for r in responses]
