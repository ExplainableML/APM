from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from llm_personalization.utils.io import iter_jsonl


@dataclass
class CachedUser:
    user_id: str
    gt_user_attributes: list[dict]
    history: list[str]
    current_messages: list[dict]
    responses: dict[tuple[str, str], str] = field(default_factory=dict)
    ratings: dict[tuple[str, str, str, str], float] = field(default_factory=dict)


class CachedDataset:
    """Joins synthetic conversations + attribute responses + judge ratings on
    user_id. History per user = first user-message of each non-held-out
    conversation (up to `history_max_len`)."""

    def __init__(
        self,
        split: Literal["train", "test"],
        conversations_path: Path | str,
        responses_path: Path | str,
        ratings_path: Path | str,
        history_max_len: int | None = None,
        limit: int | None = None,
    ):
        self.split = split
        self.history_max_len = history_max_len

        # responses (also provide user_id, current_messages, gt)
        users: dict[str, CachedUser] = {}
        order: list[str] = []
        for rec in iter_jsonl(Path(responses_path)):
            uid = str(rec["user_id"])
            if uid in users:
                continue
            users[uid] = CachedUser(
                user_id=uid,
                gt_user_attributes=rec["gt_user_attributes"],
                history=[],
                current_messages=rec["current_messages"],
                responses={
                    (r["attribute"], r["side"]): r["response"] for r in rec["responses"]
                },
            )
            order.append(uid)
            if limit is not None and len(order) >= limit:
                break

        # ratings (the ratings dir mixes splits; filter by `split` field)
        for rec in iter_jsonl(Path(ratings_path)):
            if rec.get("split", split) != split:
                continue
            uid = str(rec["user_id"])
            if uid not in users:
                continue
            user = users[uid]
            for r in rec["ratings"]:
                gen_attr, gen_side = r["gen_attribute"], r["gen_side"]
                for s in r["scores"]:
                    if s.get("score") is None:
                        continue
                    user.ratings[(gen_attr, gen_side, s["attribute"], s["side"])] = float(s["score"])

        # histories from synthetic conversations
        for rec in iter_jsonl(Path(conversations_path)):
            uid = str(rec["user_idx"])
            if uid not in users:
                continue
            user = users[uid]
            history = []
            # Held-out is the last conversation; use everything before it.
            convs = rec["conversations"][:-1]
            if history_max_len is not None:
                convs = convs[:history_max_len]
            for conv in convs:
                first_user = next(
                    (m["content"] for m in conv["messages"] if m["role"] == "user"),
                    None,
                )
                if first_user is not None:
                    history.append(first_user)
            user.history = history

        # Drop users missing any of the three sources
        kept_order = [
            uid for uid in order
            if users[uid].history and users[uid].responses and users[uid].ratings
        ]
        dropped = len(order) - len(kept_order)
        if dropped:
            print(f"[CachedDataset/{split}] dropped {dropped}/{len(order)} users missing data")

        self._users: list[CachedUser] = [users[uid] for uid in kept_order]
        self._by_id: dict[str, CachedUser] = {u.user_id: u for u in self._users}
        print(
            f"[CachedDataset/{split}] loaded {len(self._users)} users "
            f"(avg history={sum(len(u.history) for u in self._users)/max(1,len(self._users)):.1f}, "
            f"avg responses={sum(len(u.responses) for u in self._users)/max(1,len(self._users)):.1f}, "
            f"avg ratings={sum(len(u.ratings) for u in self._users)/max(1,len(self._users)):.1f})"
        )

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, i: int) -> CachedUser:
        return self._users[i]

    def __iter__(self):
        return iter(self._users)

    def by_id(self, user_id: str) -> CachedUser:
        return self._by_id[str(user_id)]

    def split_off_val(self, val_size: int) -> "CachedDataset":
        """Drops the last `val_size` users from self and returns them as a new
        CachedDataset (shared user objects, no copying)."""
        if val_size <= 0:
            return _empty_like(self)
        if val_size >= len(self._users):
            raise ValueError(f"val_size={val_size} >= dataset size={len(self._users)}")
        val_users = self._users[-val_size:]
        self._users = self._users[:-val_size]
        self._by_id = {u.user_id: u for u in self._users}
        val_ds = _empty_like(self)
        val_ds._users = val_users
        val_ds._by_id = {u.user_id: u for u in val_users}
        print(f"[CachedDataset/{self.split}] split off val: train={len(self._users)} val={len(val_users)}")
        return val_ds


def _empty_like(ds: CachedDataset) -> CachedDataset:
    """CachedDataset shell with the same metadata as `ds` but no users."""
    new = CachedDataset.__new__(CachedDataset)
    new.split = ds.split
    new.history_max_len = ds.history_max_len
    new._users = []
    new._by_id = {}
    return new
