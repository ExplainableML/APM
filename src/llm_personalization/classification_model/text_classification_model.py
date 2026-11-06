import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import gc

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

class TextClassificationModel:
    def __init__(self, num_classes: int, pooling: str = "mean", base_model: str = "answerdotai/ModernBERT-base", max_length: int | None = None, dtype: str = "float32"):
        self.base_model = base_model
        self.num_classes = num_classes
        self.pooling = pooling
        self._max_length_override = max_length
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unknown dtype: {dtype}. Must be one of {list(DTYPE_MAP.keys())}")
        self.dtype = DTYPE_MAP[dtype]
        self.model = None
        self.tokenizer = None

    def _tokenize(self, texts: list[str], track_truncation: bool = False) -> dict:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        if track_truncation:
            untrunc = self.tokenizer(texts, padding=False, truncation=False)
            lengths = [len(ids) for ids in untrunc["input_ids"]]
            truncated = [(l - self.max_length) for l in lengths if l > self.max_length]
            self._trunc_total += len(lengths)
            self._trunc_count += len(truncated)
            self._trunc_excess += sum(truncated)
        return inputs

    def _report_truncation(self):
        if self._trunc_count > 0:
            avg_excess = self._trunc_excess / self._trunc_count
            print(f"  [Truncation] {self._trunc_count}/{self._trunc_total} sequences truncated "
                  f"(max_length={self.max_length}, avg excess tokens={avg_excess:.0f})")
        self._trunc_total = self._trunc_count = self._trunc_excess = 0

    def forward(self, texts: list[str]) -> torch.Tensor:
        inputs = self._tokenize(texts)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs).logits

    def predict(self, texts: list[str], batch_size: int | None = None) -> list[int]:
        if self.model is None:
            raise ValueError("Model not loaded")
        self.model.eval()
        if batch_size is None:
            with torch.no_grad():
                logits = self.forward(texts)
                return logits.argmax(dim=-1).tolist()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                logits = self.forward(texts[i:i + batch_size])
                all_preds.extend(logits.argmax(dim=-1).tolist())
        return all_preds

    def train(self, texts: list[str], labels: list[int] | list[list[float]],
              val_texts: list[str] | None = None, val_labels: list[int] | list[list[float]] | None = None,
              batch_size: int = 8, grad_accum_steps: int = 8, epochs: int = 3, lr: float = 2e-5,
              early_stopping: bool = False, max_epochs: int | None = None,
              patience: int = 2, min_epochs: int = 1, loss_fn: str = "ce"):
        """If `early_stopping=False` (default), train for fixed `epochs`.
        If `early_stopping=True`, train up to `max_epochs` (defaulting to `epochs`),
        track val_loss, stop when it hasn't improved for `patience` consecutive
        epochs, and restore the best-val-loss checkpoint at the end. Requires
        `val_texts` + `val_labels`.
        `loss_fn="mse"` regresses the logits onto per-class float targets
        (labels must be list[list[float]]; NaN entries are masked out of the
        loss). `loss_fn="ce"` keeps the hard/soft cross-entropy behavior."""
        if self.model is None:
            raise ValueError("Model not loaded")
        if early_stopping and not (val_texts and val_labels):
            raise ValueError("early_stopping=True requires val_texts and val_labels")
        if loss_fn not in ("ce", "mse"):
            raise ValueError(f"Unknown loss_fn: {loss_fn}")
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        is_soft = isinstance(labels[0], list)
        use_mse = loss_fn == "mse"
        if use_mse and not is_soft:
            raise ValueError("loss_fn='mse' requires per-class float targets (list[list[float]])")

        self._trunc_total = self._trunc_count = self._trunc_excess = 0

        n_epochs = (max_epochs if max_epochs is not None else epochs) if early_stopping else epochs
        best_val_loss = float("inf")
        best_state: dict | None = None
        best_epoch = -1
        epochs_since_improvement = 0
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            indices = torch.randperm(len(texts)).tolist()

            optimizer.zero_grad()
            for step, i in enumerate(tqdm(range(0, len(texts), batch_size), desc=f"Epoch {epoch}")):
                batch_indices = indices[i:i + batch_size]
                batch_texts = [texts[j] for j in batch_indices]

                inputs = self._tokenize(batch_texts, track_truncation=(epoch == 0))
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                if is_soft:
                    batch_labels = torch.tensor([labels[j] for j in batch_indices], dtype=torch.float, device=self.model.device)
                    logits = self.model(**inputs).logits
                    if use_mse:
                        mask = ~torch.isnan(batch_labels)
                        loss = F.mse_loss(logits[mask], batch_labels[mask]) / grad_accum_steps
                    else:
                        loss = F.cross_entropy(logits, batch_labels) / grad_accum_steps
                else:
                    batch_labels = torch.tensor([labels[j] for j in batch_indices], device=self.model.device)
                    inputs["labels"] = batch_labels
                    loss = self.model(**inputs).loss / grad_accum_steps

                loss.backward()
                step_loss = loss.item() * grad_accum_steps
                total_loss += step_loss

                if step % 10 == 0:
                    grad_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
                    print(f"    step {step}: loss={step_loss:.4f}, grad_norm={grad_norm:.4f}")

                if (step + 1) % grad_accum_steps == 0 or i + batch_size >= len(texts):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            avg_train_loss = total_loss / (len(texts) / batch_size)
            if epoch == 0:
                self._report_truncation()

            if val_texts and val_labels:
                # Validation loss
                self.model.eval()
                with torch.no_grad():
                    val_inputs = self._tokenize(val_texts)
                    val_inputs = {k: v.to(self.model.device) for k, v in val_inputs.items()}
                    if is_soft:
                        val_labels_tensor = torch.tensor(val_labels, dtype=torch.float, device=self.model.device)
                        val_logits = self.model(**val_inputs).logits
                        if use_mse:
                            val_mask = ~torch.isnan(val_labels_tensor)
                            val_loss = F.mse_loss(val_logits[val_mask], val_labels_tensor[val_mask]).item()
                        else:
                            val_loss = F.cross_entropy(val_logits, val_labels_tensor).item()
                        val_hard_labels = torch.nan_to_num(
                            torch.tensor(val_labels), nan=float("-inf")
                        ).argmax(dim=-1).tolist()
                    else:
                        val_inputs["labels"] = torch.tensor(val_labels, device=self.model.device)
                        val_loss = self.model(**val_inputs).loss.item()
                        val_hard_labels = val_labels

                # Validation accuracy
                val_preds = self.predict(val_texts)
                val_acc = sum(p == l for p, l in zip(val_preds, val_hard_labels)) / len(val_hard_labels)
                print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

                if early_stopping:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        epochs_since_improvement = 0
                        # CPU snapshot to avoid GPU memory bloat
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        print(f"    [early_stopping] new best val_loss={val_loss:.4f}")
                    else:
                        epochs_since_improvement += 1
                        print(f"    [early_stopping] no improvement ({epochs_since_improvement}/{patience}); best val_loss={best_val_loss:.4f} @ epoch {best_epoch}")
                        if epochs_since_improvement >= patience and (epoch + 1) >= min_epochs:
                            print(f"    [early_stopping] stopping at epoch {epoch}; restoring best from epoch {best_epoch}")
                            break
            else:
                print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}")

        if early_stopping and best_state is not None:
            self.model.load_state_dict({k: v.to(self.model.device) for k, v in best_state.items()})
            print(f"  [early_stopping] restored best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f})")

    def save_to_file(self, path: str):
        if self.model is None:
            raise ValueError("Model not loaded")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


    def load_from_file(self, path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.max_length = self._max_length_override or self.model.config.max_position_embeddings
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def load_untrained(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        config = AutoConfig.from_pretrained(self.base_model)
        config.classifier_pooling = self.pooling
        config.num_labels = self.num_classes
        self.max_length = self._max_length_override or config.max_position_embeddings
        self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model, config=config, torch_dtype=self.dtype)
        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def unload(self) -> None:
        if self.model is None:
            return

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"[TextClassificationModel] GPU {i}: {free/1024**3:.1f}/{total/1024**3:.1f} GiB free after unload")

        print("[TextClassificationModel] Model unloaded")
