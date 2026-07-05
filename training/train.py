"""
training/train.py
Accelerate-based training loop for the Multimodal Sparse MoE model.

Features:
  - Reads all hyperparams from config.yaml
  - Sets global random seeds for reproducibility
  - Mixed-precision training (bfloat16 / fp16)
  - Gradient accumulation + cosine LR schedule with warmup
  - Combined loss: task_loss + lambda * aux_moe_loss
  - Per-epoch evaluation (perplexity) and checkpoint saving
  - Logging via TensorBoard (via accelerate)

Usage:
    # Single GPU / CPU:
    python training/train.py --config training/config.yaml

    # Multi-GPU (accelerate):
    accelerate launch training/train.py --config training/config.yaml
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

try:
    from accelerate import Accelerator
except ImportError:
    raise ImportError("accelerate is required: pip install accelerate")

# Ensure project root is on sys.path when launched from any working directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.multimodal_model import MultimodalMoEModel
from datasets.preprocess_text import load_phi_tokenizer, load_text_dataset, tokenize_dataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapping a tokenized HuggingFace dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Materialises a (streaming) HuggingFace tokenized dataset into a list."""

    def __init__(self, hf_dataset, max_items: int | None = None):
        self.data = []
        for i, item in enumerate(hf_dataset):
            if max_items is not None and i >= max_items:
                break
            self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(item["labels"],         dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str):
    cfg = load_config(config_path)
    t   = cfg["training"]
    m   = cfg["model"]
    d   = cfg["data"]
    p   = cfg["paths"]

    # ---- Accelerator (handles device placement, mixed-precision, logging) ----
    accelerator = Accelerator(
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        mixed_precision=t["mixed_precision"],
        log_with="tensorboard",
        project_dir=p["log_dir"],
    )

    set_seed(t["seed"])

    # ---- Directories ----
    ckpt_dir = Path(p["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer & datasets ----
    accelerator.print("[train] Loading tokenizer ...")
    tokenizer = load_phi_tokenizer(m["phi_model_name"])

    accelerator.print("[train] Loading training data ...")
    raw_train      = load_text_dataset(split="train",      num_examples=d["text_subset"],                         streaming=True)
    raw_val        = load_text_dataset(split="validation", num_examples=max(100, d["text_subset"] // 10),         streaming=True)
    tok_train      = tokenize_dataset(raw_train, tokenizer, max_length=d["max_text_length"])
    tok_val        = tokenize_dataset(raw_val,   tokenizer, max_length=d["max_text_length"])
    train_dataset  = TextDataset(tok_train, max_items=d["text_subset"])
    val_dataset    = TextDataset(tok_val,   max_items=max(100, d["text_subset"] // 10))

    train_loader = DataLoader(
        train_dataset,
        batch_size=t["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=t["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    # ---- Model ----
    accelerator.print("[train] Building MultimodalMoEModel ...")
    torch_dtype = torch.bfloat16 if t["mixed_precision"] == "bf16" else torch.float16

    n_layers = None  # resolved inside MultimodalMoEModel using replace_every_n_layers
    try:
        from transformers import AutoConfig
        phi_cfg    = AutoConfig.from_pretrained(m["phi_model_name"], trust_remote_code=True)
        n_total    = phi_cfg.num_hidden_layers
        step       = m.get("replace_every_n_layers", 2)
        replace_layers = list(range(0, n_total, step))
    except Exception:
        replace_layers = None  # fallback: MultimodalMoEModel uses default (every other layer)

    model = MultimodalMoEModel(
        phi_model_name=m["phi_model_name"],
        num_moe_experts=m["num_moe_experts"],
        top_k=m["top_k"],
        capacity_factor=m["capacity_factor"],
        aux_loss_weight=m["aux_loss_weight"],
        replace_layers=replace_layers,
        freeze_encoders=m["freeze_encoders"],
        torch_dtype=torch_dtype,
    )

    # ---- Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(
        f"[train] Trainable params: {sum(p.numel() for p in trainable_params):,}"
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )

    # ---- LR Scheduler ----
    steps_per_epoch       = len(train_loader) // t["gradient_accumulation_steps"]
    num_training_steps    = steps_per_epoch * t["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=t["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    # ---- Prepare with Accelerate ----
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ---- Training ----
    accelerator.print(f"[train] Starting {t['num_epochs']} epoch(s) ...")
    global_step = 0

    for epoch in range(t["num_epochs"]):
        model.train()
        epoch_task_loss = 0.0
        epoch_aux_loss  = 0.0
        num_batches     = 0

        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs    = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss = outputs["total_loss"]
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), t["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_task_loss += outputs["loss"].item()
            epoch_aux_loss  += outputs["aux_loss"].item()
            num_batches     += 1
            global_step     += 1

            if global_step % t["log_every_steps"] == 0 and accelerator.is_main_process:
                avg_t  = epoch_task_loss / num_batches
                avg_a  = epoch_aux_loss  / num_batches
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else t["learning_rate"]
                print(
                    f"  Epoch {epoch+1:2d} | Step {global_step:5d} | "
                    f"task={avg_t:.4f} | aux={avg_a:.6f} | lr={lr_now:.2e}"
                )

        # ---- Epoch-end evaluation ----
        if (epoch + 1) % t["eval_every_epochs"] == 0:
            model.eval()
            val_loss  = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    outputs   = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    val_loss  += outputs["loss"].item()
                    val_steps += 1

            avg_val    = val_loss / max(val_steps, 1)
            perplexity = math.exp(min(avg_val, 100.0))

            if accelerator.is_main_process:
                print(
                    f"  [Eval]  Epoch {epoch+1} | "
                    f"val_loss={avg_val:.4f} | perplexity={perplexity:.2f}"
                )

        # ---- Checkpoint ----
        if (epoch + 1) % t["save_every_epochs"] == 0 and accelerator.is_main_process:
            save_path = ckpt_dir / f"epoch_{epoch+1}"
            accelerator.save_state(str(save_path))
            print(f"  [Checkpoint] Saved → {save_path}")

    accelerator.print("[train] Training complete.")
    accelerator.end_training()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal Sparse MoE model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to YAML config file (default: training/config.yaml)",
    )
    args = parser.parse_args()
    train(args.config)
