"""
evaluation/evaluate.py
Evaluation script for the Multimodal Sparse MoE model.

Computes:
  - Text:          cross-entropy loss, perplexity, token-level accuracy
  - Image-text:    retrieval recall@1 and recall@5 (cosine similarity proxy)
  - Speech-text:   mean cosine similarity between text-only and text+audio representations
  - MoE routing:   per-layer gate entropy and inactive expert percentage

Usage:
    python evaluation/evaluate.py \\
        --config training/config.yaml \\
        --checkpoint checkpoints/epoch_1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.multimodal_model import MultimodalMoEModel
from datasets.preprocess_text import load_phi_tokenizer, load_text_dataset, tokenize_dataset
from training.train import TextDataset, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Text evaluation
# ---------------------------------------------------------------------------

def evaluate_text(model, val_loader, device: torch.device) -> dict:
    """Compute cross-entropy loss, perplexity, and token-level accuracy."""
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs["loss"].item()

            # Token-level accuracy (on text portion only)
            T_text       = batch["input_ids"].shape[1]
            text_logits  = outputs["logits"][:, :T_text, :]
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()

            valid    = shift_labels != -100
            preds    = shift_logits.argmax(dim=-1)
            correct  = ((preds == shift_labels) & valid).sum().item()
            tokens   = valid.sum().item()

            total_correct += correct
            total_tokens  += tokens

    n          = max(len(val_loader), 1)
    avg_loss   = total_loss / n
    perplexity = math.exp(min(avg_loss, 100.0))
    accuracy   = 100.0 * total_correct / max(total_tokens, 1)

    return {
        "val_loss":   avg_loss,
        "perplexity": perplexity,
        "accuracy":   accuracy,
    }


# ---------------------------------------------------------------------------
# Image-text retrieval
# ---------------------------------------------------------------------------

def evaluate_image_text_retrieval(
    model,
    val_loader,
    device: torch.device,
    top_k: tuple = (1, 5),
) -> dict:
    """Image-text retrieval accuracy via cosine similarity.

    Computes the recall@k by comparing:
      - text-only  representation (mean logit vector)
      - text+image representation (mean logit vector after multimodal fusion)

    A high recall@k indicates that the image embedding pulls the representation
    toward the correct text item in the batch.
    """
    model.eval()
    text_reps  = []
    fused_reps = []

    with torch.no_grad():
        for batch in val_loader:
            if "pixel_values" not in batch:
                break
            input_ids    = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            text_out  = model(input_ids=input_ids)
            fused_out = model(input_ids=input_ids, pixel_values=pixel_values)

            t_rep = F.normalize(text_out["logits"].mean(dim=1).float(),  dim=-1)
            f_rep = F.normalize(fused_out["logits"].mean(dim=1).float(), dim=-1)
            text_reps.append(t_rep)
            fused_reps.append(f_rep)

    if not text_reps:
        return {"image_text_retrieval": "no image data available"}

    text_e  = torch.cat(text_reps,  dim=0)  # (N, V)
    fused_e = torch.cat(fused_reps, dim=0)

    sim  = text_e @ fused_e.T              # (N, N)
    gt   = torch.arange(len(text_e), device=device)

    results = {}
    for k in top_k:
        k_clipped = min(k, len(text_e))
        topk_idx  = sim.topk(k_clipped, dim=-1).indices
        correct   = (topk_idx == gt.unsqueeze(1)).any(dim=1).float()
        results[f"image_text_recall@{k}"] = round(correct.mean().item() * 100.0, 2)

    return results


# ---------------------------------------------------------------------------
# Speech-text similarity
# ---------------------------------------------------------------------------

def evaluate_speech_text_similarity(
    model,
    val_loader,
    device: torch.device,
) -> dict:
    """Mean cosine similarity between text-only and text+audio representations."""
    model.eval()
    similarities = []

    with torch.no_grad():
        for batch in val_loader:
            if "input_features" not in batch:
                break
            input_ids      = batch["input_ids"].to(device)
            input_features = batch["input_features"].to(device)

            text_out  = model(input_ids=input_ids)
            fused_out = model(input_ids=input_ids, input_features=input_features)

            t_rep = F.normalize(text_out["logits"].mean(dim=1).float(),  dim=-1)
            f_rep = F.normalize(fused_out["logits"].mean(dim=1).float(), dim=-1)

            cos_sim = (t_rep * f_rep).sum(dim=-1).cpu().tolist()
            similarities.extend(cos_sim)

    if not similarities:
        return {"speech_text_similarity": "no audio data available"}

    return {
        "speech_text_mean_cosine": round(float(np.mean(similarities)), 4),
        "speech_text_std_cosine":  round(float(np.std(similarities)),  4),
    }


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(config_path: str, checkpoint_path: str | None = None):
    cfg    = load_config(config_path)
    device = get_device()
    set_seed(cfg["training"]["seed"])

    mp = cfg["training"].get("mixed_precision", "bf16")
    torch_dtype = torch.bfloat16 if mp == "bf16" else torch.float16

    # ---- Build model ----
    print("[evaluate] Building model ...")
    try:
        from transformers import AutoConfig
        phi_cfg        = AutoConfig.from_pretrained(cfg["model"]["phi_model_name"], trust_remote_code=True)
        n_total        = phi_cfg.num_hidden_layers
        step           = cfg["model"].get("replace_every_n_layers", 2)
        replace_layers = list(range(0, n_total, step))
    except Exception:
        replace_layers = None

    model = MultimodalMoEModel(
        phi_model_name=cfg["model"]["phi_model_name"],
        num_moe_experts=cfg["model"]["num_moe_experts"],
        top_k=cfg["model"]["top_k"],
        capacity_factor=cfg["model"]["capacity_factor"],
        aux_loss_weight=cfg["model"]["aux_loss_weight"],
        replace_layers=replace_layers,
        freeze_encoders=cfg["model"]["freeze_encoders"],
        torch_dtype=torch_dtype,
    )

    # ---- Load checkpoint ----
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[evaluate] Loading checkpoint: {checkpoint_path}")
        if Path(checkpoint_path).is_dir():
            from accelerate import Accelerator
            acc = Accelerator()
            acc.load_state(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
    else:
        print("[evaluate] No checkpoint provided — using untrained weights.")

    model = model.to(device)
    model.eval()

    # ---- Load validation data ----
    print("[evaluate] Loading validation data ...")
    tokenizer   = load_phi_tokenizer(cfg["model"]["phi_model_name"])
    raw_val     = load_text_dataset(split="validation", num_examples=500, streaming=True)
    tok_val     = tokenize_dataset(raw_val, tokenizer, max_length=cfg["data"]["max_text_length"])
    val_dataset = TextDataset(tok_val, max_items=500)
    val_loader  = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # ---- Text evaluation ----
    print("[evaluate] Text metrics ...")
    text_res = evaluate_text(model, val_loader, device)
    print(f"  val_loss   = {text_res['val_loss']:.4f}")
    print(f"  perplexity = {text_res['perplexity']:.2f}")
    print(f"  accuracy   = {text_res['accuracy']:.2f}%")

    # ---- MoE routing stats ----
    print("[evaluate] MoE routing statistics ...")
    routing = model.get_routing_stats()
    for layer_name, stats in routing.items():
        print(
            f"  {layer_name}: entropy={stats['gate_entropy']:.4f}  "
            f"inactive={stats['pct_inactive_experts']:.1f}%"
        )

    results = {
        **text_res,
        "routing_stats": routing,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Sparse MoE model")
    parser.add_argument("--config",     type=str, default="training/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint dir or file (optional)")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
