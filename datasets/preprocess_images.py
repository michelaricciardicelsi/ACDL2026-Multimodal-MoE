"""
datasets/preprocess_images.py
Image preprocessing pipeline using HuggingFace datasets and CLIP's processor.

Dataset:  HuggingFaceM4/COCO — validation 2017 subset (image + caption pairs).
Encoder:  openai/clip-vit-base-patch32  (CLIPProcessor handles resize & normalisation).
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"
IMAGE_SIZE        = 224       # ViT-B/32 expects 224×224
DEFAULT_SUBSET    = 2_000     # keep small for Colab feasibility
MAX_CAPTION_LEN   = 77        # CLIP's default max token length


# ---------------------------------------------------------------------------
# Processor loader
# ---------------------------------------------------------------------------

def load_clip_processor(model_name: str = CLIP_MODEL_NAME) -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_image_dataset(
    split: str = "validation",
    num_examples: int = DEFAULT_SUBSET,
    streaming: bool = False,
):
    """Load COCO captions via HuggingFace datasets (image + caption pairs).

    Args:
        split:        "train" or "validation".
        num_examples: Number of examples to use.
        streaming:    Use streaming mode to avoid full download.

    Returns:
        HuggingFace Dataset.
    """
    dataset = load_dataset(
        "HuggingFaceM4/COCO",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    if streaming:
        dataset = dataset.take(num_examples)
    else:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    return dataset


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _extract_first_caption(cap) -> str:
    """Flatten nested COCO caption structures to a plain string."""
    if isinstance(cap, list):
        first = cap[0]
        return first if isinstance(first, str) else str(first)
    return str(cap)


def preprocess_images(
    dataset,
    processor: CLIPProcessor,
    image_column: str = "image",
    caption_column: str = "sentences_raw",
    max_caption_length: int = MAX_CAPTION_LEN,
):
    """Resize, normalise, and tokenise image-caption pairs for CLIP.

    Args:
        dataset:            HuggingFace Dataset with image and caption columns.
        processor:          CLIPProcessor instance.
        image_column:       Name of the PIL Image column.
        caption_column:     Name of the caption column.
        max_caption_length: Max tokens for text captions.

    Returns:
        Dataset with columns: pixel_values, input_ids, attention_mask.
    """

    def process_fn(examples):
        images   = examples[image_column]
        captions = examples[caption_column]

        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                pil_images.append(Image.fromarray(img).convert("RGB"))

        flat_captions = [_extract_first_caption(c) for c in captions]

        processed = processor(
            text=flat_captions,
            images=pil_images,
            return_tensors="pt",
            padding="max_length",
            max_length=max_caption_length,
            truncation=True,
        )

        return {
            "pixel_values":   processed["pixel_values"].numpy().tolist(),
            "input_ids":      processed["input_ids"].numpy().tolist(),
            "attention_mask": processed["attention_mask"].numpy().tolist(),
        }

    cols_to_remove = [c for c in dataset.column_names if c not in ("pixel_values", "input_ids", "attention_mask")]
    return dataset.map(process_fn, batched=True, remove_columns=cols_to_remove)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading CLIP processor ...")
    proc = load_clip_processor()

    print("Loading 3 COCO examples ...")
    ds = load_image_dataset(num_examples=3, streaming=False)
    print(f"  Columns: {ds.column_names}")

    processed = preprocess_images(ds, proc)
    sample     = processed[0]
    pv         = torch.tensor(sample["pixel_values"])
    print(f"  pixel_values shape: {pv.shape}")   # expected: (3, 224, 224)
    print("Done.")
