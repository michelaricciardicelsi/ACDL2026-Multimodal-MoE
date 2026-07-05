"""
datasets/preprocess_text.py
Text preprocessing pipeline using HuggingFace datasets and the Phi tokenizer.

Dataset:  C4 (allenai/c4, "en" subset) — streamed in small slices for Colab feasibility.
Tokenizer: microsoft/Phi-mini-MoE-instruct
"""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI_TOKENIZER_NAME  = "microsoft/Phi-mini-MoE-instruct"
DEFAULT_MAX_LENGTH  = 512
DEFAULT_SUBSET_SIZE = 10_000


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_phi_tokenizer(model_name: str = PHI_TOKENIZER_NAME) -> AutoTokenizer:
    """Load and configure the Phi tokenizer.

    Sets pad_token to eos_token when no pad token is defined (common for causal LMs).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# C4 dataset loader
# ---------------------------------------------------------------------------

def load_text_dataset(
    split: str = "train",
    num_examples: int = DEFAULT_SUBSET_SIZE,
    streaming: bool = True,
):
    """Load a small subset of C4 (en) via HuggingFace datasets.

    Args:
        split:        "train" or "validation".
        num_examples: Maximum number of examples to load.
        streaming:    Use streaming mode to avoid downloading the full dataset.

    Returns:
        HuggingFace Dataset or IterableDataset.
    """
    dataset = load_dataset(
        "allenai/c4",
        "en",
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
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_dataset(
    dataset,
    tokenizer: AutoTokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
    text_column: str = "text",
    remove_columns: Optional[list] = None,
):
    """Tokenize a text dataset with padding and truncation.

    For causal language modeling, labels == input_ids (shift is handled inside the model).

    Args:
        dataset:        HuggingFace Dataset (streaming or in-memory).
        tokenizer:      Phi tokenizer.
        max_length:     Maximum token sequence length.
        text_column:    Column containing raw text strings.
        remove_columns: Columns to drop after tokenization; defaults to [text_column].

    Returns:
        Dataset with columns: input_ids, attention_mask, labels.
    """

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        # labels = input_ids for causal LM (shift handled in model forward)
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    cols_to_remove = remove_columns if remove_columns is not None else [text_column]

    try:
        return dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=cols_to_remove,
        )
    except TypeError:
        # Streaming datasets may not support remove_columns in older versions
        return dataset.map(tokenize_fn, batched=True)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading Phi tokenizer ...")
    tok = load_phi_tokenizer()
    print(f"  Vocab size: {tok.vocab_size}")
    print(f"  Pad token:  '{tok.pad_token}'")

    print("\nLoading 5 C4 examples (streaming) ...")
    ds = load_text_dataset(num_examples=5, streaming=True)
    for i, ex in enumerate(ds):
        enc = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
        print(f"  [{i}] input_ids[:8] = {enc['input_ids'][:8]}")

    print("\nDone.")
