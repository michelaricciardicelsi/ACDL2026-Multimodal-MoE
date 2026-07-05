"""
datasets/preprocess_audio.py
Audio preprocessing pipeline using HuggingFace datasets and Whisper's feature extractor.

Dataset:  openslr/librispeech_asr (clean split) — small streaming subset for Colab feasibility.
Encoder:  openai/whisper-small  (WhisperFeatureExtractor handles resampling + mel-spectrogram).

Also provides a multimodal_collate_fn that batches text, image, and audio samples together.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import WhisperFeatureExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_MODEL_NAME = "openai/whisper-small"
SAMPLE_RATE        = 16_000      # Whisper requires 16 kHz input
MAX_AUDIO_SECONDS  = 30          # Whisper's fixed 30-second context window
DEFAULT_SUBSET     = 1_000       # keep small for Colab feasibility


# ---------------------------------------------------------------------------
# Feature extractor loader
# ---------------------------------------------------------------------------

def load_whisper_feature_extractor(model_name: str = WHISPER_MODEL_NAME) -> WhisperFeatureExtractor:
    return WhisperFeatureExtractor.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_audio_dataset(
    split: str = "train.clean.100",
    num_examples: int = DEFAULT_SUBSET,
    streaming: bool = True,
):
    """Load LibriSpeech (clean-100) via HuggingFace datasets.

    Args:
        split:        LibriSpeech split name (e.g. "train.clean.100", "validation.clean").
        num_examples: Number of examples to use.
        streaming:    Use streaming mode to avoid the full ~30 GB download.

    Returns:
        HuggingFace Dataset (or IterableDataset) with audio resampled to 16 kHz.
    """
    dataset = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    if streaming:
        dataset = dataset.take(num_examples)
    else:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    # Automatic resampling to SAMPLE_RATE via HuggingFace Audio cast
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return dataset


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_audio(
    dataset,
    feature_extractor: WhisperFeatureExtractor,
    audio_column: str = "audio",
    max_duration_sec: float = MAX_AUDIO_SECONDS,
):
    """Extract log-mel spectrograms from audio at 16 kHz with padding/truncation.

    Args:
        dataset:           HuggingFace Dataset with an 'audio' column.
        feature_extractor: WhisperFeatureExtractor.
        audio_column:      Name of the audio column.
        max_duration_sec:  Maximum audio duration in seconds (default 30 s).

    Returns:
        Dataset with columns: input_features, text.
    """
    max_samples = int(max_duration_sec * SAMPLE_RATE)

    def process_fn(examples):
        audios  = examples[audio_column]
        arrays  = []

        for audio in audios:
            arr = audio["array"].astype(np.float32)
            # Truncate to max window
            if len(arr) > max_samples:
                arr = arr[:max_samples]
            arrays.append(arr)

        features = feature_extractor(
            arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=max_samples,
            truncation=True,
            return_attention_mask=False,
        )
        return {"input_features": features["input_features"].tolist()}

    # Remove all columns except text (transcription)
    all_cols = list(dataset.column_names) if hasattr(dataset, "column_names") else []
    cols_to_remove = [c for c in all_cols if c not in ("text",)]

    try:
        return dataset.map(process_fn, batched=True, remove_columns=cols_to_remove)
    except (TypeError, AttributeError):
        # Streaming datasets may not support remove_columns kwarg in older versions
        return dataset.map(process_fn, batched=True)


# ---------------------------------------------------------------------------
# Multimodal collate function
# ---------------------------------------------------------------------------

def multimodal_collate_fn(batch: list) -> dict:
    """Batch heterogeneous multimodal samples into tensors.

    Each element of `batch` is a dict that may contain any subset of:
        input_ids, attention_mask, labels  — text (LongTensor)
        pixel_values                       — vision (FloatTensor)
        input_features                     — audio  (FloatTensor)

    Returns a dict of stacked tensors, one per key present in the batch.
    """
    FLOAT_KEYS = {"pixel_values", "input_features"}
    out = {}

    for key in batch[0].keys():
        vals = [sample[key] for sample in batch]
        try:
            arr    = np.array(vals)
            dtype  = torch.float32 if key in FLOAT_KEYS else torch.long
            out[key] = torch.tensor(arr, dtype=dtype)
        except ValueError:
            # Ragged arrays (e.g. variable-length audio) — keep as list
            out[key] = vals

    return out


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading Whisper feature extractor ...")
    fe = load_whisper_feature_extractor()

    print("Loading 3 LibriSpeech examples (streaming) ...")
    ds = load_audio_dataset(num_examples=3, streaming=True)

    processed = preprocess_audio(ds, fe)
    for i, ex in enumerate(processed):
        arr = np.array(ex["input_features"])
        print(f"  [{i}] input_features shape: {arr.shape}")
        # Expected: (80, 3000) for 30-second whisper window

    print("Done.")
