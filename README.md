# Multimodal Sparse Mixture of Experts — ACDL 2026 Project #15

A small multimodal language model based on a **Sparse Mixture of Experts (MoE)** architecture that processes and reasons over **Text**, **Speech (audio)**, and **Images**.

Built on top of [`microsoft/Phi-mini-MoE-instruct`](https://huggingface.co/microsoft/Phi-mini-MoE-instruct) extended with modality encoders and a fusion mechanism.

---

## Architecture

```
Text   → Phi embedding table     ─┐
Image  → CLIP ViT-B/32 + FFN    ─┼─→  MultimodalFusionModule  →  Phi backbone + SparseMoELayer  →  logits
Audio  → Whisper-small + Linear ─┘           (concat)                (Top-2 gating, SwiGLU experts)
```

| Component              | Model / Method                         |
|------------------------|----------------------------------------|
| Language backbone      | `microsoft/Phi-mini-MoE-instruct`      |
| Vision encoder         | `openai/clip-vit-base-patch32`         |
| Speech encoder         | `openai/whisper-small`                 |
| MoE layer              | Top-2 gating, SwiGLU experts           |
| Fusion                 | Projection + concatenation (seq dim)   |
| Training framework     | HuggingFace `accelerate` + bfloat16    |

---

## Repository Structure

```
├── model/
│   ├── modeling_moe.py          # SparseMoELayer (Top-2 gating, SwiGLU, aux loss)
│   └── multimodal_model.py      # AudioEncoderWrapper, VisionEncoderWrapper,
│                                #   MultimodalFusionModule, MultimodalMoEModel
├── datasets/
│   ├── preprocess_text.py       # C4 → Phi tokenizer
│   ├── preprocess_images.py     # COCO Captions → CLIP processor
│   └── preprocess_audio.py      # LibriSpeech → Whisper feature extractor
├── training/
│   ├── train.py                 # Accelerate training loop
│   └── config.yaml              # All hyperparameters and paths
├── evaluation/
│   ├── evaluate.py              # Text perplexity/accuracy, image-text & speech-text retrieval
│   └── metrics.py               # matplotlib/seaborn MoE routing plots
├── notebooks/
│   └── demo.ipynb               # Colab-ready end-to-end demo
├── README.md
├── requirements.txt
└── run.sh                       # One-command pipeline: install → train → evaluate
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Google Colab (recommended):** Use a T4, L4, or A100 runtime.  
> **Local GPU:** NVIDIA with ≥ 8 GB VRAM or Apple Silicon with ≥ 16 GB unified memory.

---

## Training

```bash
# Single GPU / CPU
python training/train.py --config training/config.yaml

# Multi-GPU via accelerate
accelerate launch training/train.py --config training/config.yaml
```

Key hyperparameters (edit `training/config.yaml`):

| Parameter                  | Default   |
|---------------------------|-----------|
| `num_moe_experts`          | 8         |
| `top_k`                    | 2         |
| `capacity_factor`          | 1.25      |
| `aux_loss_weight`          | 0.01      |
| `batch_size`               | 4         |
| `gradient_accumulation_steps` | 8     |
| `learning_rate`            | 2e-5      |
| `mixed_precision`          | bf16      |
| `num_epochs`               | 3         |

---

## Evaluation

```bash
python evaluation/evaluate.py \
    --config training/config.yaml \
    --checkpoint checkpoints/epoch_3
```

Outputs:
- **Text:** validation loss, perplexity, token-level accuracy
- **Image-text:** retrieval recall@1 and recall@5
- **Speech-text:** mean cosine similarity
- **MoE routing:** per-layer gate entropy, inactive expert %

### Generate plots

```bash
python evaluation/metrics.py          # demo with synthetic data
```

Plots saved to `output/demo_plots/`:
- `expert_load_layer_N.png` — bar chart per MoE layer
- `gate_entropy_histogram.png`
- `routing_frequency_heatmap.png`
- `token_allocation_layer0.png`

---

## Full Pipeline (one command)

```bash
bash run.sh
```

This installs requirements, runs a smoke test, trains, evaluates, and generates plots.

---

## Demo Notebook

Open [`notebooks/demo.ipynb`](notebooks/demo.ipynb) in Google Colab for an interactive walkthrough:
- SparseMoELayer smoke test (CPU, no downloads)
- Full model load
- Multimodal inference on a synthetic text + image + audio sample
- Live routing statistics and plots

---

## Datasets

| Modality | Dataset                 | Split              |
|----------|------------------------|-------------------|
| Text     | `allenai/c4` (en)       | train / validation |
| Images   | `HuggingFaceM4/COCO`   | validation 2017    |
| Speech   | `openslr/librispeech_asr` clean | train.clean.100 |

Only small streaming subsets are used (5 000 text, 500 image, 500 audio examples by default).

---

## MoE Implementation Details

- **Experts:** `SwiGLUExpert` — each expert is a two-gate FFN: `out = W_down(SiLU(W_gate·x) * W_up·x)`
- **Routing:** linear gate → softmax → Top-2 selection, renormalized weights
- **Capacity factor:** tokens overflow the expert buffer are dropped (no gradient)
- **Auxiliary loss:** Switch Transformer load-balancing loss: `L_aux = E * Σ_i f_i * P_i`
- **Tracking:** expert utilization, gate entropy, routing frequency, inactive expert %

---

## Reproducibility

```bash
pip install -r requirements.txt
bash run.sh
```

All random seeds are set via `set_seed(42)` at the start of training and evaluation.

---

## Technical Report Outline

1. Introduction  
2. Background: Sparse Mixture of Experts  
3. Background: Multimodal Large Language Models  
4. Dataset description  
5. Architecture  
6. Implementation details  
7. Training methodology  
8. Experimental evaluation  
9. Expert routing analysis  
10. Discussion  
11. Conclusions  

---

## License

For academic use only (ACDL 2026).
