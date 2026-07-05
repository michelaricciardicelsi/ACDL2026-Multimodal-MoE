# COPILOT_Multimodal_MoE_INSTRUCTIONS.md

Welcome to the comprehensive GitHub Copilot guide for building your Multimodal Sparse Mixture of Experts (MoE) model. As an AI coding instructor, I've structured this document to help you systematically tackle the ACDL 2026 Project #15. We will leverage Copilot's strengths in boilerplate generation, API translation, and debugging to build this complex architecture efficiently. 

---

## 1. Project Overview & Setup

### Goals & Timeline
[cite_start]The objective of this project is to design, implement, train, and evaluate a small multimodal language model based on a Sparse Mixture of Experts (MoE) architecture[cite: 4]. [cite_start]The model must process and reason over Text, Speech (audio), and Images[cite: 4]. [cite_start]You will start from the open-source Phi-mini-MoE-instruct model and extend it with modality encoders and a fusion mechanism[cite: 5]. [cite_start]This project is intended to be feasible within 4 weeks, making extensive use of Large Language Models as programming assistants[cite: 7].

[cite_start]**4-steps Schedule [cite: 202-220]:**
* [cite_start]**Step 1:** environment setup [cite: 203][cite_start]; study Phi-mini-MoE [cite: 204][cite_start]; dataset download [cite: 205][cite_start]; preprocessing implementation[cite: 206].
* [cite_start]**Step 2:** implement image encoder [cite: 208][cite_start]; implement speech encoder [cite: 209][cite_start]; implement multimodal fusion[cite: 210].
* [cite_start]**Step 3:** implement Sparse MoE routing [cite: 212][cite_start]; perform training [cite: 214][cite_start]; debug the pipeline[cite: 215].
* [cite_start]**Step 4:** evaluation [cite: 216][cite_start]; generate plots [cite: 217][cite_start]; prepare GitHub repository [cite: 218][cite_start]; write technical report [cite: 219][cite_start]; verify reproducibility[cite: 220].

### Recommended Compute
[cite_start]The project is designed to be completed without access to expensive computing infrastructure[cite: 182]. 
* [cite_start]**Option 1 (recommended):** Google Colab Free or Pro using a T4, L4, or A100 GPU when available[cite: 184, 185, 186, 187].
* [cite_start]**Option 2:** A personal laptop equipped with an NVIDIA GPU with at least 8 GB VRAM or Apple Silicon (M1/M2/M3) with at least 16 GB unified memory[cite: 188, 189, 190, 192].
* [cite_start]**Option 3:** Free academic computing resources such as Kaggle Notebooks, Lightning AI Studio (free tier), university HPC clusters, EuroHPC training allocations, or institutional GPU servers[cite: 193, 194, 195, 196, 197, 198, 199].

### Repository Structure
[cite_start]The final GitHub repository must contain the following structure[cite: 128]:
```text
├── model/ [cite: 129]
│   ├── modeling_moe.py [cite: 130]
│   └── multimodal_model.py [cite: 131]
├── datasets/ [cite: 132]
│   ├── preprocess_text.py [cite: 133]
│   ├── preprocess_images.py [cite: 134]
│   └── preprocess_audio.py [cite: 135]
├── training/ [cite: 136]
│   ├── train.py [cite: 137]
│   └── config.yaml [cite: 138]
├── evaluation/ [cite: 139]
│   ├── evaluate.py [cite: 140]
│   └── metrics.py [cite: 141]
├── notebooks/ [cite: 142]
│   └── demo.ipynb [cite: 143]
├── README.md [cite: 144]
├── requirements.txt [cite: 145]
└── run.sh [cite: 146]



### Environment Setup
Update your `requirements.txt` to include essential libraries. [cite_start]Implement the training pipeline in PyTorch[cite: 100]. Suggested libraries:
* [cite_start]transformers [cite: 102]
* [cite_start]datasets [cite: 103]
* [cite_start]accelerate [cite: 104]
* [cite_start]peft [cite: 105]
* [cite_start]deepspeed (optional) [cite: 106]

---

## 2. Phase 1: Base Model Exploration & Environment

* [cite_start]Use as starting point: Phi-mini-MoE-instruct (https://huggingface.co/microsoft/Phi-mini-MoE-instruct)[cite: 9, 10]. 
* [cite_start]The student is expected to: understand the architecture; extend it to multimodal inputs; implement and test the complete pipeline[cite: 11, 12, 13, 14].

**Copilot Prompt:**
> "Copilot, write a Python script using the HuggingFace Transformers library to load the `microsoft/Phi-mini-MoE-instruct` model and its tokenizer. Print the model summary and the configuration to help me analyze its embedding dimensions and layer structure."

* [cite_start]**Text Encoder:** Use the tokenizer already provided with Phi-mini-MoE[cite: 18]. [cite_start]Accepted libraries: HuggingFace Transformers, Tokenizers[cite: 19, 20, 21].

---

## 3. Phase 2: Modality Encoders

### Speech Encoder
* [cite_start]Replace the video encoder with a speech encoder[cite: 23]. 
* [cite_start]Possible choices include: Whisper encoder, Wav2Vec2, HUBERT[cite: 24, 25, 26, 27]. 
* [cite_start]The speech encoder should transform audio into embeddings compatible with the transformer hidden dimension[cite: 28].

**Copilot Prompt:**
> "Create a PyTorch class `AudioEncoderWrapper` that loads the `openai/whisper-small` encoder. Include a linear projection layer that maps the Whisper output embeddings to the dimension `phi_hidden_size`. Ensure the audio is processed at 16 kHz."

### Vision Encoder
* [cite_start]Use one of: CLIP Vision Encoder, ViT, SigLIP[cite: 30, 31, 32, 33]. 
* [cite_start]The encoder should produce image embeddings aligned with the language model embedding space[cite: 34].

**Copilot Prompt:**
> "Write a PyTorch module `VisionEncoderWrapper` using the `openai/clip-vit-base-patch32` vision model. Extract the pooled output or patch embeddings, and add a feed-forward projection network to align these image embeddings with the language model embedding space."

---

## 4. Phase 3: Multimodal Fusion

* [cite_start]Implement one of the following: Projection layer + concatenation, Cross-attention, Adapter-based fusion[cite: 36, 37, 38, 39]. 
* [cite_start]The student must explain and justify the chosen design[cite: 40]. 

*Recommendation:* **Projection layer + concatenation** is a strong starting point for LLM assistants. 

**Copilot Prompt:**
> "Implement a multimodal fusion module in PyTorch using the projection + concatenation approach. It should take text embeddings, projected image embeddings, and projected audio embeddings as inputs, concatenate them along the sequence dimension, and return the combined tensor."

---

## 5. Phase 4: Sparse MoE Implementation

* [cite_start]Replace standard FFN layers with Sparse MoE layers[cite: 42]. 
* [cite_start]Required characteristics: Top-2 gating, SwiGLU experts, Capacity factor, Auxiliary load-balancing loss, bfloat 16 or float16 training when supported[cite: 43, 44, 45, 46, 47, 48].

**Copilot Prompt:**
> "Draft a PyTorch module `SparseMoELayer` to replace a standard transformer FFN. It must include SwiGLU experts and a Top-2 gating network using a linear layer and softmax. Include logic to compute a capacity factor and calculate an auxiliary load-balancing loss. Include tracking variables for expert utilization and inactive expert counts."

* [cite_start]The implementation should report: routing probabilities; expert utilization; entropy of gate assignments; percentage of inactive experts[cite: 49, 50, 51, 52, 53].

---

## 6. Phase 5: Data Preprocessing Pipeline

* [cite_start]To keep the project computationally feasible, only small subsets (or sampled versions) of the datasets should be used[cite: 55]. 
* [cite_start]**Text Dataset (Choose one):** The Pile, C4[cite: 56, 57, 58, 59]. 
* [cite_start]**Programming Dataset (Choose one):** CodeParrot, The Stack v2[cite: 60, 61, 62, 63]. 
* [cite_start]**Logical Reasoning Dataset (Choose one):** LogiQA, ReClor[cite: 64, 65, 66, 67]. 
* [cite_start]**Mathematical Dataset (Choose one):** GSM8K, MATH[cite: 68, 69, 70, 71]. 
* [cite_start]**Educational Reasoning Dataset (Choose one):** SAT, GRE, GMAT[cite: 72, 73, 74, 75, 76]. 
* [cite_start]**Image Dataset (Choose one):** COCO, ImageNet, LAION subset[cite: 77, 78, 79, 80, 81]. 
* [cite_start]**Speech Dataset (Choose one):** LibriSpeech, Mozilla Common Voice, Vox Populi, GigaSpeech (small subset)[cite: 82, 83, 84, 85, 86, 87]. [cite_start]The speech data should be converted into embeddings using the selected speech encoder[cite: 88].

* [cite_start]The preprocessing pipeline must include: text tokenization; image resizing and normalization; audio loading; resampling to 16 kHz; padding/truncation; batching; multimodal alignment[cite: 90, 91, 92, 93, 94, 95, 96, 97]. 
* [cite_start]The complete preprocessing code must be reproducible[cite: 98].

**Copilot Prompt:**
> "Write a dataset preprocessing pipeline using the HuggingFace `datasets` library. Include functions for: 1) Tokenizing text using Phi's tokenizer. 2) Resizing and normalizing images. 3) Loading audio files with `librosa`, resampling to 16 kHz, and applying padding/truncation. Add a custom collate function to batch these diverse modalities together."

---

## 7. Phase 6: Training Loop

* [cite_start]Implement the training pipeline in PyTorch[cite: 100]. 
* [cite_start]Training should include: checkpoint saving; evaluation every epoch; logging; reproducibility through random seeds[cite: 107, 108, 109, 110, 111]. 
* [cite_start]Training on reduced dataset subsets is acceptable and encouraged[cite: 200].

**Copilot Prompt:**
> "Generate a PyTorch training loop in `train.py` using `accelerate`. Include mixed-precision training (bfloat16), gradient accumulation, and a scheduler. Set up random seeds for reproducibility. Add logic to compute and log the main task loss combined with the MoE auxiliary load-balancing loss. Save checkpoints at the end of each epoch."

---

## 8. Phase 7: Evaluation & Analysis

* Evaluate the model on:
    * [cite_start]**Text:** perplexity, accuracy[cite: 114, 115, 116].
    * [cite_start]**Images:** image-text retrieval or caption matching accuracy[cite: 117, 118].
    * [cite_start]**Speech:** speech-text retrieval or transcription embedding similarity[cite: 119, 120].
* [cite_start]MoE Analysis: Produce plots showing: expert load distribution; gate entropy; routing frequency; token allocation per expert[cite: 121, 122, 123, 124, 125, 126].

**Copilot Prompt:**
> "Using `matplotlib` and `seaborn`, write a Python script `metrics.py` that takes a dictionary of MoE tracking statistics (expert load distribution, gate entropy, routing frequency, token allocation per expert) and generates labeled bar charts and histograms."

---

## 9. Phase 8: Repository Polish & Documentation

* [cite_start]The GitHub repository must be public and contain: source code: documentation; installation instructions; inference example; training instructions[cite: 149, 150, 151, 152, 153, 154].
* [cite_start]The repository must execute without modifications after: `pip install -r requirements.txt` and `bash run.sh`[cite: 147, 148].
* Deliverables: A public GitHub repository containing the fully functional source code. [cite_start]The code must execute successfully and reproduce the reported experiments[cite: 176, 177, 178].

---

## 10. Copilot Mastery Section

* **Context is King:** When asking Copilot to write MoE routing logic, keep `modeling_moe.py` open in the active tab so Copilot understands the class structure.
* **Debugging Shape Errors:** Multimodal fusion often breaks at the concatenation step. Use prompt comments to explicitly state expected tensor dimensions to Copilot.
* **Handling PyTorch Device Placement:** Ask Copilot explicitly to ensure all tensors in custom collate functions are mapped to the correct device using accelerate's prepare method.

---

## 11. Technical Report Outline

* [cite_start]Submit a report (minimum 15 pages) including[cite: 156]:
    * [cite_start]1. Introduction [cite: 157]
    * [cite_start]2. Background on Sparse Mixture of Experts [cite: 158]
    * [cite_start]3. Background on Multimodal Large Language Models [cite: 159]
    * [cite_start]4. Description of the datasets [cite: 160]
    * [cite_start]5. Architecture [cite: 161]
    * [cite_start]6. Implementation details [cite: 162]
    * [cite_start]7. Training methodology [cite: 163]
    * [cite_start]8. Experimental evaluation [cite: 164]
    * [cite_start]9. Expert routing analysis [cite: 165]
    * [cite_start]10. Discussion [cite: 166]
    * [cite_start]11. Conclusions [cite: 167]
* [cite_start]Figures should include: architecture diagram; training loss; expert utilization histogram; routing entropy; evaluation results[cite: 168, 169, 170, 171, 172, 173].
* [cite_start]The final technical report must be in PDF format[cite: 180].

---

## 12. Final Checklist & Troubleshooting Guide

[cite_start]The project should emphasize code quality, reproducibility, and a clear understanding of Sparse Mixture of Experts for multimodal artificial intelligence[cite: 221].

- [ ] [cite_start]Does the code execute without modifications after running `pip install -r requirements.txt` and `bash run.sh`? [cite: 147, 148]
- [ ] [cite_start]Is the GitHub repo public? [cite: 149]
- [ ] [cite_start]Are text, audio, and images handled via specific encoders mapped to the shared transformer? [cite: 6]
- [ ] [cite_start]Is the MoE layer tracking routing probabilities and inactive experts? [cite: 50, 53]
- [ ] [cite_start]Is the technical report submitted in PDF format and at least 15 pages? [cite: 156, 180]