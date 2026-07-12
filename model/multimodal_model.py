"""
model/multimodal_model.py
Multimodal Sparse MoE Model: audio + vision encoders, fusion module, and full model.

Architecture:
  1. AudioEncoderWrapper    — openai/whisper-small  -> linear projection -> phi_hidden_size
  2. VisionEncoderWrapper   — openai/clip-vit-base-patch32 -> FFN projection -> phi_hidden_size
  3. MultimodalFusionModule — projection + concatenation of text / image / audio embeddings
  4. MultimodalMoEModel     — full model: encoders + fusion + Phi backbone
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPVisionModel,
    WhisperModel,
)

from model.modeling_moe import SparseMoELayer


# ---------------------------------------------------------------------------
# Audio Encoder Wrapper
# ---------------------------------------------------------------------------

class AudioEncoderWrapper(nn.Module):
    """Wraps the Whisper encoder and projects its output to phi_hidden_size.

    Input:
        input_features: (batch, num_mel_bins, time_frames)  — mel-spectrogram at 16 kHz
    Output:
        audio_embeds:   (batch, T_audio, phi_hidden_size)
    """

    WHISPER_MODEL = "openai/whisper-small"
    SAMPLE_RATE   = 16_000

    def __init__(
        self,
        phi_hidden_size: int,
        model_name: Optional[str] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        name = model_name or self.WHISPER_MODEL
        print(f"[AudioEncoderWrapper] Loading {name} ...")

        self.whisper_encoder = WhisperModel.from_pretrained(name).encoder
        whisper_hidden = self.whisper_encoder.config.d_model   # 512 for whisper-small

        self.projection = nn.Linear(whisper_hidden, phi_hidden_size)

        if freeze_encoder:
            for p in self.whisper_encoder.parameters():
                p.requires_grad = False

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch, num_mel_bins, time_frames) at 16 kHz
        Returns:
            audio_embeds: (batch, T_audio, phi_hidden_size)
        """
        encoder_out = self.whisper_encoder(input_features=input_features)
        hidden      = encoder_out.last_hidden_state          # (B, T_audio, whisper_hidden)
        return self.projection(hidden)                       # (B, T_audio, phi_hidden_size)


# ---------------------------------------------------------------------------
# Vision Encoder Wrapper
# ---------------------------------------------------------------------------

class VisionEncoderWrapper(nn.Module):
    """Wraps the CLIP ViT-B/32 vision encoder and projects patch embeddings to phi_hidden_size.

    Input:
        pixel_values: (batch, 3, 224, 224) — normalized to CLIP's expected range
    Output:
        image_embeds: (batch, T_image, phi_hidden_size)
                      T_image = 197  (CLS + 196 patches for ViT-B/32) when use_patch_tokens=True
                             = 1     when use_patch_tokens=False (pooled CLS only)
    """

    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        phi_hidden_size: int,
        model_name: Optional[str] = None,
        use_patch_tokens: bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        name = model_name or self.CLIP_MODEL
        print(f"[VisionEncoderWrapper] Loading {name} ...")

        self.clip_vision     = CLIPVisionModel.from_pretrained(name)
        clip_hidden          = self.clip_vision.config.hidden_size   # 768 for ViT-B/32
        self.use_patch_tokens = use_patch_tokens

        # Two-layer MLP projection
        self.projection = nn.Sequential(
            nn.Linear(clip_hidden, phi_hidden_size * 2),
            nn.GELU(),
            nn.Linear(phi_hidden_size * 2, phi_hidden_size),
        )

        if freeze_encoder:
            for p in self.clip_vision.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, 224, 224)
        Returns:
            image_embeds: (batch, T_image, phi_hidden_size)
        """
        vision_out = self.clip_vision(pixel_values=pixel_values, output_hidden_states=False)

        if self.use_patch_tokens:
            hidden = vision_out.last_hidden_state          # (B, 197, clip_hidden)
        else:
            hidden = vision_out.pooler_output.unsqueeze(1) # (B, 1, clip_hidden)

        return self.projection(hidden)                     # (B, T_image, phi_hidden_size)


# ---------------------------------------------------------------------------
# Multimodal Fusion Module (projection + concatenation)
# ---------------------------------------------------------------------------

class MultimodalFusionModule(nn.Module):
    """Concatenation-based multimodal fusion.

    Applies per-modality LayerNorm for stability, then concatenates
    all available modality embeddings along the sequence dimension.

    Args:
        phi_hidden_size: Shared hidden dimension for all modalities.
    """

    def __init__(self, phi_hidden_size: int):
        super().__init__()
        self.text_norm  = nn.LayerNorm(phi_hidden_size)
        self.image_norm = nn.LayerNorm(phi_hidden_size)
        self.audio_norm = nn.LayerNorm(phi_hidden_size)

    def forward(
        self,
        text_embeds:  torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embeds:  (batch, T_text,  phi_hidden_size)
            image_embeds: (batch, T_image, phi_hidden_size) or None
            audio_embeds: (batch, T_audio, phi_hidden_size) or None
        Returns:
            fused:         (batch, T_total, phi_hidden_size)
            attention_mask (batch, T_total) — all ones
        """
        parts = [self.text_norm(text_embeds)]
        if image_embeds is not None:
            parts.append(self.image_norm(image_embeds))
        if audio_embeds is not None:
            parts.append(self.audio_norm(audio_embeds))

        fused = torch.cat(parts, dim=1)                    # (B, T_total, H)
        attention_mask = torch.ones(
            fused.shape[:2], dtype=torch.long, device=fused.device
        )
        return fused, attention_mask


# ---------------------------------------------------------------------------
# Full Multimodal MoE Model
# ---------------------------------------------------------------------------

class MultimodalMoEModel(nn.Module):
    """End-to-end Multimodal Sparse Mixture of Experts model.

    Architecture:
        1. Encode audio  (Whisper)     -> project -> phi_hidden_size
        2. Encode images (CLIP ViT)    -> project -> phi_hidden_size
        3. Embed text                  -> phi_hidden_size  (Phi embedding table)
        4. Fuse all modalities         -> concatenate along sequence dim
        5. Pass fused embeddings       -> Phi backbone (causal LM)
        6. Custom SparseMoELayer       -> tracked in self.moe_layers (ModuleDict)

    Args:
        phi_model_name:          HuggingFace model ID for the Phi backbone.
        num_moe_experts:         Experts per SparseMoE layer.
        top_k:                   Top-K experts per token.
        capacity_factor:         MoE capacity factor.
        aux_loss_weight:         Weight for the auxiliary load-balancing loss.
        replace_layers:          Phi layer indices to inject a SparseMoE.
                                 If None, replaces every other layer (0, 2, 4, ...).
        freeze_encoders:         Whether to freeze Whisper and CLIP weights.
        torch_dtype:             Model dtype (torch.bfloat16 recommended on Colab A100/L4).
    """

    PHI_MODEL_NAME = "microsoft/Phi-mini-MoE-instruct"

    def __init__(
        self,
        phi_model_name:  Optional[str]  = None,
        num_moe_experts: int            = 8,
        top_k:           int            = 2,
        capacity_factor: float          = 1.25,
        aux_loss_weight: float          = 0.01,
        replace_layers:  Optional[List[int]] = None,
        freeze_encoders: bool           = True,
        torch_dtype:     torch.dtype    = torch.bfloat16,
    ):
        super().__init__()
        name = phi_model_name or self.PHI_MODEL_NAME
        print(f"[MultimodalMoEModel] Loading Phi backbone: {name}")

        # ---- Phi backbone ----
        self.config = AutoConfig.from_pretrained(name, trust_remote_code=True)
        # Patch rope_scaling: newer transformers uses "rope_type" key but the
        # model's custom code expects the old "type" key.
        if (self.config.rope_scaling and
                "type" not in self.config.rope_scaling and
                "rope_type" in self.config.rope_scaling):
            if self.config.rope_scaling["rope_type"] == "default":
                self.config.rope_scaling = None  # model treats None as default
            else:
                self.config.rope_scaling["type"] = self.config.rope_scaling["rope_type"]
        self.backbone = AutoModelForCausalLM.from_pretrained(
            name, config=self.config, dtype=torch_dtype, trust_remote_code=True,
            attn_implementation="eager", device_map="auto",
            offload_folder="/content/offload"
        )

        hidden_size       = self.config.hidden_size
        num_backbone_layers = self.config.num_hidden_layers

        # ---- Inject SparseMoELayer into selected positions ----
        if replace_layers is None:
            replace_layers = list(range(0, num_backbone_layers, 2))

        intermediate_size = getattr(self.config, "intermediate_size", hidden_size * 4)

        self.moe_layers: Dict[str, SparseMoELayer] = nn.ModuleDict()
        for idx in replace_layers:
            self.moe_layers[str(idx)] = SparseMoELayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_moe_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                aux_loss_weight=aux_loss_weight,
            )
        print(f"[MultimodalMoEModel] SparseMoE injected at layers: {replace_layers}")

        # ---- Modality encoders ----
        self.audio_encoder  = AudioEncoderWrapper(
            phi_hidden_size=hidden_size, freeze_encoder=freeze_encoders
        )
        self.vision_encoder = VisionEncoderWrapper(
            phi_hidden_size=hidden_size, freeze_encoder=freeze_encoders
        )

        # ---- Fusion module ----
        self.fusion = MultimodalFusionModule(phi_hidden_size=hidden_size)

        # ---- Text embedding (reuse Phi's embedding table, no copy) ----
        self._embed_tokens = self.backbone.get_input_embeddings()

    def _apply_demo_moe(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the custom SparseMoE stack to fused embeddings before the backbone.

        This keeps the demo path simple and observable on Colab without replacing
        Phi's internal FFN modules.
        """
        total_aux_loss = hidden_states.new_zeros(())
        if not self.moe_layers:
            return hidden_states, total_aux_loss

        moe_hidden = hidden_states
        for layer_idx in sorted(self.moe_layers.keys(), key=int):
            layer = self.moe_layers[layer_idx]
            if any(param.device != moe_hidden.device or param.dtype != moe_hidden.dtype
                   for param in layer.parameters()):
                layer = layer.to(device=moe_hidden.device, dtype=moe_hidden.dtype)
                self.moe_layers[layer_idx] = layer
            moe_hidden, aux_loss = layer(moe_hidden)
            total_aux_loss = total_aux_loss + aux_loss

        return moe_hidden, total_aux_loss

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def get_routing_stats(self) -> Dict[str, dict]:
        """Aggregate routing stats from all SparseMoE layers."""
        return {f"layer_{idx}": layer.get_routing_stats()
                for idx, layer in self.moe_layers.items()}

    def reset_moe_tracking(self):
        for layer in self.moe_layers.values():
            layer.reset_tracking()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:      Optional[torch.Tensor] = None,
        pixel_values:   Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (batch, T_text)
            pixel_values:   (batch, 3, 224, 224)  or None
            input_features: (batch, 80, 3000)     or None — Whisper mel-spectrogram
            attention_mask: (batch, T_text)        or None
            labels:         (batch, T_text)        or None — causal LM labels
        Returns:
            dict with keys: 'loss', 'aux_loss', 'total_loss', 'logits'
        """
        embed_device = self._embed_tokens.weight.device
        device = embed_device

        # 1. Text embeddings via Phi embedding table
        text_embeds = self._embed_tokens(input_ids.to(embed_device))  # (B, T_text, H)

        # 2. Optional vision embeddings
        image_embeds = None
        if pixel_values is not None:
            vision_device = next(self.vision_encoder.parameters()).device
            image_embeds = self.vision_encoder(pixel_values.to(vision_device)).to(embed_device)

        # 3. Optional audio embeddings
        audio_embeds = None
        if input_features is not None:
            audio_device = next(self.audio_encoder.parameters()).device
            audio_embeds = self.audio_encoder(input_features.to(audio_device)).to(embed_device)

        # 4. Fuse all modalities
        fused_embeds, fused_mask = self.fusion(text_embeds, image_embeds, audio_embeds)

        # 5. Demo-safe custom MoE path on fused embeddings
        fused_embeds, total_aux_loss = self._apply_demo_moe(fused_embeds)

        # 6. Phi backbone forward (via inputs_embeds — skip token embedding layer)
        backbone_out = self.backbone(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            labels=None,
            output_hidden_states=False,
            **kwargs,
        )
        logits = backbone_out.logits  # (B, T_fused, vocab_size)

        # 7. Causal LM loss restricted to the text prefix tokens
        task_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            labels       = labels.to(logits.device)
            T_text       = input_ids.shape[1]
            text_logits  = logits[:, :T_text, :].contiguous()
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        total_loss = task_loss + total_aux_loss

        return {
            "loss":       task_loss,
            "aux_loss":   total_aux_loss,
            "total_loss": total_loss,
            "logits":     logits,
        }
