# VRAM Analysis — Why Old Config Uses ~80GB

## The Two Processes

The reward server and training script run on the **same GPU(s)**. Together they consume ~80GB.

---

## 1. Reward Server (`reward_server.py`) — ~5-6 GB

| Component | VRAM |
|-----------|------|
| `Qwen2.5-VL-3B-Instruct` model weights (BF16) | ~6 GB |
| vLLM KV cache (40% × GPU, `max_model_len=4096`) | ~2-4 GB |
| **Subtotal** | **~5-8 GB** |

vLLM pre-allocates `gpu_memory_utilization=0.4` of **total** VRAM, so on an 80GB GPU this reserves **~32 GB** even if the model only needs 6 GB. This is the first big problem.

> [!CAUTION]
> **vLLM's `gpu_memory_utilization=0.4` means 40% of the ENTIRE GPU (32GB on A100-80GB), not 40% of what the model needs.** This alone could eat 32GB.

---

## 2. Training Script (`train_nft_qwen_image_edit.py`) — ~50-60 GB

### Model Weights on GPU

| Component | Size (BF16) | Notes |
|-----------|-------------|-------|
| `Qwen-Image-Edit-2511` Transformer | ~14 GB | The main diffusion transformer |
| `Qwen-Image-Edit-2511` Text Encoder (Qwen2.5-VL-7B) | ~14 GB | FSDP-sharded, but with `cpu_offload=False` → stays on GPU |
| `Qwen-Image-Edit-2511` VAE | ~0.5 GB | BF16 |
| **Subtotal (weights)** | **~28.5 GB** |

### LoRA — 3× Transformer Forward Passes per Timestep

This is the **core VRAM killer**. Per training step (lines 1059-1100), the code runs the transformer **3 times**:

```
1. old_prediction    — "old" LoRA adapter (no_grad)        → ~14 GB activations
2. forward_prediction — "default" LoRA adapter (WITH grad) → ~14 GB activations + grad graph
3. ref_forward_prediction — adapter disabled (no_grad)     → ~14 GB activations
```

Even with `gradient_checkpointing`, the forward pass with gradients (`forward_prediction`) stores activation checkpoints. And these are all done at `bsz=3` with `512×512` latents.

### Activation Memory (Peak)

| What | Estimate |
|------|----------|
| `xt_input` latents (batch=3, packed) | ~0.5 GB |
| Transformer activations (with grad checkpointing) | ~4-8 GB |
| Prompt embeddings (1024 seq len × batch) | ~0.3 GB |
| Gradient graph for backward | ~4-8 GB |
| **Peak activation subtotal** | **~10-17 GB** |

### Optimizer State (AdamW)

AdamW stores 2 momentum buffers per trainable param (LoRA only, r=32):

| Component | Estimate |
|-----------|----------|
| LoRA params + 2× momentum | ~1-2 GB |

### Stored Samples Between Sampling and Training

```python
samples_data_list.append({
    "prompt_ids": ...,
    "prompt_embeds": ...,      # [bsz, 1024, hidden_dim] × num_batches_per_epoch
    "latents_clean": ...,       # [bsz, seq, channels] × num_batches_per_epoch  
    "image_latents": ...,       # [bsz, seq, channels] × num_batches_per_epoch
    "timesteps": ...,
})
```

With old config: `num_batches_per_epoch = 24×12 / (1×3) = 96` batches, each storing latents for batch_size=3. That's **96 × 3 = 288 latent sets** held in GPU memory simultaneously.

> [!WARNING]
> **`samples_data_list` accumulates ALL sampling results on GPU before training begins.** With 96 batches × bsz=3, latent storage alone can use **5-10 GB**.

---

## Total VRAM Breakdown (Old Config, Single GPU)

| Component | VRAM |
|-----------|------|
| vLLM reservation (reward server) | ~32 GB (40% of 80GB) |
| Training: Model weights (transformer + text encoder + VAE) | ~28.5 GB |
| Training: LoRA optimizer state | ~1-2 GB |
| Training: Activations + gradients (peak) | ~10-17 GB |
| Training: Stored samples between phases | ~5-10 GB |
| **Total** | **~76-89 GB** ❌ OOM |

---

## Root Causes (Ranked by Impact)

### 🔴 #1: vLLM pre-allocates 40% of the ENTIRE GPU
On an 80GB A100, `gpu_memory_utilization=0.4` reserves **32GB** regardless of actual model size. The 3B model only needs ~6GB.

**Fix:** Already changed to `0.3` (24GB). But ideally, run reward server on a **separate GPU** or use `gpu_memory_utilization=0.15` if sharing.

### 🔴 #2: Text Encoder loaded on GPU without CPU offload  
```python
prepare_fsdp_model(
    pipeline.text_encoder,
    cpu_offload=False,  # ← 14GB text encoder stays on GPU!
)
```
The Qwen2.5-VL-7B text encoder is ~14GB in BF16. It's only used during the sampling phase, but stays resident.

**Fix:** Set `cpu_offload=True` for the text encoder.

### 🔴 #3: 3× transformer forward passes per training timestep
The GRPO training requires `old`, `default`, and `ref` predictions per timestep. Each runs the full ~14GB transformer. Even with gradient checkpointing, peak memory is high.

**Fix:** No easy fix — this is architecturally required by the NFT/GRPO algorithm.

### 🟡 #4: 96 sampling batches stored in GPU memory
With `num_groups=24`, `num_image_per_prompt=12`, `bsz=3`: that's 96 batches of latents accumulated before training.

**Fix:** Already reduced to 16 groups × 8 images = 128 total / (1×2) = 64 batches.

### 🟡 #5: Resolution 512×512
Bigger images = bigger latents = more memory for activations and stored samples.

**Fix:** Already reduced to 384×384.
