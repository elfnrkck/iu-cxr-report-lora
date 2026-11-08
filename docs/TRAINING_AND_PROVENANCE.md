# Training and inference provenance

Released arm: unet_full, full-report conditioning with UNet-only LoRA.
“UNet-only” describes trainable weights; reports condition every image.

| Item | Recovered value |
| --- | --- |
| Training source | IU X-Ray only |
| Training seeds | 42, 1337, 2026 |
| Base | Stable Diffusion 1.5 |
| Revision | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 |
| Rank / alpha / dropout | 32 / 32 / 0 |
| Targets | to_q, to_k, to_v, to_out.0 |
| UNet adapter parameters | 6,377,472 |
| Trainable text-encoder parameters | 0 |
| Format / tensor count | safetensors FP32 / 256 |
| Case draws per epoch | 2,512 |
| Epochs / updates | 5 / 3,140 |
| Updates per epoch | 628 |
| Effective case batch | 4, inferred from 2,512 / 628 |
| Checkpoint | Fixed final update |
| Conditioning | Lossless sentence-packed CLIP windows |
| Sampling | DDIM, 30 steps, CFG 7.5, eta 0 |
| Output | 512 × 512 RGB PNG with radiograph appearance |

The recovered sampler chooses one frontal image per training case each epoch
and rejects held-out rows. Saved states record 12,560 total case exposures.
The recovered training code uses AdamW (betas 0.9/0.999, eps 1e-8,
weight decay 0.01), warmup, and cosine decay to 10% of the initial rate.
Numerical learning rate and warmup fraction are not reasserted in this minimal
release: the complete original configuration is not included. This is an
inference release and partial documented recipe, not turnkey training reproduction.

Recorded protocol identifier:

    e5e4a2b3257d276db561c15b838b1df3d630c8e79b0f4ba5040e7b11c55997d8

Exact component hashes are in model_registry.json. Original configurations
and saved states are included unchanged. The loader checks key names, shapes,
every loaded tensor value, and the expected parameter scope. It does not
merge adapters into base weights. The reference functions remain available.

Default precision is BF16 UNet/CLIP, FP32 LoRA and VAE. Initial noise is CPU
FP32. A generic short-prompt pipeline can truncate reports and is not equivalent.
Cross-GPU bitwise identity is not guaranteed.
