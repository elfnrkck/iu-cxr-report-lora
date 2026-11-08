# IU-CXR Report LoRA

Research software for generating a synthetic frontal chest radiograph from
an English radiology report using a Stable Diffusion 1.5 UNet LoRA trained on
IU X-Ray. Reports condition every output; the CLIP encoder remains frozen.
A source radiograph is not required at inference.

**Release candidate: 0.1.0rc1.** This public repository contains inference
code, a Colab notebook, documentation, fictional example reports, and
software-validation records. The three trained adapters are prepared
separately; a public model-weight download URL has not yet been assigned.
No source patient reports, radiographs, patient-level split lists, or
MIMIC-trained model weights are included.

[Türkçe rehber](docs/README_TR.md) · [Training provenance](docs/TRAINING_AND_PROVENANCE.md)
· [Verification](verification/colab_acceptance/README.md)

## Installation

Use a fresh Python environment with Python 3.10 or newer and CUDA-enabled
PyTorch. The research precision profile is BF16 UNet/CLIP with FP32 adapters
and VAE decoding. L4, A100, H100, and RTX 4090 support native BF16 operation.

    git clone https://github.com/elfnrkck/iu-cxr-report-lora.git
    cd iu-cxr-report-lora
    python -m pip install -e .

Base weights are downloaded separately (about 4.3 GB) from:
stable-diffusion-v1-5/stable-diffusion-v1-5, revision
451f4fe16113bff5a5d2269ed5ad43b0592e9a14.
Use --base-model-dir for an existing exact snapshot.

## Model weights and generation

Obtain the author-provided IU release archive and extract its
huggingface_model/ directory. It must contain model_registry.json and
seed_42/, seed_1337/, seed_2026/. The GitHub clone alone does not contain these
weights. Do not substitute a MIMIC-trained continuation or a different
checkpoint: the loader checks the exact registered hashes.

If the model directory is next to this repository:

    iu-cxr-generate --model-dir ../huggingface_model --verify-only
    iu-cxr-generate --model-dir ../huggingface_model --report-file examples/report_short.txt --output outputs/example.png

A dependency-free weight-integrity check is also available:

    PYTHONPATH=src python -m iu_cxr.cli --model-dir ../huggingface_model --verify-only

Python interface:

    from iu_cxr.pipeline import IUReportGenerator
    generator = IUReportGenerator("../huggingface_model", training_seed=42)
    image, metadata = generator.generate(
        "Frontal chest radiograph. The cardiac silhouette is enlarged.",
        generation_seed=202601,
    )
    image.save("synthetic.png")

Training seed selects one of three independently trained adapters:
42, 1337, 2026. Generation seed selects the initial Gaussian noise.
Default training seed 42 is fixed, not selected as the best checkpoint.

## Full-report conditioning

Reports are packed losslessly into 75-content-token CLIP windows with
BOS/EOS padding. Each window is encoded separately, and the representations
are concatenated for UNet cross-attention. Unconditional conditioning uses
the same number of empty windows. The default 16-window limit rejects
overlong inputs explicitly; it does not silently truncate them.
No automatic Turkish translation is performed.

Default sampling: DDIM, 30 steps, guidance 7.5, eta 0, 512 × 512.
Initial noise is sampled in CPU FP32. CLI generation writes a PNG and a
provenance JSON with model hashes, both seeds, token/window counts, sampler
settings, precision and runtime versions. The report text is not copied into
that sidecar; its hash is recorded.

## Verified software behavior

Fourteen dependency-free unit tests passed. Actual adapter loading and the
frozen CLIP tokenizer were also checked on CPU. A separate user-run Colab GPU
acceptance run passed with all three adapters. That run reported Tesla T4,
Python 3.13.16 and PyTorch 2.14.1+cu130 with BF16 tensors; native BF16 hardware
acceleration was not assessed in that run.

- For seed 42 and the long fictional fixture, packaged and preserved study
  samplers produced identical pixels and PNG hashes.
- The 203-token long fixture retained four CLIP windows; the short fixture
  used one window and 30 content tokens.
- All three adapters used identical initial noise for the same generation seed.
- Disabling the adapter changed the output.
- Six output files include one identical reference (five unique PNG hashes).

These are software acceptance checks, not a clinical evaluation, privacy
guarantee, speed benchmark, or cross-hardware equivalence claim.
See verification/colab_acceptance/independent_verification.json.

The fixed verification prompt is available in examples/report_long.txt.
Generated PNG files are retained separately from this public source tree.
Their SHA-256 hashes, sampler settings and independently checked comparison
results are recorded in verification/colab_acceptance/.

## Colab and local checks

notebooks/IU_Yayin_Dogrulama_Colab.ipynb runs the six-output acceptance check.
It currently requires the author-provided full release ZIP. Its revised setup
uses separate Python processes and logs failures from each stage.
It performs no model training.

    PYTHONPATH=src python -m unittest discover -s tests -v

GitHub Actions runs those dependency-free tests on pushes and pull requests.
GPU generation is not part of the GitHub Actions workflow.

## Scope, limitations and terms

Research generator; not for diagnosis, patient management, or medical advice.
Images may contain incorrect anatomy, missing/invented findings and bias.
Retaining all report tokens does not establish clinical finding fidelity.
Keep generated images labelled synthetic. This release does not establish
absence of memorization or membership-inference risk.

This is an inference implementation with original reference functions and a
partial documented training recipe. It does not claim turnkey reproduction
of the complete data preparation, model training, reader study or manuscript
statistics. Original data must be obtained from the provider under its terms.

Original software: MIT (LICENSE). Trained adapters and base model:
CreativeML Open RAIL-M, including use restrictions (MODEL_TERMS.md).
Third-party tokenizer files retain their source terms. See
THIRD_PARTY_NOTICES.md, licenses/ and docs/DATA_AND_LICENSES.md.
Source-data rights are separate from software/model licensing.

Citation metadata is in CITATION.cff. A manuscript citation, confirmed author
metadata and DOI can be added when available; none is invented here.
