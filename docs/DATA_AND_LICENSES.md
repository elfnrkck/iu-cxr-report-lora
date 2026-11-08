# Data access and release boundary

This candidate contains only the IU-only UNet adapters in model_registry.json.
It excludes MIMIC-trained continuations, source radiographs/reports, clinical
reader responses, private Drive links, patient IDs, and patient-level splits.
Example reports are newly authored fictional software fixtures.

Obtain IU data from the original provider. Open-i's FAQ asks researchers not
to redistribute the dataset outside their research group and to direct others
to NLM. It does not explicitly settle distribution of trained LoRA parameters.
Authors should confirm the terms applicable to their downloaded IU collection
and institutional arrangements before public upload. No provider clearance
for trained weights is asserted by this package.

PhysioNet provides a separate process for MIMIC-derived datasets and models.
Removing a data directory would not make a MIMIC-trained checkpoint IU-only.
All MIMIC-trained checkpoints are excluded from this candidate.

Software, weights, tokenizer files, base model, and source dataset have
separate terms; see LICENSE, MODEL_TERMS.md, and THIRD_PARTY_NOTICES.md.

Primary sources:
- https://openi-vip.nlm.nih.gov/faq
- https://doi.org/10.1093/jamia/ocv080
- https://physionet.org/news/post/mimic-derived-datasets-models/
- https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- https://github.com/CompVis/stable-diffusion/blob/main/LICENSE
