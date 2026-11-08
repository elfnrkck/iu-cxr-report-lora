# Release status and model-weight publication

Repository created by the owner; source upload is pending because the integration returned GitHub HTTP 403:
https://github.com/elfnrkck/iu-cxr-report-lora

Version: 0.1.0rc1. This repository distributes inference software, documentation,
fictional example reports and software-validation records. The trained model
adapters are prepared separately. No public Hugging Face model repository or
DOI is claimed yet.

## Completed

- Original IU-only adapter identities and byte hashes were checked.
- Fourteen unit tests passed; actual PEFT loading and CLIP token packing were
  verified on CPU.
- A user-run Colab GPU acceptance run passed. The independently inspected
  reference and packaged outputs for one fixed long fixture and seed 42 had
  zero pixel difference.
- MIMIC-trained weights, source radiographs/reports, patient-level manifests
  and private Drive links are excluded from this code repository.

## Remaining model release

1. Choose the author's Hugging Face account/organization and repository ID.
2. Confirm the applicable IU source terms and contributor/citation metadata.
   Source-data rights are separate from model-adapter licensing.
3. Upload the prepared huggingface_model/ directory using the authenticated
   owner's environment. Preserve the CreativeML Open RAIL-M license,
   modification notice, tokenizer notices and checkpoint hashes.
4. Add the real model URL here and the source URL to its model card. Test a
   clean download, loading and generation before claiming open model access.
5. Record source and model commits in the manuscript. A DOI may be obtained
   through a suitable archive once release ownership and metadata are settled.

Use interactive authentication or a secret manager. Never commit tokens,
caches, patient reports or clinical-reader responses.

Primary documentation:
- https://huggingface.co/docs/hub/model-cards
- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
