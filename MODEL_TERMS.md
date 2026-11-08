# Model-adapter terms and modification notice

The IU-only UNet adapters modify Stable Diffusion 1.5. Base weights are not
included. The adapters inherit CreativeML Open RAIL-M; its complete text is
preserved in licenses/CreativeML_OpenRAIL_M.txt.

Modification notice: the project contributors trained rank-32 UNet attention
LoRA adapters on frontal IU report–image pairs, froze CLIP and VAE, and added
lossless full-report inference. No MIMIC-trained checkpoint is included.

The software MIT license does not remove model use restrictions. Preserve the
complete model license, provenance, and modification notice on redistribution.
Consult Attachment A: this research release is not intended to provide medical
advice or medical-results interpretation. Source-dataset access rights are
separate. No assertion is made that source data are public domain.
