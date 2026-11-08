"""Paper-matched full-report conditioning and DDIM inference.

Model loading is intentionally separate from StableDiffusionPipeline's
short-prompt tokenizer and its generic LoRA file conversion heuristics.
"""
from pathlib import Path
from contextlib import nullcontext
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import platform
import time

from .conditioning import plan_report, empty_windows
from .integrity import checked_path, read_registry, sha256_file, verify_model


class IUReportGenerator:
    def __init__(self, model_dir, training_seed=42, base_model_dir=None,
                 device='cuda', precision='bfloat16', max_windows=16):
        import torch
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig
        self.torch=torch
        self.model_dir=Path(model_dir).resolve()
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use a GPU runtime, or explicitly select CPU float32.')
        if precision not in ['bfloat16','float16','float32']:
            raise ValueError('Unknown precision.')
        if self.device.type=='cpu' and precision!='float32':
            raise ValueError('CPU inference requires explicit float32 precision.')
        if precision=='bfloat16' and self.device.type=='cuda' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support BF16. L4/A100/H100 supports the paper precision; float16 is a compatibility mode.')
        self.precision=precision
        self.dtype=getattr(torch,precision)
        self.max_windows=max_windows
        self.base_record=json.loads((Path(__file__).parent/'data/base_model.json').read_text())
        self.adapter_check=verify_model(self.model_dir,training_seed)
        if base_model_dir is None:
            from huggingface_hub import snapshot_download
            self.base=Path(snapshot_download(repo_id=self.base_record['model_id'],revision=self.base_record['revision'],
                allow_patterns=list(self.base_record['files'])+['model_index.json']))
        else:
            self.base=Path(base_model_dir).resolve()
        for relative,record in self.base_record['files'].items():
            if sha256_file(checked_path(self.base,relative,True))!=record['sha256']:
                raise ValueError('Base-model file differs from the pinned paper snapshot: '+relative)
        tok_root=Path(__file__).parent/'data/tokenizer'
        tok_hashes=json.loads((Path(__file__).parent/'data/tokenizer_hashes.json').read_text())
        for name,sha in tok_hashes.items():
            if sha256_file(tok_root/name)!=sha:
                raise ValueError('Frozen tokenizer changed: '+name)
        self.tokenizer=CLIPTokenizer.from_pretrained(str(tok_root),local_files_only=True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        self.unet=UNet2DConditionModel.from_pretrained(str(self.base),subfolder='unet',
            local_files_only=True,use_safetensors=True,torch_dtype=self.dtype,low_cpu_mem_usage=True).to(self.device)
        self.text_encoder=CLIPTextModel.from_pretrained(str(self.base),subfolder='text_encoder',
            local_files_only=True,use_safetensors=True,torch_dtype=self.dtype,low_cpu_mem_usage=True).to(self.device)
        self.vae=AutoencoderKL.from_pretrained(str(self.base),subfolder='vae',
            local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,low_cpu_mem_usage=True).to(self.device)
        if self.text_encoder.config.max_position_embeddings!=77 or float(self.vae.config.scaling_factor)!=0.18215:
            raise ValueError('Unexpected CLIP or VAE configuration.')
        for model in [self.unet,self.text_encoder,self.vae]:
            model.requires_grad_(False).eval()
        torch.manual_seed(int(training_seed))
        self.unet.add_adapter(LoraConfig(r=32,lora_alpha=32,lora_dropout=0.0,
            target_modules=['to_q','to_k','to_v','to_out.0'],init_lora_weights='gaussian',bias='none'),adapter_name='default')
        count=0
        for name,parameter in self.unet.named_parameters():
            if parameter.requires_grad:
                if 'lora_' not in name:
                    raise ValueError('Unexpected trainable base-model parameter.')
                parameter.data=parameter.data.float()
                count+=parameter.numel()
        if count!=6377472:
            raise ValueError('Wrong LoRA parameter scope.')
        self.scheduler=DDIMScheduler.from_pretrained(str(self.base),subfolder='scheduler',local_files_only=True)
        self.set_training_seed(training_seed)

    def set_training_seed(self, training_seed):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        check=verify_model(self.model_dir,training_seed)
        item=read_registry()['models'][str(training_seed)]
        state=load_file(str(checked_path(self.model_dir,item['weights'],True)),device='cpu')
        current=get_peft_model_state_dict(self.unet)
        if set(state)!=set(current) or any(state[k].shape!=current[k].shape for k in state):
            raise ValueError('Checkpoint keys or shapes differ from the attached adapter.')
        result=set_peft_model_state_dict(self.unet,state,adapter_name='default')
        if result.unexpected_keys:
            raise ValueError('Unexpected checkpoint keys after loading.')
        actual=get_peft_model_state_dict(self.unet)
        if any(not self.torch.equal(actual[k].detach().float().cpu(),state[k]) for k in state):
            raise ValueError('Loaded LoRA values differ from the original file.')
        self.unet.requires_grad_(False).eval()
        self.training_seed=int(training_seed)
        self.adapter_check=check

    def _autocast(self):
        if self.device.type=='cuda' and self.precision!='float32':
            return self.torch.autocast(self.device.type,dtype=self.dtype)
        return nullcontext()

    def _encode(self, windows):
        torch=self.torch
        ids=torch.tensor([windows],dtype=torch.long,device=self.device)
        b,k,_=ids.shape
        hidden=self.text_encoder(input_ids=ids.reshape(b*k,77),return_dict=True).last_hidden_state
        hidden=hidden.reshape(b,k*77,hidden.shape[-1])
        if not torch.isfinite(hidden).all().item():
            raise FloatingPointError('Nonfinite CLIP embeddings.')
        return hidden

    def runtime(self):
        packages={}
        for name in ['torch','diffusers','transformers','peft','accelerate','huggingface-hub','safetensors','numpy','pillow']:
            try:packages[name]=version(name)
            except PackageNotFoundError:packages[name]='not installed'
        return {'python':platform.python_version(),'packages':packages,'device':str(self.device),
            'gpu':self.torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
            'precision':self.precision,'base_revision':self.base_record['revision'],
            'tf32_allowed':False,'cross_hardware_bitwise_equivalence_guaranteed':False}

    def generate(self, report, generation_seed=202601, steps=30, guidance_scale=7.5, progress=False):
        import numpy as np
        from PIL import Image
        torch=self.torch
        if not isinstance(generation_seed,int) or not 0<=generation_seed<2**63:
            raise ValueError('Generation seed must be an integer in [0, 2**63).')
        if not isinstance(steps,int) or not 1<=steps<=1000 or not 0<=float(guidance_scale)<=100:
            raise ValueError('Invalid sampler settings.')
        plan=plan_report(report,self.tokenizer,self.max_windows)
        self.scheduler.set_timesteps(steps,device=self.device)
        started=time.monotonic()
        with torch.no_grad():
            with self._autocast():
                conditional=self._encode(plan.input_ids)
                unconditional=self._encode(empty_windows(plan.windows))
            embeddings=torch.cat([unconditional,conditional])
            rng=torch.Generator(device='cpu').manual_seed(generation_seed)
            initial=torch.randn((1,4,64,64),generator=rng,dtype=torch.float32)
            noise_sha=hashlib.sha256(initial.contiguous().numpy().tobytes()).hexdigest()
            latent=initial.to(self.device,dtype=self.dtype)*self.scheduler.init_noise_sigma
            for index,timestep in enumerate(self.scheduler.timesteps,1):
                model_input=self.scheduler.scale_model_input(torch.cat([latent,latent]),timestep)
                with self._autocast():
                    predicted=self.unet(model_input,timestep,encoder_hidden_states=embeddings).sample
                noise_u,noise_c=predicted.chunk(2)
                guided=noise_u+float(guidance_scale)*(noise_c-noise_u)
                latent=self.scheduler.step(guided,timestep,latent,eta=0.0).prev_sample
                if progress and (index==1 or index%10==0 or index==steps):
                    print(f'Sampling {index}/{steps}',flush=True)
            decoded=self.vae.decode(latent.float()/self.vae.config.scaling_factor).sample
            if not torch.isfinite(decoded).all().item():
                raise FloatingPointError('Nonfinite decoded image.')
            array=decoded[0].float().add(1).div(2).clamp(0,1).permute(1,2,0).cpu().numpy()
            image=Image.fromarray(np.rint(array*255).astype('uint8'))
        adapter_enabled=not any(getattr(module,'disable_adapters',False) is True
            for module in self.unet.modules() if hasattr(module,'lora_A'))
        metadata={'release':'0.1.0rc1','training_data':'IU X-Ray only','training_seed':self.training_seed,
            'adapter_enabled':adapter_enabled,
            'adapter_sha256':self.adapter_check['sha256'],'generation_seed':generation_seed,
            'initial_cpu_noise_sha256':noise_sha,'steps':steps,'guidance_scale':float(guidance_scale),'eta':0.0,
            'resolution':[512,512],'content_tokens':plan.content_tokens,'CLIP_windows':plan.windows,
            'conditioning_positions':77*plan.windows,'tokens_truncated':0,
            'token_ids_sha256':plan.token_ids_sha256,
            'normalized_report_sha256':hashlib.sha256(plan.text.encode()).hexdigest(),
            'elapsed_seconds':time.monotonic()-started,'runtime':self.runtime(),
            'synthetic_image':True,'clinical_accuracy_measured':False,
            'sampler_defaults_match_paper':steps==30 and guidance_scale==7.5 and self.precision=='bfloat16'}
        return image,metadata

