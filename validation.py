"""A small software acceptance check using fictional reports, not a new efficacy study."""
from pathlib import Path
from contextlib import nullcontext
import argparse,json,time,traceback
from .conditioning import plan_report,normalize_report
from .integrity import verify_model,sha256_file


def run_validation(model_dir,examples_dir,output_dir,base_model_dir=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    status={'status':'RUNNING','software_validation_only':True,'new_training_updates':0,
            'fictional_reports_only':True,'clinical_performance_evaluated':False}
    def save():
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();started=time.monotonic()
    try:
        checks=[verify_model(model_dir,seed) for seed in [42,1337,2026]]
        status['checkpoint_checks']=checks;save()
        import torch,numpy as np
        from .pipeline import IUReportGenerator
        from .reference import frozen
        frozen.torch=torch;frozen.np=np;frozen.nullcontext=nullcontext
        def finite(tensor,name):
            if not torch.isfinite(tensor).all().item():raise FloatingPointError(name)
        frozen.r4_finite=finite
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This paper-precision comparison needs a BF16 CUDA GPU such as L4, A100, H100, or RTX 4090. High RAM is not required.')
        generator=IUReportGenerator(model_dir,42,base_model_dir,precision='bfloat16')
        examples=Path(examples_dir)
        long_report=(examples/'report_long.txt').read_text(encoding='utf-8')
        short_report=(examples/'report_short.txt').read_text(encoding='utf-8')
        long_plan=plan_report(long_report,generator.tokenizer)
        if long_plan.windows<2:raise ValueError('Long-report fixture must require multiple CLIP windows.')
        if plan_report(short_report,generator.tokenizer).windows!=1:
            raise ValueError('Short-report fixture must fit one CLIP window.')
        for text in [long_report,short_report,'A &amp;amp; B. No acute process.']:
            plan=plan_report(text,generator.tokenizer)
            original=frozen.r4_token_plan(normalize_report(text),generator.tokenizer)
            if plan.input_ids!=json.loads(original['input_id_chunks_json']):
                raise ValueError('New report packing differs from the original study function.')
        status['real_CLIP_tokenizer_matches_frozen_study_packing']=True
        outputs=[]
        def keep(name,image,metadata):
            path=out/(name+'.png');image.save(path,format='PNG')
            metadata=dict(metadata,png_sha256=sha256_file(path))
            (out/(name+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
            outputs.append({'file':path.name,'png_sha256':metadata['png_sha256']})
        noise_hashes=[]
        for seed in [42,1337,2026]:
            print('Checking training seed',seed,flush=True)
            generator.set_training_seed(seed)
            image,metadata=generator.generate(long_report,generation_seed=202601,progress=True)
            keep('long_seed_'+str(seed),image,metadata)
            noise_hashes.append(metadata['initial_cpu_noise_sha256'])
            if seed==42:
                row={'input_id_chunks_json':json.dumps(long_plan.input_ids),
                     'chunk_content_lengths_json':json.dumps(long_plan.content_lengths)}
                target={'text_mode':'full_chunks','steps':30,'generation_seed':202601,'guidance_scale':7.5}
                tools={'torch':torch,'r4b_ids':frozen.r4b_ids,'r4b_condition':frozen.r4b_condition,'r4_finite':finite}
                reference,refmeta=frozen.o_generate_image(tools,generator.unet,generator.text_encoder,
                    generator.vae,generator.scheduler,row,target,generator.device)
                error=np.abs(np.asarray(image,dtype=np.int16)-np.asarray(reference,dtype=np.int16))
                status['reference_sampler_comparison']={'maximum_pixel_difference_0_255':int(error.max()),
                    'allowed_maximum_pixel_difference_0_255':1,'same_runtime_and_components':True,
                    'initial_noise_matches':refmeta['initial_cpu_noise_sha256']==metadata['initial_cpu_noise_sha256']}
                keep('frozen_reference_seed_42',reference,refmeta)
                if error.max()>1 or refmeta['initial_cpu_noise_sha256']!=metadata['initial_cpu_noise_sha256']:
                    raise ValueError('Packaged sampler differs from the frozen study sampler.')
                generator.unet.disable_adapters()
                try:
                    plain,plainmeta=generator.generate(long_report,generation_seed=202601,progress=True)
                finally:
                    generator.unet.enable_adapters()
                keep('base_without_adapter',plain,plainmeta)
                status['adapter_has_image_effect']=bool(np.any(np.asarray(image)!=np.asarray(plain)))
                if not status['adapter_has_image_effect']:raise ValueError('Adapter produced no visible numerical effect.')
            save()
        if len(set(noise_hashes))!=1:raise ValueError('Initial noise was not shared across model checkpoints.')
        generator.set_training_seed(42)
        short,shortmeta=generator.generate(short_report,generation_seed=202601,progress=True)
        keep('short_seed_42',short,shortmeta)
        status.update(status='PASSED',paired_model_noise_verified=True,output_images=len(outputs),
            outputs=outputs,runtime=generator.runtime(),long_report_windows=long_plan.windows,
            short_report_windows=1,models_tested=3)
        print('Software acceptance checks passed. This is not a clinical evaluation.',flush=True)
    except Exception as error:
        status.update(status='FAILED',error_type=type(error).__name__,error=str(error))
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('Validation failed; status and traceback were saved.',flush=True)
    finally:
        status['elapsed_seconds']=time.monotonic()-started
        save()
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True,type=Path)
    p.add_argument('--examples-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--base-model-dir',type=Path)
    a=p.parse_args()
    result=run_validation(a.model_dir,a.examples_dir,a.output_dir,a.base_model_dir)
    raise SystemExit(0 if result['status']=='PASSED' else 1)


if __name__=='__main__':main()
