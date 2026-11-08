from pathlib import Path
import argparse,json
from .integrity import verify_model,sha256_file


def main():
    parser=argparse.ArgumentParser(description='Generate a synthetic frontal chest radiograph from an English report.')
    parser.add_argument('--model-dir',required=True,type=Path,help='Local model repository containing seed_42, seed_1337 and seed_2026.')
    parser.add_argument('--training-seed',type=int,choices=[42,1337,2026],default=42)
    parser.add_argument('--verify-only',action='store_true',help='Verify checkpoint bytes and values without torch or a GPU.')
    parser.add_argument('--report-file',type=Path,help='UTF-8 English report text file.')
    parser.add_argument('--output',type=Path,default=Path('outputs/synthetic.png'))
    parser.add_argument('--generation-seed',type=int,default=202601)
    parser.add_argument('--base-model-dir',type=Path,help='Optional local copy of the exact pinned SD1.5 snapshot.')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--precision',choices=['bfloat16','float16','float32'],default='bfloat16')
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--guidance-scale',type=float,default=7.5)
    parser.add_argument('--max-windows',type=int,default=16)
    args=parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_model(args.model_dir,args.training_seed),indent=2));return
    if args.report_file is None:
        parser.error('--report-file is required for generation.')
    sidecar=args.output.with_suffix('.json')
    if args.output.exists() or sidecar.exists():
        parser.error('Output already exists. Choose a new output name.')
    from .pipeline import IUReportGenerator
    generator=IUReportGenerator(args.model_dir,args.training_seed,args.base_model_dir,
        device=args.device,precision=args.precision,max_windows=args.max_windows)
    image,metadata=generator.generate(args.report_file.read_text(encoding='utf-8-sig'),
        generation_seed=args.generation_seed,steps=args.steps,guidance_scale=args.guidance_scale,progress=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    image.save(args.output,format='PNG')
    metadata['png_sha256']=sha256_file(args.output)
    sidecar.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('Saved',args.output,'and',sidecar)


if __name__=='__main__':main()
