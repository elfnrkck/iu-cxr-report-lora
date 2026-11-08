"""Verbatim study functions; extracted for software comparison, not new experiments."""
import json,re,hashlib

def r4_token_plan(prompt, tokenizer):
    full = tokenizer.encode(prompt, add_special_tokens=False, truncation=False, verbose=False)
    sentences = [tokenizer.encode(s, add_special_tokens=False, truncation=False, verbose=False)
                 for s in re.split(r'(?<=[.!?])\s+', prompt) if s.strip()]
    if [v for sentence in sentences for v in sentence] != full:
        raise ValueError('Cümle paketlemesi tam rapor tokenlarını değiştirdi.')
    chunks, current, forced = [], [], 0
    for sentence in sentences:
        if len(sentence) > 75:
            if current:
                chunks.append(current)
                current = []
            forced += 1
            chunks.extend(sentence[i:i+75] for i in range(0, len(sentence), 75))
        elif len(current) + len(sentence) <= 75:
            current.extend(sentence)
        else:
            if current:
                chunks.append(current)
            current = list(sentence)
    if current:
        chunks.append(current)
    if not chunks:
        chunks = [[]]
    ids = [[tokenizer.bos_token_id] + c + [tokenizer.eos_token_id] * (76-len(c)) for c in chunks]
    recovered = [v for chunk, content in zip(ids, chunks) for v in chunk[1:1+len(content)]]
    if recovered != full:
        raise ValueError('Yeniden hazırlıkta token kaybı.')
    return {'input_id_chunks_json': json.dumps(ids), 'chunk_content_lengths_json': json.dumps([len(c) for c in chunks]),
            'chunk_count': str(len(chunks)), 'content_token_count': str(len(full)),
            'forced_long_sentence_splits': str(forced), 'single_window_would_truncate': str(len(full)>75)}

def r4b_ids(row, mode, bos=49406, eos=49407):
    chunks = json.loads(row['input_id_chunks_json'])
    lengths = json.loads(row['chunk_content_lengths_json'])
    if mode == 'full_chunks':
        return chunks
    if mode != 'short75':
        raise ValueError('Unknown text mode.')
    # Ordinary full-prompt truncation, not the possibly shorter first sentence-packed window.
    content = [v for chunk, n in zip(chunks, lengths) for v in chunk[1:1+n]][:75]
    return [[bos] + content + [eos] * (76-len(content))]

def r4b_condition(text, ids, trainable):
    # Batch contains only a single K. No artificial windows and no missing-window masks.
    if ids.ndim != 3 or ids.shape[-1] != 77:
        raise ValueError('Expected [B,K,77] CLIP input.')
    b, k, _ = ids.shape
    with nullcontext() if trainable else torch.no_grad():
        hidden = text(input_ids=ids.reshape(b*k, 77), return_dict=True).last_hidden_state
        hidden = hidden.reshape(b, k*77, hidden.shape[-1])
    r4_finite(hidden, 'batch_condition')
    return hidden

def o_generate_image(T,unet,text,vae,ddim,row,target,device):
    """Identical conditioning, CPU noise, CFG, DDIM and FP32 decode math to r4b_previews."""
    torch=T['torch'];chunks=T['r4b_ids'](row,target['text_mode'])
    ddim.set_timesteps(int(target['steps']),device=device)
    with torch.no_grad():
        with torch.autocast(device.type,dtype=torch.bfloat16):
            conditional=T['r4b_condition'](text,torch.tensor([chunks],device=device),False)
            unconditional=T['r4b_condition'](text,torch.tensor([[[49406]+[49407]*76 for _ in chunks]],device=device),False)
        embeddings=torch.cat([unconditional,conditional])
        rng=torch.Generator(device='cpu').manual_seed(int(target['generation_seed']))
        initial=torch.randn((1,4,64,64),generator=rng)
        noise_sha256=hashlib.sha256(initial.contiguous().numpy().tobytes()).hexdigest()
        latent=initial.to(device,dtype=torch.bfloat16)*ddim.init_noise_sigma
        for timestep in ddim.timesteps:
            model_input=ddim.scale_model_input(torch.cat([latent,latent]),timestep)
            with torch.autocast(device.type,dtype=torch.bfloat16):
                predicted=unet(model_input,timestep,encoder_hidden_states=embeddings).sample
            noise_u,noise_c=predicted.chunk(2)
            guided=noise_u+float(target['guidance_scale'])*(noise_c-noise_u)
            latent=ddim.step(guided,timestep,latent,eta=0.0).prev_sample
        decoded=vae.decode(latent.float()/vae.config.scaling_factor).sample
        T['r4_finite'](decoded,'generated_pixels')
        array=decoded[0].float().add(1).div(2).clamp(0,1).permute(1,2,0).cpu().numpy()
        from PIL import Image
        image=Image.fromarray(np.rint(array*255).astype('uint8'))
    return image,{'actual_conditioning_chunks':len(chunks),'actual_conditioning_ids_sha256':hashlib.sha256(json.dumps(chunks,sort_keys=True).encode()).hexdigest(),
        'initial_cpu_noise_sha256':noise_sha256,'initial_noise_shared_across_cases_and_models_for_each_generation_seed':True,
        'pixel_std_0_1':float(array.std()),'low_variance_review_flag':bool(array.std()<0.01),
        'technical_flags_do_not_exclude_or_select_images':True,'clinical_accuracy_evaluated':False}
