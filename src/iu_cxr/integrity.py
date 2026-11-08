"""Verify released adapters without importing torch or unpickling anything."""
from array import array
from pathlib import Path
import hashlib
import json
import math
import os
import struct
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative, allow_cache_symlinks=False):
    root = Path(root).resolve()
    lexical = Path(os.path.abspath(root / relative))
    if lexical == root or not lexical.is_relative_to(root):
        raise ValueError('File path must be below the selected model directory.')
    path = lexical.resolve()
    if not allow_cache_symlinks and not path.is_relative_to(root):
        raise ValueError('File path must be below the selected model directory.')
    # Hub snapshots point to shared cache blobs. Callers must authenticate every
    # allowed target against the bundled, trusted SHA-256 registry before use.
    if allow_cache_symlinks:
        return lexical
    return path


def read_registry():
    path = Path(__file__).parent / 'data/model_registry.json'
    return json.loads(path.read_text(encoding='utf-8'))


def inspect_safetensors(path, expected_sha256=None, expected_shapes=None,
                        expected_parameters=6377472, check_finite=True):
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError('Adapter SHA-256 differs from the paper checkpoint.')
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Incomplete safetensors prefix.')
        header_length = struct.unpack('<Q', prefix)[0]
        if header_length < 2 or header_length > 16*1024*1024:
            raise ValueError('Invalid safetensors header length.')
        header = json.loads(stream.read(header_length))
        tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        if not tensors:
            raise ValueError('Adapter contains no tensors.')
        shapes = {k:v['shape'] for k,v in tensors.items()}
        if expected_shapes is not None and shapes != expected_shapes:
            raise ValueError('Adapter tensor names or shapes differ from the frozen model.')
        cursor, count = 0, 0
        for name, value in sorted(tensors.items(), key=lambda x:x[1]['data_offsets'][0]):
            shape = value['shape']
            if value['dtype'] != 'F32' or len(shape) != 2 or any(not isinstance(v,int) or v<1 for v in shape):
                raise ValueError('Expected nonempty two-dimensional FP32 LoRA tensors.')
            if '.lora_A.weight' in name:
                if not name.endswith('.lora_A.weight') or shape[0] != 32:
                    raise ValueError('Invalid LoRA A rank.')
                partner = tensors.get(name.replace('.lora_A.weight','.lora_B.weight'))
                if partner is None or partner['shape'][1] != 32:
                    raise ValueError('Missing paired rank-32 LoRA B tensor.')
            elif name.endswith('.lora_B.weight'):
                if shape[1] != 32 or name.replace('.lora_B.weight','.lora_A.weight') not in tensors:
                    raise ValueError('Missing paired rank-32 LoRA A tensor.')
            else:
                raise ValueError('Non-LoRA tensor in adapter.')
            n = math.prod(shape)
            begin,end = value['data_offsets']
            if begin != cursor or end-begin != n*4:
                raise ValueError('Noncontiguous or inconsistent tensor storage.')
            cursor=end
            count+=n
        if count != expected_parameters or path.stat().st_size != 8+header_length+cursor:
            raise ValueError('Parameter count or file length is inconsistent.')
        if check_finite:
            values=array('f')
            values.fromfile(stream,count)
            if sys.byteorder != 'little':
                values.byteswap()
            if not all(math.isfinite(x) for x in values):
                raise ValueError('Nonfinite adapter value.')
    return {'sha256':sha256_file(path),'parameters':count,'tensor_count':len(tensors),
            'dtype':'float32','finite_values_checked':check_finite}


def verify_model(model_dir, training_seed=42, check_finite=True):
    registry=read_registry()
    key=str(training_seed)
    if key not in registry['models']:
        raise ValueError('Supported training seeds are 42, 1337, and 2026.')
    record=registry['models'][key]
    for relative, expected in record['files_sha256'].items():
        if sha256_file(checked_path(model_dir,relative,True)) != expected:
            raise ValueError('A checkpoint component does not match the paper: '+relative)
    config=json.loads(checked_path(model_dir,record['config'],True).read_text())
    targets={'to_q','to_k','to_v','to_out.0'}
    if config['r']!=32 or config['lora_alpha']!=32 or config['lora_dropout']!=0 or set(config['target_modules'])!=targets:
        raise ValueError('Unexpected LoRA configuration.')
    if config.get('use_dora') or config.get('use_rslora') or config.get('bias')!='none':
        raise ValueError('Unsupported adapter modification.')
    shapes=json.loads((Path(__file__).parent/'data/expected_tensor_shapes.json').read_text())
    result=inspect_safetensors(checked_path(model_dir,record['weights'],True),
        record['files_sha256'][record['weights']],shapes,check_finite=check_finite)
    state=json.loads(checked_path(model_dir,record['state'],True).read_text())
    if state['arm']!='unet_full' or state['seed']!=int(training_seed) or state['dual'] or state['global_step']!=3140:
        raise ValueError('Wrong training arm, seed, or final step.')
    return dict(result,training_seed=int(training_seed),global_step=3140,training_data='IU X-Ray only')

