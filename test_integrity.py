from pathlib import Path
import json,struct,tempfile,unittest
from iu_cxr.integrity import inspect_safetensors,checked_path


def fixture(path,nonfinite=False,bad_offsets=False):
    arrays={'layer.lora_A.weight':[32,2],'layer.lora_B.weight':[2,32]}
    header={};offset=0
    for name,shape in arrays.items():
        size=shape[0]*shape[1]*4
        header[name]={'dtype':'F32','shape':shape,'data_offsets':[offset,offset+size]}
        offset+=size
    if bad_offsets:header['layer.lora_B.weight']['data_offsets'][0]-=4
    text=json.dumps(header).encode();text+=b' ' * ((-len(text))%8)
    body=struct.pack('<128f',*([float('nan')]+[.25]*127 if nonfinite else [.25]*128))
    path.write_bytes(struct.pack('<Q',len(text))+text+body)


class IntegrityTests(unittest.TestCase):
    def test_safe_tensor_header_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.safetensors';fixture(p)
            result=inspect_safetensors(p,expected_parameters=128)
            self.assertEqual(result['parameters'],128)
            self.assertEqual(result['tensor_count'],2)

    def test_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.safetensors';fixture(p,nonfinite=True)
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                inspect_safetensors(p,expected_parameters=128)

    def test_changed_weight_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.safetensors';fixture(p)
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                inspect_safetensors(p,expected_sha256='0'*64,expected_parameters=128)

    def test_overlapping_storage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.safetensors';fixture(p,bad_offsets=True)
            with self.assertRaises(ValueError):inspect_safetensors(p,expected_parameters=128)

    def test_wrong_parameter_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.safetensors';fixture(p)
            with self.assertRaises(ValueError):inspect_safetensors(p)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):checked_path(tmp,'../other')

    def test_hub_cache_symlink_is_explicit_and_still_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'snapshot';root.mkdir()
            blob=Path(tmp)/'blob.safetensors';fixture(blob)
            (root/'adapter.safetensors').symlink_to(blob)
            with self.assertRaises(ValueError):checked_path(root,'adapter.safetensors')
            allowed=checked_path(root,'adapter.safetensors',True)
            self.assertEqual(allowed.read_bytes(),blob.read_bytes())
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                inspect_safetensors(allowed,expected_sha256='0'*64,expected_parameters=128)
            with self.assertRaises(ValueError):checked_path(root,'../blob.safetensors',True)


if __name__=='__main__':unittest.main()

