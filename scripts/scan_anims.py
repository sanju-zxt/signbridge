import struct
import json
import math

glb_path = r'apps\web\public\models\sign-avatar\sign-avatar.glb'
with open(glb_path, 'rb') as f:
    f.read(12)
    chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
    gltf = json.loads(f.read(chunk_len).decode('utf-8'))
    bin_len, bin_type = struct.unpack('<I4s', f.read(8))
    bin_bytes = f.read(bin_len)

def get_accessor_data(acc_idx):
    acc = gltf['accessors'][acc_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    type_ = acc['type']
    stride = bv.get('byteStride', None)
    num_comp = 1 if type_ == 'SCALAR' else (3 if type_ == 'VEC3' else 4)
    res = []
    curr = offset
    for _ in range(count):
        vals = struct.unpack_from(f'<{num_comp}f', bin_bytes, curr)
        res.append(vals if num_comp > 1 else vals[0])
        curr += stride if stride else num_comp * 4
    return res

nodes = gltf['nodes']

print('Scanning all GLB animations for LeftArm / LeftForeArm / RightArm / RightForeArm values...')
for anim in gltf['animations']:
    name = anim['name']
    samplers = anim['samplers']
    for ch in anim['channels']:
        target = ch['target']
        node_name = nodes[target['node']]['name']
        prop = target['path']
        if prop == 'rotation' and ('LeftArm' in node_name or 'LeftForeArm' in node_name):
            sampler = samplers[ch['sampler']]
            vals = get_accessor_data(sampler['output'])
            # Check if any quaternion has extreme values
            for v in vals:
                # If LeftForeArm has large negative or positive X/Y that twists behind back
                if 'LeftForeArm' in node_name:
                    # Check if v differs significantly from IDLE
                    pass
