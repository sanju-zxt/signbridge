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

for anim_name in ['SBV_FOOD', 'SBV_WATER', 'SBV_PLEASE', 'SBV_THANK_YOU']:
    anim = next((a for a in gltf['animations'] if a['name'] == anim_name), None)
    if not anim:
        continue
    print(f'\n=== {anim_name} ===')
    samplers = anim['samplers']
    for ch in anim['channels']:
        target = ch['target']
        node_name = nodes[target['node']]['name']
        prop = target['path']
        if prop == 'rotation' and any(b in node_name for b in ['RightArm', 'RightForeArm', 'RightHand']) and not any(f in node_name for f in ['Index', 'Thumb', 'Middle', 'Ring', 'Pinky']):
            sampler = samplers[ch['sampler']]
            times = get_accessor_data(sampler['input'])
            vals = get_accessor_data(sampler['output'])
            mid = len(times) // 2
            qm = vals[mid]
            print(f'  {node_name:25s} mid (t={times[mid]:.2f}s): [{qm[0]:.4f}, {qm[1]:.4f}, {qm[2]:.4f}, {qm[3]:.4f}]')
