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

def quat_to_euler_deg(x, y, z, w):
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

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

for anim_name in ['SBV_IDLE', 'SBV_HELLO', 'SBV_HELP', 'SBV_NAME']:
    anim = next((a for a in gltf['animations'] if a['name'] == anim_name), None)
    if not anim:
        print(f'Animation {anim_name} not found')
        continue
    print(f'\n================== ANIMATION: {anim_name} ==================')
    samplers = anim['samplers']
    for ch in anim['channels']:
        target = ch['target']
        node_idx = target['node']
        prop = target['path']
        node_name = nodes[node_idx]['name']
        if prop == 'rotation' and any(b in node_name for b in ['LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']) and not any(f in node_name for f in ['Index', 'Thumb', 'Middle', 'Ring', 'Pinky']):
            sampler = samplers[ch['sampler']]
            times = get_accessor_data(sampler['input'])
            vals = get_accessor_data(sampler['output'])
            q0 = vals[0]
            e0 = quat_to_euler_deg(q0[0], q0[1], q0[2], q0[3])
            print(f'{node_name:30s} {prop:10s} frames={len(times):3d} dur={times[-1]:.2f}s')
            print(f'   t=0.0s: quat=[{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}] Euler(deg)=({e0[0]:.1f}, {e0[1]:.1f}, {e0[2]:.1f})')
            if len(times) > 1:
                mid = len(times) // 2
                qm = vals[mid]
                em = quat_to_euler_deg(qm[0], qm[1], qm[2], qm[3])
                print(f'   t={times[mid]:.2f}s: quat=[{qm[0]:.4f}, {qm[1]:.4f}, {qm[2]:.4f}, {qm[3]:.4f}] Euler(deg)=({em[0]:.1f}, {em[1]:.1f}, {em[2]:.1f})')
