import struct, json

glb_path = r'C:\Users\drx20\.gemini\antigravity\scratch\signbridge2vision\apps\web\public\models\sign-avatar\sign-avatar.glb'
with open(glb_path, 'rb') as f:
    magic, version, length = struct.unpack('<4sII', f.read(12))
    chunk_len, chunk_type = struct.unpack('<I4s', f.read(8))
    json_bytes = f.read(chunk_len)
    gltf = json.loads(json_bytes.decode('utf-8'))

animations = gltf.get('animations', [])
print(f'Total Animations in GLB: {len(animations)}')
for i, anim in enumerate(animations):
    name = anim.get('name', 'unnamed')
    channels = len(anim.get('channels', []))
    print(f'  {i:2d}: {name:20s} (channels: {channels})')

materials = gltf.get('materials', [])
print(f'\nTotal Materials in GLB: {len(materials)}')
for mat in materials:
    name = mat.get('name')
    pbr = mat.get('pbrMetallicRoughness', {})
    bc = pbr.get('baseColorFactor')
    rf = pbr.get('roughnessFactor')
    mf = pbr.get('metallicFactor')
    print(f'  {name:20s} baseColor={bc} roughness={rf} metallic={mf}')
