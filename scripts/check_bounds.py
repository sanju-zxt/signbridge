import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\signbridge2vision\apps\web\public\models\sign-avatar\sign-avatar.glb")

meshes = [o for o in bpy.data.objects if o.type == 'MESH']
min_z = min(min((o.matrix_world @ v.co).z for v in o.data.vertices) for o in meshes)
max_z = max(max((o.matrix_world @ v.co).z for v in o.data.vertices) for o in meshes)
min_y = min(min((o.matrix_world @ v.co).y for v in o.data.vertices) for o in meshes)
max_y = max(max((o.matrix_world @ v.co).y for v in o.data.vertices) for o in meshes)
min_x = min(min((o.matrix_world @ v.co).x for v in o.data.vertices) for o in meshes)
max_x = max(max((o.matrix_world @ v.co).x for v in o.data.vertices) for o in meshes)

print(f"Bounding Box X: {min_x:.3f} to {max_x:.3f} (width: {max_x - min_x:.3f})")
print(f"Bounding Box Y: {min_y:.3f} to {max_y:.3f} (depth: {max_y - min_y:.3f})")
print(f"Bounding Box Z: {min_z:.3f} to {max_z:.3f} (height: {max_z - min_z:.3f})")
