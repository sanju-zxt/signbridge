import bpy
import os

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")

arm = bpy.data.objects['Armature']
print("Initial actions:", [a.name for a in bpy.data.actions])

# Import Standing Greeting FBX
bpy.ops.import_scene.fbx(filepath=r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx")
print("Actions after FBX:", [a.name for a in bpy.data.actions])

# Find imported anim armature
anim_arms = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o != arm]
print("Anim armatures:", [o.name for o in anim_arms])

# Check gltf export settings
out_test = r"C:\Users\drx20\.gemini\antigravity\scratch\test_multi_export.glb"
# Delete imported extra armatures/meshes to leave only ch28
for o in list(bpy.data.objects):
    if o != arm and o.parent != arm and o.name != 'Armature':
        if 'Ch28' not in o.name:
            bpy.data.objects.remove(o, do_unlink=True)

print("Remaining objects:", [o.name for o in bpy.data.objects])

# Create NLA tracks on arm for each action
if not arm.animation_data:
    arm.animation_data_create()

for act in bpy.data.actions:
    act.use_fake_user = True
    track = arm.animation_data.nla_tracks.new()
    track.name = act.name
    strip = track.strips.new(act.name, int(act.frame_range[0]), act)
    print(f"Added NLA strip for {act.name}: {strip.name}")

bpy.ops.export_scene.gltf(
    filepath=out_test,
    export_format='GLB',
    export_animations=True,
    export_nla_strips=True
)
print("Export finished! File size:", os.path.getsize(out_test))

# Verify imported GLB
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=out_test)
print("Actions in re-imported GLB:", [a.name for a in bpy.data.actions])
