import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx")
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
act = arm.animation_data.action

print(f"Action: {act.name}")
# In Blender 5.2, check fcurves or channels
if hasattr(act, 'fcurves'):
    print(f"Fcurves count: {len(act.fcurves)}")
    for fc in act.fcurves[:10]:
        print(f"  fc data_path: {fc.data_path}, index: {fc.array_index}")
else:
    print("Action has no fcurves attribute, checking layers/strips/bindings:")
    for attr in dir(act):
        if not attr.startswith('__'):
            print(f"  attr: {attr}")
