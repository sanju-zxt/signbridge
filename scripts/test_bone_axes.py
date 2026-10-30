import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")

arm = bpy.data.objects['Armature']

for name in ['mixamorig10:RightArm', 'mixamorig10:LeftArm', 'mixamorig10:RightForeArm', 'mixamorig10:LeftForeArm']:
    bone = arm.data.bones.get(name)
    pbone = arm.pose.bones.get(name)
    print(f"=== {name} ===")
    print(f"  head: {bone.head}, tail: {bone.tail}, length: {bone.length:.3f}")
    print(f"  matrix_local:\n{bone.matrix_local}")
