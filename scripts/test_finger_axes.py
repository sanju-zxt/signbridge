import bpy
import json
import mathutils
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")
arm = bpy.data.objects['Armature']

with open(r"C:\Users\drx20\.gemini\antigravity\scratch\idle_pose.json", 'r') as f:
    idle_pose = json.load(f)

for pb_name in ['mixamorig10:RightHandIndex1', 'mixamorig10:RightHandThumb1']:
    pb = arm.pose.bones[pb_name]
    q_idle = mathutils.Quaternion(idle_pose[pb_name]['rotation_quaternion'])
    print(f"{pb_name} idle quat: {q_idle}")
    print(f"{pb_name} idle euler deg: {[round(math.degrees(a), 1) for a in q_idle.to_euler('XYZ')]}")
