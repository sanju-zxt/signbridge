import bpy
import json
import mathutils
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")
arm = bpy.data.objects['Armature']

with open(r"C:\Users\drx20\.gemini\antigravity\scratch\idle_pose.json", 'r') as f:
    idle_pose = json.load(f)

# Apply idle pose
for pb in arm.pose.bones:
    if pb.name in idle_pose:
        pb.rotation_quaternion = idle_pose[pb.name]['rotation_quaternion']

r_arm = arm.pose.bones['mixamorig10:RightArm']
q_idle = mathutils.Quaternion(idle_pose['mixamorig10:RightArm']['rotation_quaternion'])
print("Base idle RightArm quat:", q_idle)

# Test rotating RightArm up by 60 degrees around local X axis
q_rot = mathutils.Euler((math.radians(60), 0, 0), 'XYZ').to_quaternion()
r_arm.rotation_quaternion = q_idle @ q_rot
print("Rotated RightArm quat:", r_arm.rotation_quaternion)
