import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")

arm = bpy.data.objects['Armature']
print(f"Armature object: {arm.name}")
print(f"Pose bones count: {len(arm.pose.bones)}")

# Check rotation modes of key bones
key_bones = [
    'mixamorig10:Hips', 'mixamorig10:Spine', 'mixamorig10:Spine1', 'mixamorig10:Spine2',
    'mixamorig10:Neck', 'mixamorig10:Head',
    'mixamorig10:RightShoulder', 'mixamorig10:RightArm', 'mixamorig10:RightForeArm', 'mixamorig10:RightHand',
    'mixamorig10:LeftShoulder', 'mixamorig10:LeftArm', 'mixamorig10:LeftForeArm', 'mixamorig10:LeftHand',
    'mixamorig10:RightHandIndex1', 'mixamorig10:RightHandThumb1'
]

for b_name in key_bones:
    pb = arm.pose.bones.get(b_name)
    if pb:
        print(f"{b_name}: rot_mode={pb.rotation_mode}, quat={tuple(round(v, 3) for v in pb.rotation_quaternion)}, euler={tuple(round(v, 3) for v in pb.rotation_euler)}")
    else:
        print(f"{b_name}: NOT FOUND")
