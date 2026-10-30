import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")

arm = bpy.data.objects['Armature']

# Create a test action
test_act = bpy.data.actions.new(name="TEST_KEYFRAME_ACTION")
arm.animation_data_create()
arm.animation_data.action = test_act

# Switch to euler for easy angle manipulation
for pb in arm.pose.bones:
    pb.rotation_mode = 'XYZ'

# Test keyframing RightArm at frame 1 and frame 30
r_arm = arm.pose.bones['mixamorig10:RightArm']
r_arm.rotation_euler = (0, 0, 0)
r_arm.keyframe_insert(data_path='rotation_euler', frame=1)

r_arm.rotation_euler = (math.radians(-45), math.radians(20), math.radians(10))
r_arm.keyframe_insert(data_path='rotation_euler', frame=30)

r_arm.rotation_euler = (0, 0, 0)
r_arm.keyframe_insert(data_path='rotation_euler', frame=60)

print(f"Created action: {test_act.name}, frame_range: {test_act.frame_range}")
