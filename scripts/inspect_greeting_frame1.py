import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx")
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]

bpy.context.scene.frame_set(1)

for b_name in ['mixamorig10:RightArm', 'mixamorig10:LeftArm', 'mixamorig10:RightForeArm', 'mixamorig10:LeftForeArm', 'mixamorig10:RightHand', 'mixamorig10:LeftHand']:
    pb = arm.pose.bones.get(b_name)
    if pb:
        # Convert quat to euler in degrees
        euler = pb.rotation_quaternion.to_euler('XYZ')
        deg = tuple(round(math.degrees(a), 1) for a in euler)
        print(f"Frame 1 {b_name}: quat={tuple(round(v, 4) for v in pb.rotation_quaternion)}, euler_deg={deg}")
