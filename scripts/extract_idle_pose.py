import bpy
import json

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx")
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]

bpy.context.scene.frame_set(1)

pose_dict = {}
for pb in arm.pose.bones:
    pose_dict[pb.name] = {
        'location': [round(v, 6) for v in pb.location],
        'rotation_quaternion': [round(v, 6) for v in pb.rotation_quaternion],
    }

out_path = r"C:\Users\drx20\.gemini\antigravity\scratch\idle_pose.json"
with open(out_path, 'w') as f:
    json.dump(pose_dict, f, indent=2)

print(f"Extracted resting pose for {len(pose_dict)} bones to {out_path}")
