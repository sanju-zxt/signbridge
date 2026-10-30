import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")
ch28_arm = bpy.data.objects['Armature']

# Import Standing Greeting
bpy.ops.import_scene.fbx(filepath=r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx")
anim_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o != ch28_arm][0]
greeting_action = anim_arm.animation_data.action

# Assign to ch28_arm
ch28_arm.animation_data_create()
ch28_arm.animation_data.action = greeting_action

# Evaluate at frame 1 and frame 60
bpy.context.scene.frame_set(1)
ch28_r_arm = ch28_arm.pose.bones['mixamorig10:RightArm']
print("Frame 1 ch28 RightArm quat:", tuple(round(v, 4) for v in ch28_r_arm.rotation_quaternion))

bpy.context.scene.frame_set(60)
print("Frame 60 ch28 RightArm quat:", tuple(round(v, 4) for v in ch28_r_arm.rotation_quaternion))

# Check mesh deformation (e.g. Ch28_Body)
body = bpy.data.objects['Ch28_Body']
depsgraph = bpy.context.evaluated_depsgraph_get()
eval_body = body.evaluated_get(depsgraph)
eval_mesh = eval_body.to_mesh()
print("Evaluated body mesh vertices:", len(eval_mesh.vertices))
print("Sample vertex 100 coord at frame 60:", eval_mesh.vertices[100].co)
eval_body.to_mesh_clear()
print("ANIMATION EVALUATION SUCCESSFUL!")
