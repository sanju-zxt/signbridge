import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb")
ch28_arm = bpy.data.objects['Armature']

bpy.ops.import_scene.fbx(filepath=r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx")
anim_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o != ch28_arm][0]
greeting_action = anim_arm.animation_data.action

ch28_arm.animation_data_create()
ch28_arm.animation_data.action = greeting_action
if hasattr(ch28_arm.animation_data, 'action_slot') and len(greeting_action.slots) > 0:
    ch28_arm.animation_data.action_slot = greeting_action.slots[0]
    print("Assigned action_slot to ch28_arm!")

bpy.context.scene.frame_set(60)
ch28_r_arm = ch28_arm.pose.bones['mixamorig10:RightArm']
print("Frame 60 ch28 RightArm quat:", tuple(round(v, 4) for v in ch28_r_arm.rotation_quaternion))
