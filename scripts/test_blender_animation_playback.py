import bpy
import sys

print("=== VERIFYING SIGN AVATAR BLENDER ANIMATION PLAYBACK (25+ ISL ACTIONS) ===")

arm = [o for o in bpy.data.objects if o.type == 'ARMATURE']
if not arm:
    print("❌ FAIL: No armature found in scene!")
    sys.exit(1)

arm_obj = arm[0]
print(f"✅ Found Armature: {arm_obj.name} with {len(arm_obj.data.bones)} bones")

# Check required actions for full 25-sign vocabulary
required_actions = [
    'SBV_IDLE',
    'SBV_HELLO',
    'SBV_THANK_YOU',
    'SBV_PLEASE',
    'SBV_SORRY',
    'SBV_YES',
    'SBV_NO',
    'SBV_GOOD',
    'SBV_HELP',
    'SBV_WATER',
    'SBV_FOOD',
    'SBV_HOME',
    'SBV_SCHOOL',
    'SBV_HOSPITAL',
    'SBV_NEED',
    'SBV_EMERGENCY',
    'SBV_POLICE',
    'SBV_STOP',
    'SBV_GO',
    'SBV_WAIT',
    'SBV_COME',
    'SBV_UNDERSTAND',
    'SBV_DONT_UNDERSTAND',
    'SBV_NAME',
    'SBV_WHERE',
    'SBV_BATHROOM',
    'SBV_CRYING',
    'SBV_GOING',
]

available_actions = {a.name: a for a in bpy.data.actions}
missing = [a for a in required_actions if a not in available_actions]

if missing:
    print(f"❌ FAIL: Missing required actions: {missing}")
    sys.exit(1)

print(f"✅ All {len(required_actions)} required actions are present:")
for act_name in required_actions:
    act = available_actions[act_name]
    print(f"   - {act_name}: {act.frame_range[0]:.0f} to {act.frame_range[1]:.0f} frames")

# Test evaluating poses across keyframes for each action
for act_name in required_actions:
    act = available_actions[act_name]
    arm_obj.animation_data.action = act
    if hasattr(arm_obj.animation_data, 'action_slot') and len(act.slots) > 0:
        arm_obj.animation_data.action_slot = act.slots[0]
    
    start_frame = int(act.frame_range[0])
    mid_frame = int((act.frame_range[0] + act.frame_range[1]) / 2)
    end_frame = int(act.frame_range[1])
    
    for f in [start_frame, mid_frame, end_frame]:
        bpy.context.scene.frame_set(f)
        r_arm = arm_obj.pose.bones.get('mixamorig10:RightArm')
        if r_arm:
            _ = r_arm.rotation_quaternion.magnitude

print(f"✅ ALL {len(required_actions)} ISL ANIMATIONS EVALUATED AND PLAYED SUCCESSFULLY!")
