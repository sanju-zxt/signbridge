import bpy
import json
import math
import mathutils
import os

print("=== STARTING 25-SIGN ISL AVATAR BUILD ===")

# 1. Clean slate
bpy.ops.wm.read_factory_settings(use_empty=True)

# 2. Import base Ch28 human character
base_glb = r"C:\Users\drx20\.gemini\antigravity\scratch\test_ch28_1024.glb"
bpy.ops.import_scene.gltf(filepath=base_glb)

ch28_arm = bpy.data.objects['Armature']
print(f"Base armature loaded: {ch28_arm.name}, bones: {len(ch28_arm.data.bones)}")

# Load idle resting pose
with open(r"C:\Users\drx20\.gemini\antigravity\scratch\idle_pose.json", 'r') as f:
    idle_pose = json.load(f)

# Helper to apply base idle pose to a frame
def apply_idle_pose(arm_obj):
    for pb in arm_obj.pose.bones:
        if pb.name in idle_pose:
            data = idle_pose[pb.name]
            pb.location = mathutils.Vector(data['location'])
            pb.rotation_quaternion = mathutils.Quaternion(data['rotation_quaternion'])

# Helper to insert keyframe for all bones in arm
def keyframe_all_bones(arm_obj, frame):
    for pb in arm_obj.pose.bones:
        pb.keyframe_insert(data_path='location', frame=frame)
        pb.keyframe_insert(data_path='rotation_quaternion', frame=frame)

# Helper to rotate a bone relative to idle pose
def rotate_bone_rel(arm_obj, bone_name, euler_deg):
    pb = arm_obj.pose.bones.get(bone_name)
    if not pb or bone_name not in idle_pose:
        return
    q_base = mathutils.Quaternion(idle_pose[bone_name]['rotation_quaternion'])
    q_offset = mathutils.Euler((
        math.radians(euler_deg[0]),
        math.radians(euler_deg[1]),
        math.radians(euler_deg[2])
    ), 'XYZ').to_quaternion()
    pb.rotation_quaternion = q_base @ q_offset

# Helper to set finger curl (X axis in degrees for segments 1..3)
def set_finger_curl(arm_obj, hand_prefix, finger_name, angle_deg):
    for seg in [1, 2, 3]:
        b_name = f"mixamorig10:{hand_prefix}Hand{finger_name}{seg}"
        pb = arm_obj.pose.bones.get(b_name)
        if pb:
            q_base = mathutils.Quaternion(idle_pose.get(b_name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
            q_bend = mathutils.Euler((math.radians(angle_deg), 0, 0), 'XYZ').to_quaternion()
            pb.rotation_quaternion = q_base @ q_bend

# Helper to curl entire hand into a fist
def set_fist(arm_obj, hand_prefix='Right', curl_deg=65):
    for f in ['Index', 'Middle', 'Ring', 'Pinky']:
        set_finger_curl(arm_obj, hand_prefix, f, curl_deg)
    # Thumb curled inward
    tb1 = arm_obj.pose.bones.get(f"mixamorig10:{hand_prefix}HandThumb1")
    tb2 = arm_obj.pose.bones.get(f"mixamorig10:{hand_prefix}HandThumb2")
    if tb1:
        q_b = mathutils.Quaternion(idle_pose.get(tb1.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb1.rotation_quaternion = q_b @ mathutils.Euler((math.radians(20), math.radians(25), math.radians(-30)), 'XYZ').to_quaternion()
    if tb2:
        q_b = mathutils.Quaternion(idle_pose.get(tb2.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb2.rotation_quaternion = q_b @ mathutils.Euler((math.radians(35), 0, 0), 'XYZ').to_quaternion()

# Helper to set flat open hand
def set_flat_hand(arm_obj, hand_prefix='Right'):
    for f in ['Index', 'Middle', 'Ring', 'Pinky']:
        set_finger_curl(arm_obj, hand_prefix, f, -5) # straight fingers

def set_w_handshape(arm_obj):
    set_finger_curl(arm_obj, 'Right', 'Index', -5)
    set_finger_curl(arm_obj, 'Right', 'Middle', -5)
    set_finger_curl(arm_obj, 'Right', 'Ring', -5)
    set_finger_curl(arm_obj, 'Right', 'Pinky', 75)
    tb1 = arm_obj.pose.bones.get("mixamorig10:RightHandThumb1")
    if tb1:
        q_b = mathutils.Quaternion(idle_pose.get(tb1.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb1.rotation_quaternion = q_b @ mathutils.Euler((math.radians(35), math.radians(20), math.radians(-25)), 'XYZ').to_quaternion()

def set_pinch_handshape(arm_obj):
    for f in ['Index', 'Middle', 'Ring', 'Pinky']:
        set_finger_curl(arm_obj, 'Right', f, 40)
    tb1 = arm_obj.pose.bones.get("mixamorig10:RightHandThumb1")
    if tb1:
        q_b = mathutils.Quaternion(idle_pose.get(tb1.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb1.rotation_quaternion = q_b @ mathutils.Euler((math.radians(35), math.radians(10), math.radians(-15)), 'XYZ').to_quaternion()

def set_thumbs_up(arm_obj, hand_prefix='Right'):
    for f in ['Index', 'Middle', 'Ring', 'Pinky']:
        set_finger_curl(arm_obj, hand_prefix, f, 70)
    tb1 = arm_obj.pose.bones.get(f"mixamorig10:{hand_prefix}HandThumb1")
    tb2 = arm_obj.pose.bones.get(f"mixamorig10:{hand_prefix}HandThumb2")
    if tb1:
        q_b = mathutils.Quaternion(idle_pose.get(tb1.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb1.rotation_quaternion = q_b @ mathutils.Euler((math.radians(-15), math.radians(10), math.radians(25)), 'XYZ').to_quaternion()
    if tb2:
        q_b = mathutils.Quaternion(idle_pose.get(tb2.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb2.rotation_quaternion = q_b @ mathutils.Euler((math.radians(-20), 0, 0), 'XYZ').to_quaternion()

# Ensure animation data on ch28_arm
if not ch28_arm.animation_data:
    ch28_arm.animation_data_create()

for track in list(ch28_arm.animation_data.nla_tracks):
    ch28_arm.animation_data.nla_tracks.remove(track)

actions_to_export = []

# ------------------------------------------------------------------------------
# 1. SBV_IDLE (Natural standing breathing loop, 60 frames)
# ------------------------------------------------------------------------------
act_idle = bpy.data.actions.new(name="SBV_IDLE")
act_idle.use_fake_user = True
ch28_arm.animation_data.action = act_idle
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
rotate_bone_rel(ch28_arm, 'mixamorig10:Spine1', (-1.5, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (1.0, 0, 0))
keyframe_all_bones(ch28_arm, 30)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 60)
actions_to_export.append(act_idle)

# ------------------------------------------------------------------------------
# 2. SBV_THANK_YOU (Chin touch to forward bow sweep, 90 frames)
# ------------------------------------------------------------------------------
act_ty = bpy.data.actions.new(name="SBV_THANK_YOU")
act_ty.use_fake_user = True
ch28_arm.animation_data.action = act_ty
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Chin touch
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-45, 30, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 20, 45))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (15, -25, 20))
set_flat_hand(ch28_arm, 'Right')
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (5, 0, 0))
keyframe_all_bones(ch28_arm, 25)
keyframe_all_bones(ch28_arm, 35)
# Forward sweep & polite bow
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Spine', (10, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:Spine1', (5, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (8, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-20, 20, -35))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (40, 10, 20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (5, -10, 10))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 60)
keyframe_all_bones(ch28_arm, 70)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_ty)

# ------------------------------------------------------------------------------
# 3. SBV_SORRY (Fist circular rub over chest with bowed head, 90 frames)
# ------------------------------------------------------------------------------
act_sorry = bpy.data.actions.new(name="SBV_SORRY")
act_sorry.use_fake_user = True
ch28_arm.animation_data.action = act_sorry
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (8, -6, 6))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 30, -55))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 15, 35))
set_fist(ch28_arm, 'Right', 60)
keyframe_all_bones(ch28_arm, 25)
# Rub 1
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 28, -52))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 18, 38))
keyframe_all_bones(ch28_arm, 40)
# Rub 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-38, 32, -58))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (95, 12, 32))
keyframe_all_bones(ch28_arm, 55)
# Hold
keyframe_all_bones(ch28_arm, 68)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_sorry)

# ------------------------------------------------------------------------------
# 4. SBV_PLEASE (Flat palm rubs chest gently, 85 frames)
# ------------------------------------------------------------------------------
act_please = bpy.data.actions.new(name="SBV_PLEASE")
act_please.use_fake_user = True
ch28_arm.animation_data.action = act_please
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (6, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 28, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (10, -20, 10))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 20)
# Circle 1
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-30, 25, -45))
keyframe_all_bones(ch28_arm, 38)
# Circle 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-38, 30, -55))
keyframe_all_bones(ch28_arm, 55)
keyframe_all_bones(ch28_arm, 68)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_please)

# ------------------------------------------------------------------------------
# 5. SBV_HELP (Flat left hand support + right fist lifting, 90 frames)
# ------------------------------------------------------------------------------
act_help = bpy.data.actions.new(name="SBV_HELP")
act_help.use_fake_user = True
ch28_arm.animation_data.action = act_help
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Frame 25: base pose
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-25, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (75, -25, -20))
set_flat_hand(ch28_arm, 'Left')
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-25, 20, -25))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 25))
set_fist(ch28_arm, 'Right', 65)
keyframe_all_bones(ch28_arm, 25)
# Frame 55: lifted upward
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-38, -25, 30))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (88, -25, -20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-38, 25, -30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (88, 20, 25))
keyframe_all_bones(ch28_arm, 55)
keyframe_all_bones(ch28_arm, 68)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_help)

# ------------------------------------------------------------------------------
# 6. SBV_WATER (W-hand tapping chin, 85 frames)
# ------------------------------------------------------------------------------
act_water = bpy.data.actions.new(name="SBV_WATER")
act_water.use_fake_user = True
ch28_arm.animation_data.action = act_water
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-50, 32, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (100, 20, 45))
set_w_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 25)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 18, 40))
keyframe_all_bones(ch28_arm, 38)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (100, 20, 45))
keyframe_all_bones(ch28_arm, 50)
keyframe_all_bones(ch28_arm, 65)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_water)

# ------------------------------------------------------------------------------
# 7. SBV_FOOD (Pinch tapping mouth, 85 frames)
# ------------------------------------------------------------------------------
act_food = bpy.data.actions.new(name="SBV_FOOD")
act_food.use_fake_user = True
ch28_arm.animation_data.action = act_food
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-52, 35, -52))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (105, 20, 48))
set_pinch_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 25)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (95, 18, 42))
keyframe_all_bones(ch28_arm, 38)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (105, 20, 48))
keyframe_all_bones(ch28_arm, 50)
keyframe_all_bones(ch28_arm, 65)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_food)

# ------------------------------------------------------------------------------
# 8. SBV_GOOD (Thumbs up thrust forward with approving nod, 80 frames)
# ------------------------------------------------------------------------------
act_good = bpy.data.actions.new(name="SBV_GOOD")
act_good.use_fake_user = True
ch28_arm.animation_data.action = act_good
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Raise thumbs up
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 20, -25))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 15, 20))
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (4, 0, 0))
set_thumbs_up(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 22)
# Thrust forward + approving nod
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-25, 15, -20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (55, 10, 15))
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (8, 0, 0))
keyframe_all_bones(ch28_arm, 40)
# Hold apex
keyframe_all_bones(ch28_arm, 58)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 80)
actions_to_export.append(act_good)

# ------------------------------------------------------------------------------
# 9. SBV_STOP (Right hand chops onto left flat palm barrier, 85 frames)
# ------------------------------------------------------------------------------
act_stop = bpy.data.actions.new(name="SBV_STOP")
act_stop.use_fake_user = True
ch28_arm.animation_data.action = act_stop
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Prepare chop: left palm flat, right hand raised
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-25, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (70, -20, -15))
set_flat_hand(ch28_arm, 'Left')
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-45, 25, -35))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 20, 30))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 25)
# Chop down firmly
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-25, 18, -25))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (70, 18, 22))
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (5, 0, 0))
keyframe_all_bones(ch28_arm, 40)
keyframe_all_bones(ch28_arm, 60)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_stop)

# ------------------------------------------------------------------------------
# 10. SBV_GO (Both index fingers pointing and sweeping forward, 85 frames)
# ------------------------------------------------------------------------------
act_go = bpy.data.actions.new(name="SBV_GO")
act_go.use_fake_user = True
ch28_arm.animation_data.action = act_go
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Hands raised, index fingers ready
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-30, 20, -25))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 15, 20))
set_fist(ch28_arm, 'Right', 65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-30, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (75, -15, -20))
set_fist(ch28_arm, 'Left', 65)
set_finger_curl(ch28_arm, 'Left', 'Index', -5)
keyframe_all_bones(ch28_arm, 25)
# Roll and sweep forward
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-18, 15, -15))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (45, 10, 15))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-18, -15, 15))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (45, -10, -15))
keyframe_all_bones(ch28_arm, 48)
keyframe_all_bones(ch28_arm, 62)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_go)

# ------------------------------------------------------------------------------
# 11. SBV_WAIT (Open palms pressing down in calming pause, 80 frames)
# ------------------------------------------------------------------------------
act_wait = bpy.data.actions.new(name="SBV_WAIT")
act_wait.use_fake_user = True
ch28_arm.animation_data.action = act_wait
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Palms flat forward at waist
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-28, 18, -20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (65, 15, 15))
set_flat_hand(ch28_arm, 'Right')
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-28, -18, 20))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (65, -15, -15))
set_flat_hand(ch28_arm, 'Left')
keyframe_all_bones(ch28_arm, 25)
# Press down 1
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (55, 15, 15))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (55, -15, -15))
keyframe_all_bones(ch28_arm, 40)
# Press down 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (65, 15, 15))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (65, -15, -15))
keyframe_all_bones(ch28_arm, 52)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (55, 15, 15))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (55, -15, -15))
keyframe_all_bones(ch28_arm, 62)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 80)
actions_to_export.append(act_wait)

# ------------------------------------------------------------------------------
# 12. SBV_COME (Right palm inward beckoning towards chest, 80 frames)
# ------------------------------------------------------------------------------
act_come = bpy.data.actions.new(name="SBV_COME")
act_come.use_fake_user = True
ch28_arm.animation_data.action = act_come
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Arm forward, palm facing inward
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-25, 15, -20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (60, 20, 25))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 22)
# Curl inward stroke 1
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 25, 20))
for f in ['Index', 'Middle', 'Ring', 'Pinky']:
    set_finger_curl(ch28_arm, 'Right', f, 50)
keyframe_all_bones(ch28_arm, 38)
# Extend slightly
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (60, 20, 25))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 48)
# Curl inward stroke 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 30))
for f in ['Index', 'Middle', 'Ring', 'Pinky']:
    set_finger_curl(ch28_arm, 'Right', f, 50)
keyframe_all_bones(ch28_arm, 60)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 80)
actions_to_export.append(act_come)

# ------------------------------------------------------------------------------
# 13. SBV_UNDERSTAND (Index finger flicking up at temple + firm nod, 80 frames)
# ------------------------------------------------------------------------------
act_under = bpy.data.actions.new(name="SBV_UNDERSTAND")
act_under.use_fake_user = True
ch28_arm.animation_data.action = act_under
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Hand near temple, index finger ready
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-55, 30, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (105, 18, 40))
set_fist(ch28_arm, 'Right', 65)
keyframe_all_bones(ch28_arm, 22)
# Flick upward + emphatic head nod
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (8, 0, 0))
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
keyframe_all_bones(ch28_arm, 38)
# Hold nod & understanding
keyframe_all_bones(ch28_arm, 58)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 80)
actions_to_export.append(act_under)

# ------------------------------------------------------------------------------
# 14. SBV_DONT_UNDERSTAND (Index at temple + head shake + outward flick, 85 frames)
# ------------------------------------------------------------------------------
act_dont = bpy.data.actions.new(name="SBV_DONT_UNDERSTAND")
act_dont.use_fake_user = True
ch28_arm.animation_data.action = act_dont
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Hand near temple
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-50, 28, -45))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (100, 15, 35))
set_fist(ch28_arm, 'Right', 60)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
keyframe_all_bones(ch28_arm, 22)
# Head shake left
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (0, 12, 0))
keyframe_all_bones(ch28_arm, 35)
# Head shake right + outward hand flick
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (0, -12, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-40, 20, -35))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 10, 25))
keyframe_all_bones(ch28_arm, 50)
keyframe_all_bones(ch28_arm, 65)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_dont)

# ------------------------------------------------------------------------------
# 15. SBV_NAME (Both H-hands tapping across each other twice, 85 frames)
# ------------------------------------------------------------------------------
act_name = bpy.data.actions.new(name="SBV_NAME")
act_name.use_fake_user = True
ch28_arm.animation_data.action = act_name
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# H-handshapes (Index + Middle extended together)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-28, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (75, -20, -15))
set_fist(ch28_arm, 'Left', 65)
set_finger_curl(ch28_arm, 'Left', 'Index', -5)
set_finger_curl(ch28_arm, 'Left', 'Middle', -5)

rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 22, -28))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (82, 20, 22))
set_fist(ch28_arm, 'Right', 65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
set_finger_curl(ch28_arm, 'Right', 'Middle', -5)
keyframe_all_bones(ch28_arm, 25)
# Tap 1 down across left H-hand
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 22))
keyframe_all_bones(ch28_arm, 38)
# Rebound
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (82, 20, 22))
keyframe_all_bones(ch28_arm, 48)
# Tap 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 22))
keyframe_all_bones(ch28_arm, 58)
keyframe_all_bones(ch28_arm, 68)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_name)

# ------------------------------------------------------------------------------
# 16. SBV_EMERGENCY (Alert posture + rapid urgent alternating shake, 85 frames)
# ------------------------------------------------------------------------------
act_emerg = bpy.data.actions.new(name="SBV_EMERGENCY")
act_emerg.use_fake_user = True
ch28_arm.animation_data.action = act_emerg
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Both fists raised alert at chest
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 20, -25))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (80, 18, 20))
set_fist(ch28_arm, 'Right', 65)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-35, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (80, -18, -20))
set_fist(ch28_arm, 'Left', 65)
keyframe_all_bones(ch28_arm, 22)
# Urgent shake 1
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 15, 10))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftHand', (0, -15, -10))
keyframe_all_bones(ch28_arm, 32)
# Urgent shake 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, -15, -10))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftHand', (0, 15, 10))
keyframe_all_bones(ch28_arm, 44)
# Urgent shake 3
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 12, 8))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftHand', (0, -12, -8))
keyframe_all_bones(ch28_arm, 56)
keyframe_all_bones(ch28_arm, 68)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_emerg)

# ------------------------------------------------------------------------------
# 17. SBV_POLICE (Badge tap at left chest + crisp posture, 85 frames)
# ------------------------------------------------------------------------------
act_pol = bpy.data.actions.new(name="SBV_POLICE")
act_pol.use_fake_user = True
ch28_arm.animation_data.action = act_pol
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# C-hand tap at left chest badge
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Spine', (-3, 0, 0)) # crisp erect posture
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-38, 30, -55))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 18, 35))
set_finger_curl(ch28_arm, 'Right', 'Index', 25)
set_finger_curl(ch28_arm, 'Right', 'Middle', 25)
set_finger_curl(ch28_arm, 'Right', 'Ring', 25)
set_finger_curl(ch28_arm, 'Right', 'Pinky', 25)
keyframe_all_bones(ch28_arm, 25)
# Tap 1
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (95, 18, 35))
keyframe_all_bones(ch28_arm, 38)
# Rebound
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (88, 18, 35))
keyframe_all_bones(ch28_arm, 48)
# Tap 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (95, 18, 35))
keyframe_all_bones(ch28_arm, 58)
keyframe_all_bones(ch28_arm, 68)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_pol)

# ------------------------------------------------------------------------------
# 18. SBV_HOME (Both hands pitched roof touching over chest, 85 frames)
# ------------------------------------------------------------------------------
act_home = bpy.data.actions.new(name="SBV_HOME")
act_home.use_fake_user = True
ch28_arm.animation_data.action = act_home
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Hands form pitched roof
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 25, -35))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 20, 25))
set_flat_hand(ch28_arm, 'Right')
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-32, -25, 35))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (85, -20, -25))
set_flat_hand(ch28_arm, 'Left')
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (4, 0, 0))
keyframe_all_bones(ch28_arm, 25)
# Hold roof
keyframe_all_bones(ch28_arm, 62)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_home)

# ------------------------------------------------------------------------------
# 19. SBV_SCHOOL (Flat hands clapping horizontally twice, 85 frames)
# ------------------------------------------------------------------------------
act_school = bpy.data.actions.new(name="SBV_SCHOOL")
act_school.use_fake_user = True
ch28_arm.animation_data.action = act_school
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Base clap posture
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-25, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (75, -20, -15))
set_flat_hand(ch28_arm, 'Left')
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-30, 22, -28))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (82, 20, 22))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 25)
# Clap 1
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 22))
keyframe_all_bones(ch28_arm, 38)
# Rebound
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (82, 20, 22))
keyframe_all_bones(ch28_arm, 48)
# Clap 2
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 22))
keyframe_all_bones(ch28_arm, 58)
keyframe_all_bones(ch28_arm, 68)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_school)

# ------------------------------------------------------------------------------
# 20. SBV_NEED (Cupped hands pulling downward with intent, 80 frames)
# ------------------------------------------------------------------------------
act_need = bpy.data.actions.new(name="SBV_NEED")
act_need.use_fake_user = True
ch28_arm.animation_data.action = act_need
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
# Hands cupped at chest
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 20, -25))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (80, 15, 20))
set_finger_curl(ch28_arm, 'Right', 'Index', 45)
set_finger_curl(ch28_arm, 'Right', 'Middle', 45)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-32, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (80, -15, -20))
set_finger_curl(ch28_arm, 'Left', 'Index', 45)
set_finger_curl(ch28_arm, 'Left', 'Middle', 45)
keyframe_all_bones(ch28_arm, 22)
# Firm downward pull
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (60, 12, 15))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (60, -12, -15))
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (6, 0, 0))
keyframe_all_bones(ch28_arm, 45)
keyframe_all_bones(ch28_arm, 60)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 80)
actions_to_export.append(act_need)

# ------------------------------------------------------------------------------
# 21. SBV_HOSPITAL (Cross touch on left upper arm, 85 frames)
# ------------------------------------------------------------------------------
act_hosp = bpy.data.actions.new(name="SBV_HOSPITAL")
act_hosp.use_fake_user = True
ch28_arm.animation_data.action = act_hosp
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 30, -60))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 20, 35))
set_fist(ch28_arm, 'Right', 65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
set_finger_curl(ch28_arm, 'Right', 'Middle', -5)
keyframe_all_bones(ch28_arm, 25)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 18, 32))
keyframe_all_bones(ch28_arm, 40)
keyframe_all_bones(ch28_arm, 58)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_hosp)

# ------------------------------------------------------------------------------
# 22. SBV_BATHROOM (T-hand shaking side to side, 85 frames)
# ------------------------------------------------------------------------------
act_bath = bpy.data.actions.new(name="SBV_BATHROOM")
act_bath.use_fake_user = True
ch28_arm.animation_data.action = act_bath
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 20, -30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 20))
set_fist(ch28_arm, 'Right', 65)
keyframe_all_bones(ch28_arm, 25)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 20, 15))
keyframe_all_bones(ch28_arm, 38)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, -20, -15))
keyframe_all_bones(ch28_arm, 50)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 15, 10))
keyframe_all_bones(ch28_arm, 62)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_bath)

# ------------------------------------------------------------------------------
# 23. SBV_CRYING (Tears down cheeks, 90 frames)
# ------------------------------------------------------------------------------
act_crying = bpy.data.actions.new(name="SBV_CRYING")
act_crying.use_fake_user = True
ch28_arm.animation_data.action = act_crying
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (10, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-55, 35, -55))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (105, 20, 50))
set_fist(ch28_arm, 'Right', 65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-55, -35, 55))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (105, -20, -50))
set_fist(ch28_arm, 'Left', 65)
set_finger_curl(ch28_arm, 'Left', 'Index', -5)
keyframe_all_bones(ch28_arm, 25)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (14, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-42, 28, -42))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 38))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-42, -28, 42))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (85, -15, -38))
keyframe_all_bones(ch28_arm, 55)
keyframe_all_bones(ch28_arm, 70)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_crying)

# ------------------------------------------------------------------------------
# 24. SBV_GOING (Outward sweeping wave, 85 frames)
# ------------------------------------------------------------------------------
act_going = bpy.data.actions.new(name="SBV_GOING")
act_going.use_fake_user = True
ch28_arm.animation_data.action = act_going
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 25, -40))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (70, 15, 25))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 22)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (0, -10, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-15, -20, 30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (30, 5, 10))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 48)
keyframe_all_bones(ch28_arm, 65)
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_going)

# ------------------------------------------------------------------------------
# 25. High-Fidelity Imported FBX Clips (HELLO, YES, NO, WHERE)
# ------------------------------------------------------------------------------
fbx_map = [
    (r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx", "SBV_HELLO"),
    (r"C:\Users\drx20\Downloads\3d model\Head Nod Yes.fbx", "SBV_YES"),
    (r"C:\Users\drx20\Downloads\3d model\No.fbx", "SBV_NO"),
    (r"C:\Users\drx20\Downloads\3d model\Whatever Gesture.fbx", "SBV_WHERE"),
]

for fbx_path, target_name in fbx_map:
    if not os.path.exists(fbx_path):
        print(f"Skipping {fbx_path} (not found)")
        continue
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    imported_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o != ch28_arm]
    if imported_arm:
        arm_obj = imported_arm[0]
        if arm_obj.animation_data and arm_obj.animation_data.action:
            src_act = arm_obj.animation_data.action
            src_act.name = target_name
            src_act.use_fake_user = True
            actions_to_export.append(src_act)
            print(f"Imported {target_name} ({src_act.frame_range[1]} frames)")
    for o in list(bpy.data.objects):
        if o != ch28_arm and o.name not in ['Ch28_Body', 'Ch28_Eyelashes', 'Ch28_Hair', 'Ch28_Hoody', 'Ch28_Pants', 'Ch28_Sneakers']:
            bpy.data.objects.remove(o, do_unlink=True)

for img in list(bpy.data.images):
    if '.00' in img.name or 'Ch28' not in img.name:
        bpy.data.images.remove(img)

# ------------------------------------------------------------------------------
# STASH ALL ACTIONS INTO NLA TRACKS FOR COMPLETE GLTF EXPORT
# ------------------------------------------------------------------------------
print(f"Total actions to export: {len(actions_to_export)}")
for act in actions_to_export:
    track = ch28_arm.animation_data.nla_tracks.new()
    track.name = act.name
    strip = track.strips.new(act.name, int(act.frame_range[0]), act)
    print(f"  Stashed NLA Track: {track.name}")

# Set active action to SBV_IDLE
ch28_arm.animation_data.action = act_idle
if len(act_idle.slots) > 0:
    ch28_arm.animation_data.action_slot = act_idle.slots[0]

for o in list(bpy.data.objects):
    if 'Icosphere' in o.name or (o.type == 'MESH' and not o.name.startswith('Ch28')):
        bpy.data.objects.remove(o, do_unlink=True)

prod_glb = r"C:\Users\drx20\.gemini\antigravity\scratch\signbridge2vision\apps\web\public\models\sign-avatar\sign-avatar.glb"
prod_blend = r"C:\Users\drx20\.gemini\antigravity\scratch\signbridge2vision\apps\web\public\models\sign-avatar\sign-avatar.blend"

# Save .blend
bpy.ops.wm.save_as_mainfile(filepath=prod_blend)
print(f"Saved blend file to: {prod_blend}")

# Export GLB
bpy.ops.export_scene.gltf(
    filepath=prod_glb,
    export_format='GLB',
    export_animations=True,
    export_nla_strips=True,
    export_apply=False
)

size_mb = os.path.getsize(prod_glb) / (1024 * 1024)
print(f"GLB EXPORT COMPLETE! Size: {size_mb:.2f} MB at {prod_glb}")

# Verification
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=prod_glb)
print("=== VERIFICATION ===")
print("Objects in exported GLB:", [o.name for o in bpy.data.objects])
print(f"Total actions in exported GLB: {len(bpy.data.actions)}")
for a in bpy.data.actions:
    print(f" - {a.name}: {a.frame_range[1]} frames")
