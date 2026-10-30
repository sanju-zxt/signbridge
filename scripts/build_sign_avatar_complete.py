import bpy
import json
import math
import mathutils
import os

print("=== STARTING SIGN AVATAR BUILD ===")

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

# Ensure animation data on ch28_arm
if not ch28_arm.animation_data:
    ch28_arm.animation_data_create()

# Clear existing NLA tracks
for track in list(ch28_arm.animation_data.nla_tracks):
    ch28_arm.animation_data.nla_tracks.remove(track)

actions_to_export = []

# ==============================================================================
# ACTION 1: SBV_IDLE (Natural standing breathing loop, 60 frames)
# ==============================================================================
act_idle = bpy.data.actions.new(name="SBV_IDLE")
act_idle.use_fake_user = True
ch28_arm.animation_data.action = act_idle

# Frame 1: idle
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 30: subtle inhale/breathing
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Spine1', (-1.5, 0, 0)) # slight chest rise
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (1.0, 0, 0))
keyframe_all_bones(ch28_arm, 30)

# Frame 60: return to idle
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 60)
actions_to_export.append(act_idle)
print("Built SBV_IDLE")

# ==============================================================================
# ACTION 2: SBV_THANK_YOU (Chin touch -> forward bow sweep, 90 frames)
# ==============================================================================
act_ty = bpy.data.actions.new(name="SBV_THANK_YOU")
act_ty.use_fake_user = True
ch28_arm.animation_data.action = act_ty

# Frame 1: rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 25: fingers touch chin
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-45, 30, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 20, 45))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (15, -25, 20))
set_flat_hand(ch28_arm, 'Right')
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (5, 0, 0))
keyframe_all_bones(ch28_arm, 25)

# Frame 35: hold at chin
keyframe_all_bones(ch28_arm, 35)

# Frame 60: forward sweep toward viewer + polite torso bow
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Spine', (10, 0, 0)) # bow
rotate_bone_rel(ch28_arm, 'mixamorig10:Spine1', (5, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (8, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-20, 20, -35))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (40, 10, 20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (5, -10, 10))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 60)

# Frame 70: hold apex
keyframe_all_bones(ch28_arm, 70)

# Frame 90: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_ty)
print("Built SBV_THANK_YOU")

# ==============================================================================
# ACTION 3: SBV_SORRY (Right fist on chest over heart, circular rub, 90 frames)
# ==============================================================================
act_sorry = bpy.data.actions.new(name="SBV_SORRY")
act_sorry.use_fake_user = True
ch28_arm.animation_data.action = act_sorry

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 25: fist over sternum, head sorrow tilt
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (5, -8, 8)) # empathetic sorrow tilt
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 30, -55))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 15, 35))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (10, -20, 15))
set_fist(ch28_arm, 'Right', curl_deg=60)
keyframe_all_bones(ch28_arm, 25)

# Frame 35: circular rub stroke 1 up-right
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (6, -8, 8))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 28, -52))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 18, 38))
set_fist(ch28_arm, 'Right', curl_deg=60)
keyframe_all_bones(ch28_arm, 35)

# Frame 45: circular rub stroke 1 down-left
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (6, -8, 8))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-38, 32, -58))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (95, 12, 32))
set_fist(ch28_arm, 'Right', curl_deg=60)
keyframe_all_bones(ch28_arm, 45)

# Frame 55: circular rub stroke 2 up-right
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (6, -8, 8))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 28, -52))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 18, 38))
set_fist(ch28_arm, 'Right', curl_deg=60)
keyframe_all_bones(ch28_arm, 55)

# Frame 65: circular rub stroke 2 down-left
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (5, -8, 8))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 30, -55))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 15, 35))
set_fist(ch28_arm, 'Right', curl_deg=60)
keyframe_all_bones(ch28_arm, 65)

# Frame 90: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_sorry)
print("Built SBV_SORRY")

# ==============================================================================
# ACTION 4: SBV_CRYING (Both index fingers trace tears down cheeks, 90 frames)
# ==============================================================================
act_crying = bpy.data.actions.new(name="SBV_CRYING")
act_crying.use_fake_user = True
ch28_arm.animation_data.action = act_crying

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 25: fingers at lower eyelid
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (10, 0, 0)) # sorrow bow
# Right hand
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-55, 35, -55))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (105, 20, 50))
set_fist(ch28_arm, 'Right', curl_deg=65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5) # index straight pointing at eye
# Left hand
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-55, -35, 55))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (105, -20, -50))
set_fist(ch28_arm, 'Left', curl_deg=65)
set_finger_curl(ch28_arm, 'Left', 'Index', -5) # index straight pointing at eye
keyframe_all_bones(ch28_arm, 25)

# Frame 45: trace tears down cheeks (halfway) + slight tremor
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (12, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-50, 32, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (95, 18, 45))
set_fist(ch28_arm, 'Right', curl_deg=65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-50, -32, 50))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (95, -18, -45))
set_fist(ch28_arm, 'Left', curl_deg=65)
set_finger_curl(ch28_arm, 'Left', 'Index', -5)
keyframe_all_bones(ch28_arm, 45)

# Frame 65: trace tears to jawline
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (14, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-42, 28, -42))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 38))
set_fist(ch28_arm, 'Right', curl_deg=65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-42, -28, 42))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (85, -15, -38))
set_fist(ch28_arm, 'Left', curl_deg=65)
set_finger_curl(ch28_arm, 'Left', 'Index', -5)
keyframe_all_bones(ch28_arm, 65)

# Frame 90: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_crying)
print("Built SBV_CRYING")

# ==============================================================================
# ACTION 5: SBV_GOING (Dominant hand sweeps forward and right, 85 frames)
# ==============================================================================
act_going = bpy.data.actions.new(name="SBV_GOING")
act_going.use_fake_user = True
ch28_arm.animation_data.action = act_going

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 25: hand raised at chest
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 25, -40))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (70, 15, 25))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 25)

# Frame 50: sweep outward and away
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (0, -10, 0)) # turn with departure
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-15, -20, 30)) # swept to right
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (30, 5, 10))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 20, 25))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 50)

# Frame 65: hold sweep
keyframe_all_bones(ch28_arm, 65)

# Frame 85: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_going)
print("Built SBV_GOING")

# ==============================================================================
# ACTION 6: SBV_PLEASE (Flat palm rubs center of chest, 85 frames)
# ==============================================================================
act_please = bpy.data.actions.new(name="SBV_PLEASE")
act_please.use_fake_user = True
ch28_arm.animation_data.action = act_please

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 20: palm flat on chest
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (6, 0, 0)) # polite nod
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 28, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (10, -20, 10))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 20)

# Frame 35: circular rub stroke 1
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (8, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-30, 25, -45))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (80, 18, 35))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 35)

# Frame 50: circular rub stroke 2
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (6, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-38, 30, -55))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 12, 28))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 50)

# Frame 65: hold flat
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (5, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 28, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 30))
set_flat_hand(ch28_arm, 'Right')
keyframe_all_bones(ch28_arm, 65)

# Frame 85: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_please)
print("Built SBV_PLEASE")

# ==============================================================================
# ACTION 7: SBV_HELP (Open left palm base + right fist lifted upward, 90 frames)
# ==============================================================================
act_help = bpy.data.actions.new(name="SBV_HELP")
act_help.use_fake_user = True
ch28_arm.animation_data.action = act_help

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 25: left palm face up at chest, right fist placed on top
apply_idle_pose(ch28_arm)
# Left hand (support base, palm up)
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-25, -20, 25))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (75, -25, -20))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftHand', (-20, 30, -10))
set_flat_hand(ch28_arm, 'Left')
# Right hand (fist on palm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-25, 20, -25))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (75, 20, 25))
set_fist(ch28_arm, 'Right', curl_deg=65)
keyframe_all_bones(ch28_arm, 25)

# Frame 55: lift both hands together upward towards chin
apply_idle_pose(ch28_arm)
# Left hand lifted
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftArm', (-38, -25, 30))
rotate_bone_rel(ch28_arm, 'mixamorig10:LeftForeArm', (88, -25, -20))
set_flat_hand(ch28_arm, 'Left')
# Right hand lifted
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-38, 25, -30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (88, 20, 25))
set_fist(ch28_arm, 'Right', curl_deg=65)
keyframe_all_bones(ch28_arm, 55)

# Frame 68: hold lifted
keyframe_all_bones(ch28_arm, 68)

# Frame 90: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 90)
actions_to_export.append(act_help)
print("Built SBV_HELP")

# ==============================================================================
# ACTION 8: SBV_WATER (W-handshape tapping chin twice, 85 frames)
# ==============================================================================
act_water = bpy.data.actions.new(name="SBV_WATER")
act_water.use_fake_user = True
ch28_arm.animation_data.action = act_water

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

def set_w_handshape(arm_obj):
    set_finger_curl(arm_obj, 'Right', 'Index', -5)  # extended
    set_finger_curl(arm_obj, 'Right', 'Middle', -5) # extended
    set_finger_curl(arm_obj, 'Right', 'Ring', -5)   # extended
    set_finger_curl(arm_obj, 'Right', 'Pinky', 75)  # pinned down
    # Thumb pinning pinky
    tb1 = arm_obj.pose.bones.get("mixamorig10:RightHandThumb1")
    if tb1:
        q_b = mathutils.Quaternion(idle_pose.get(tb1.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb1.rotation_quaternion = q_b @ mathutils.Euler((math.radians(35), math.radians(20), math.radians(-25)), 'XYZ').to_quaternion()

# Frame 25: tap 1 at chin
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-50, 32, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (100, 20, 45))
set_w_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 25)

# Frame 35: slight retract
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-45, 30, -45))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 18, 40))
set_w_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 35)

# Frame 45: tap 2 at chin
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-50, 32, -50))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (100, 20, 45))
set_w_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 45)

# Frame 55: hold
keyframe_all_bones(ch28_arm, 55)

# Frame 85: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_water)
print("Built SBV_WATER")

# ==============================================================================
# ACTION 9: SBV_FOOD (Pinched flat-O tapping mouth twice, 85 frames)
# ==============================================================================
act_food = bpy.data.actions.new(name="SBV_FOOD")
act_food.use_fake_user = True
ch28_arm.animation_data.action = act_food

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

def set_pinch_handshape(arm_obj):
    for f in ['Index', 'Middle', 'Ring', 'Pinky']:
        set_finger_curl(arm_obj, 'Right', f, 40) # pinched cluster
    tb1 = arm_obj.pose.bones.get("mixamorig10:RightHandThumb1")
    if tb1:
        q_b = mathutils.Quaternion(idle_pose.get(tb1.name, {}).get('rotation_quaternion', (1, 0, 0, 0)))
        tb1.rotation_quaternion = q_b @ mathutils.Euler((math.radians(35), math.radians(10), math.radians(-15)), 'XYZ').to_quaternion()

# Frame 25: tap 1 at mouth
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:Head', (3, 0, 0))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-52, 35, -52))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (105, 20, 48))
set_pinch_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 25)

# Frame 35: slight retract
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-46, 30, -46))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (95, 18, 42))
set_pinch_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 35)

# Frame 45: tap 2 at mouth
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-52, 35, -52))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (105, 20, 48))
set_pinch_handshape(ch28_arm)
keyframe_all_bones(ch28_arm, 45)

# Frame 55: hold
keyframe_all_bones(ch28_arm, 55)

# Frame 85: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_food)
print("Built SBV_FOOD")

# ==============================================================================
# ACTION 10: SBV_HOSPITAL (Cross touch on left upper arm, 85 frames)
# ==============================================================================
act_hosp = bpy.data.actions.new(name="SBV_HOSPITAL")
act_hosp.use_fake_user = True
ch28_arm.animation_data.action = act_hosp

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 25: right index/middle touch left upper arm
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 30, -60))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (90, 20, 35))
set_fist(ch28_arm, 'Right', 65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
set_finger_curl(ch28_arm, 'Right', 'Middle', -5)
keyframe_all_bones(ch28_arm, 25)

# Frame 40: cross downward tap
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-32, 28, -58))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 18, 32))
set_fist(ch28_arm, 'Right', 65)
set_finger_curl(ch28_arm, 'Right', 'Index', -5)
set_finger_curl(ch28_arm, 'Right', 'Middle', -5)
keyframe_all_bones(ch28_arm, 40)

# Frame 55: hold
keyframe_all_bones(ch28_arm, 55)

# Frame 85: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_hosp)
print("Built SBV_HOSPITAL")

# ==============================================================================
# ACTION 11: SBV_BATHROOM (T-hand shaking side to side, 85 frames)
# ==============================================================================
act_bath = bpy.data.actions.new(name="SBV_BATHROOM")
act_bath.use_fake_user = True
ch28_arm.animation_data.action = act_bath

apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 1)

# Frame 25: T-fist raised at shoulder
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 20, -30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 20))
set_fist(ch28_arm, 'Right', 65)
keyframe_all_bones(ch28_arm, 25)

# Shake 1
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 20, -30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 20, 15))
set_fist(ch28_arm, 'Right', 65)
keyframe_all_bones(ch28_arm, 35)

# Shake 2
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 20, -30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, -20, -15))
set_fist(ch28_arm, 'Right', 65)
keyframe_all_bones(ch28_arm, 45)

# Shake 3
apply_idle_pose(ch28_arm)
rotate_bone_rel(ch28_arm, 'mixamorig10:RightArm', (-35, 20, -30))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightForeArm', (85, 15, 20))
rotate_bone_rel(ch28_arm, 'mixamorig10:RightHand', (0, 15, 10))
set_fist(ch28_arm, 'Right', 65)
keyframe_all_bones(ch28_arm, 55)

# Frame 85: return to rest
apply_idle_pose(ch28_arm)
keyframe_all_bones(ch28_arm, 85)
actions_to_export.append(act_bath)
print("Built SBV_BATHROOM")

# ==============================================================================
# 5. IMPORT HIGH-FIDELITY MIXAMO ANIMATIONS
#    (Standing Greeting -> SBV_HELLO, Head Nod Yes -> SBV_YES, No -> SBV_NO, Whatever -> SBV_WHERE)
# ==============================================================================
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
    # Import into current scene
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    # Find the newly imported armature
    imported_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o != ch28_arm]
    if imported_arm:
        arm_obj = imported_arm[0]
        if arm_obj.animation_data and arm_obj.animation_data.action:
            src_act = arm_obj.animation_data.action
            src_act.name = target_name
            src_act.use_fake_user = True
            actions_to_export.append(src_act)
            print(f"Imported {target_name} ({src_act.frame_range[1]} frames)")
    # Delete imported objects and meshes
    for o in list(bpy.data.objects):
        if o != ch28_arm and o.name not in ['Ch28_Body', 'Ch28_Eyelashes', 'Ch28_Hair', 'Ch28_Hoody', 'Ch28_Pants', 'Ch28_Sneakers']:
            bpy.data.objects.remove(o, do_unlink=True)

# Delete unneeded images and materials brought by FBX
for img in list(bpy.data.images):
    if '.00' in img.name or 'Ch28' not in img.name:
        bpy.data.images.remove(img)

# ==============================================================================
# 6. STASH ALL ACTIONS INTO NLA TRACKS FOR GLTF EXPORT
# ==============================================================================
print(f"Total actions to export: {len(actions_to_export)}")
for act in actions_to_export:
    track = ch28_arm.animation_data.nla_tracks.new()
    track.name = act.name
    # Strip start at frame 1
    strip = track.strips.new(act.name, int(act.frame_range[0]), act)
    print(f"  Stashed NLA Track: {track.name}")

# Assign SBV_IDLE as active action on armature
ch28_arm.animation_data.action = act_idle
if len(act_idle.slots) > 0:
    ch28_arm.animation_data.action_slot = act_idle.slots[0]

# Ensure Icosphere and any non-Ch28 objects are deleted
for o in list(bpy.data.objects):
    if 'Icosphere' in o.name or (o.type == 'MESH' and not o.name.startswith('Ch28')):
        bpy.data.objects.remove(o, do_unlink=True)

print("Remaining objects in scene:", [o.name for o in bpy.data.objects])

# ==============================================================================
# 7. EXPORT TO PRODUCTION GLB & BLEND
# ==============================================================================
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
print("Clips in exported GLB:", [a.name for a in bpy.data.actions])
