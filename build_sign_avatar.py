import bpy
import math
import os
from mathutils import Vector, Euler, Matrix, Quaternion

print("=" * 60)
print("BUILDING PRODUCTION SIGNBRIDGE2VISION 3D SIGNING AVATAR")
print("=" * 60)

# 1. Reset / Clear Default Scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# 2. Collections Setup
def create_collection(name, parent=None):
    col = bpy.data.collections.new(name)
    if parent:
        parent.children.link(col)
    else:
        scene.collection.children.link(col)
    return col

col_avatar = create_collection("SBV_AVATAR")
col_mesh = create_collection("SBV_Mesh", col_avatar)
col_armature = create_collection("SBV_Armature", col_avatar)
col_controls = create_collection("SBV_Controls", col_avatar)
col_hands = create_collection("SBV_Hands", col_avatar)
col_head = create_collection("SBV_Head", col_avatar)
col_clothing = create_collection("SBV_Clothing", col_avatar)

col_cameras = create_collection("SBV_CAMERAS")
col_lights = create_collection("SBV_LIGHTS")
col_animations = create_collection("SBV_ANIMATIONS")

# 3. Create Armature
armature_data = bpy.data.armatures.new("SBV_Armature_Data")
armature_obj = bpy.data.objects.new("SBV_Armature", armature_data)
col_armature.objects.link(armature_obj)
bpy.context.view_layer.objects.active = armature_obj
armature_obj.select_set(True)

bpy.ops.object.mode_set(mode='EDIT')
edit_bones = armature_data.edit_bones

def add_bone(name, head, tail, parent_name=None, use_connect=False):
    b = edit_bones.new(name)
    b.head = Vector(head)
    b.tail = Vector(tail)
    if parent_name and parent_name in edit_bones:
        b.parent = edit_bones[parent_name]
        b.use_connect = use_connect
    return b

# Core Spine / Body Bones
add_bone("root", (0, 0, 0), (0, 0, 0.1))
add_bone("pelvis", (0, 0, 0.90), (0, 0, 0.98), "root")
add_bone("spine", (0, 0, 0.98), (0, 0, 1.10), "pelvis")
add_bone("spine_01", (0, 0, 1.10), (0, 0, 1.22), "spine")
add_bone("spine_02", (0, 0, 1.22), (0, 0, 1.34), "spine_01")
add_bone("chest", (0, 0, 1.34), (0, 0, 1.45), "spine_02")
add_bone("neck", (0, 0, 1.45), (0, 0, 1.55), "chest")
add_bone("head", (0, 0, 1.55), (0, 0, 1.75), "neck")

# Legs
add_bone("left_thigh", (0.10, 0, 0.90), (0.10, 0, 0.50), "pelvis")
add_bone("left_shin", (0.10, 0, 0.50), (0.10, 0, 0.08), "left_thigh")
add_bone("left_foot", (0.10, 0, 0.08), (0.10, 0.16, 0.0), "left_shin")

add_bone("right_thigh", (-0.10, 0, 0.90), (-0.10, 0, 0.50), "pelvis")
add_bone("right_shin", (-0.10, 0, 0.50), (-0.10, 0, 0.08), "right_thigh")
add_bone("right_foot", (-0.10, 0, 0.08), (-0.10, 0.16, 0.0), "right_shin")

# Left Arm & Shoulder
add_bone("left_shoulder", (0.04, 0, 1.40), (0.18, 0, 1.40), "chest")
add_bone("left_upper_arm", (0.18, 0, 1.40), (0.42, 0, 1.40), "left_shoulder")
add_bone("left_forearm", (0.42, 0, 1.40), (0.66, 0, 1.40), "left_upper_arm")
add_bone("left_hand", (0.66, 0, 1.40), (0.75, 0, 1.40), "left_forearm")

# Left Hand 5 Fingers (3 segments each = 15 bones)
# Thumb
add_bone("left_thumb_01", (0.69, 0.03, 1.40), (0.72, 0.05, 1.40), "left_hand")
add_bone("left_thumb_02", (0.72, 0.05, 1.40), (0.74, 0.07, 1.40), "left_thumb_01")
add_bone("left_thumb_03", (0.74, 0.07, 1.40), (0.76, 0.08, 1.40), "left_thumb_02")

# Index
add_bone("left_index_01", (0.75, 0.025, 1.40), (0.78, 0.025, 1.40), "left_hand")
add_bone("left_index_02", (0.78, 0.025, 1.40), (0.805, 0.025, 1.40), "left_index_01")
add_bone("left_index_03", (0.805, 0.025, 1.40), (0.825, 0.025, 1.40), "left_index_02")

# Middle
add_bone("left_middle_01", (0.75, 0.008, 1.40), (0.785, 0.008, 1.40), "left_hand")
add_bone("left_middle_02", (0.785, 0.008, 1.40), (0.815, 0.008, 1.40), "left_middle_01")
add_bone("left_middle_03", (0.815, 0.008, 1.40), (0.835, 0.008, 1.40), "left_middle_02")

# Ring
add_bone("left_ring_01", (0.75, -0.008, 1.40), (0.78, -0.008, 1.40), "left_hand")
add_bone("left_ring_02", (0.78, -0.008, 1.40), (0.805, -0.008, 1.40), "left_ring_01")
add_bone("left_ring_03", (0.805, -0.008, 1.40), (0.825, -0.008, 1.40), "left_ring_02")

# Pinky
add_bone("left_pinky_01", (0.75, -0.025, 1.40), (0.775, -0.025, 1.40), "left_hand")
add_bone("left_pinky_02", (0.775, -0.025, 1.40), (0.795, -0.025, 1.40), "left_pinky_01")
add_bone("left_pinky_03", (0.795, -0.025, 1.40), (0.815, -0.025, 1.40), "left_pinky_02")

# Right Arm & Shoulder
add_bone("right_shoulder", (-0.04, 0, 1.40), (-0.18, 0, 1.40), "chest")
add_bone("right_upper_arm", (-0.18, 0, 1.40), (-0.42, 0, 1.40), "right_shoulder")
add_bone("right_forearm", (-0.42, 0, 1.40), (-0.66, 0, 1.40), "right_upper_arm")
add_bone("right_hand", (-0.66, 0, 1.40), (-0.75, 0, 1.40), "right_forearm")

# Right Hand 5 Fingers (3 segments each = 15 bones)
# Thumb
add_bone("right_thumb_01", (-0.69, 0.03, 1.40), (-0.72, 0.05, 1.40), "right_hand")
add_bone("right_thumb_02", (-0.72, 0.05, 1.40), (-0.74, 0.07, 1.40), "right_thumb_01")
add_bone("right_thumb_03", (-0.74, 0.07, 1.40), (-0.76, 0.08, 1.40), "right_thumb_02")

# Index
add_bone("right_index_01", (-0.75, 0.025, 1.40), (-0.78, 0.025, 1.40), "right_hand")
add_bone("right_index_02", (-0.78, 0.025, 1.40), (-0.805, 0.025, 1.40), "right_index_01")
add_bone("right_index_03", (-0.805, 0.025, 1.40), (-0.825, 0.025, 1.40), "right_index_02")

# Middle
add_bone("right_middle_01", (-0.75, 0.008, 1.40), (-0.785, 0.008, 1.40), "right_hand")
add_bone("right_middle_02", (-0.785, 0.008, 1.40), (-0.815, 0.008, 1.40), "right_middle_01")
add_bone("right_middle_03", (-0.815, 0.008, 1.40), (-0.835, 0.008, 1.40), "right_middle_02")

# Ring
add_bone("right_ring_01", (-0.75, -0.008, 1.40), (-0.78, -0.008, 1.40), "right_hand")
add_bone("right_ring_02", (-0.78, -0.008, 1.40), (-0.805, -0.008, 1.40), "right_ring_01")
add_bone("right_ring_03", (-0.805, -0.008, 1.40), (-0.825, -0.008, 1.40), "right_ring_02")

# Pinky
add_bone("right_pinky_01", (-0.75, -0.025, 1.40), (-0.775, -0.025, 1.40), "right_hand")
add_bone("right_pinky_02", (-0.775, -0.025, 1.40), (-0.795, -0.025, 1.40), "right_pinky_01")
add_bone("right_pinky_03", (-0.795, -0.025, 1.40), (-0.815, -0.025, 1.40), "right_pinky_02")

# IK Control Bones
add_bone("left_hand_IK", (0.75, 0, 1.40), (0.75, 0.1, 1.40), "root")
add_bone("right_hand_IK", (-0.75, 0, 1.40), (-0.75, 0.1, 1.40), "root")
add_bone("left_elbow_pole", (0.42, -0.4, 1.40), (0.42, -0.4, 1.50), "root")
add_bone("right_elbow_pole", (-0.42, -0.4, 1.40), (-0.42, -0.4, 1.50), "root")

bpy.ops.object.mode_set(mode='POSE')

# Add IK Constraints
pose_bones = armature_obj.pose.bones

# Left Arm IK
ik_left = pose_bones["left_forearm"].constraints.new('IK')
ik_left.target = armature_obj
ik_left.subtarget = "left_hand_IK"
ik_left.pole_target = armature_obj
ik_left.pole_subtarget = "left_elbow_pole"
ik_left.pole_angle = math.radians(90)
ik_left.chain_count = 2

# Right Arm IK
ik_right = pose_bones["right_forearm"].constraints.new('IK')
ik_right.target = armature_obj
ik_right.subtarget = "right_hand_IK"
ik_right.pole_target = armature_obj
ik_right.pole_subtarget = "right_elbow_pole"
ik_right.pole_angle = math.radians(-90)
ik_right.chain_count = 2

bpy.ops.object.mode_set(mode='OBJECT')

total_bones = len(armature_data.bones)
hand_bone_count = 32
print(f"Armature created: Total bones = {total_bones} (Hand/Finger bones = {hand_bone_count})")

# 4. Materials Setup
def create_material(name, color_rgba, roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color_rgba
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic
    return mat

mat_skin = create_material("Skin_Material", (0.61, 0.42, 0.26, 1.0), roughness=0.55) # Realistic warm skin tone
mat_shirt = create_material("Shirt_Material", (0.58, 0.82, 0.74, 1.0), roughness=0.45) # Mint/Teal Polo Shirt
mat_pants = create_material("Pants_Material", (0.12, 0.14, 0.13, 1.0), roughness=0.7) # Charcoal Trousers
mat_hair = create_material("Hair_Material", (0.09, 0.07, 0.05, 1.0), roughness=0.9) # Dark hair
mat_shoes = create_material("Shoes_Material", (0.05, 0.05, 0.05, 1.0), roughness=0.35) # Black shoes

# 5. Create Detailed Avatar Mesh
# We create modular mesh primitives joined into one optimized rigged mesh with dedicated vertex groups.
mesh_parts = []

def create_cylinder(name, radius_top, radius_bottom, depth, location, rotation=(0,0,0), mat=None):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius_bottom,
        depth=depth,
        location=location,
        rotation=rotation,
        vertices=16
    )
    obj = bpy.context.active_object
    obj.name = name
    if mat:
        obj.data.materials.append(mat)
    mesh_parts.append(obj)
    return obj

def create_box(name, size, location, rotation=(0,0,0), mat=None):
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=location,
        rotation=rotation
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        obj.data.materials.append(mat)
    mesh_parts.append(obj)
    return obj

def create_sphere(name, radius, location, scale=(1,1,1), mat=None):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=location,
        segments=20,
        ring_count=16
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        obj.data.materials.append(mat)
    mesh_parts.append(obj)
    return obj

# Head & Hair
head_obj = create_sphere("Mesh_Head", 0.11, (0, 0, 1.66), (0.9, 1.15, 1.0), mat_skin)
hair_obj = create_sphere("Mesh_Hair", 0.115, (0, -0.01, 1.72), (0.95, 0.85, 1.05), mat_hair)
neck_obj = create_cylinder("Mesh_Neck", 0.065, 0.07, 0.12, (0, 0, 1.50), mat=mat_skin)

# Torso (Polo Shirt)
chest_obj = create_cylinder("Mesh_Chest", 0.20, 0.18, 0.24, (0, 0, 1.35), mat=mat_shirt)
spine_obj = create_cylinder("Mesh_Spine", 0.18, 0.16, 0.24, (0, 0, 1.16), mat=mat_shirt)
pelvis_obj = create_box("Mesh_Pelvis", (0.34, 0.22, 0.16), (0, 0, 0.98), mat=mat_pants)

# Legs & Feet
l_thigh_obj = create_cylinder("Mesh_L_Thigh", 0.08, 0.065, 0.40, (0.10, 0, 0.70), mat=mat_pants)
l_shin_obj = create_cylinder("Mesh_L_Shin", 0.065, 0.055, 0.42, (0.10, 0, 0.29), mat=mat_pants)
l_foot_obj = create_box("Mesh_L_Foot", (0.09, 0.20, 0.08), (0.10, 0.06, 0.04), mat=mat_shoes)

r_thigh_obj = create_cylinder("Mesh_R_Thigh", 0.08, 0.065, 0.40, (-0.10, 0, 0.70), mat=mat_pants)
r_shin_obj = create_cylinder("Mesh_R_Shin", 0.065, 0.055, 0.42, (-0.10, 0, 0.29), mat=mat_pants)
r_foot_obj = create_box("Mesh_R_Foot", (0.09, 0.20, 0.08), (-0.10, 0.06, 0.04), mat=mat_shoes)

# Left Arm
l_sleeve_obj = create_cylinder("Mesh_L_Sleeve", 0.07, 0.065, 0.14, (0.24, 0, 1.40), (0, math.radians(90), 0), mat=mat_shirt)
l_uarm_obj = create_cylinder("Mesh_L_UpperArm", 0.055, 0.05, 0.24, (0.30, 0, 1.40), (0, math.radians(90), 0), mat=mat_skin)
l_farm_obj = create_cylinder("Mesh_L_ForeArm", 0.05, 0.04, 0.24, (0.54, 0, 1.40), (0, math.radians(90), 0), mat=mat_skin)
l_palm_obj = create_box("Mesh_L_Palm", (0.09, 0.075, 0.026), (0.705, 0, 1.40), mat=mat_skin)

# Left Fingers (Thumb + 4 fingers x 3 segments)
# Thumb
create_cylinder("Mesh_L_T1", 0.012, 0.011, 0.03, (0.705, 0.04, 1.40), (math.radians(45), 0, math.radians(30)), mat_skin)
create_cylinder("Mesh_L_T2", 0.011, 0.010, 0.025, (0.73, 0.06, 1.40), (math.radians(45), 0, math.radians(30)), mat_skin)
create_cylinder("Mesh_L_T3", 0.010, 0.009, 0.02, (0.75, 0.075, 1.40), (math.radians(45), 0, math.radians(30)), mat_skin)

# 4 Fingers (Index, Middle, Ring, Pinky)
finger_y_offsets = [0.025, 0.008, -0.008, -0.025]
finger_names = ["Index", "Middle", "Ring", "Pinky"]
for idx, (fy, fname) in enumerate(zip(finger_y_offsets, finger_names)):
    create_cylinder(f"Mesh_L_{fname}1", 0.011, 0.010, 0.03, (0.765, fy, 1.40), (0, math.radians(90), 0), mat_skin)
    create_cylinder(f"Mesh_L_{fname}2", 0.010, 0.009, 0.025, (0.795, fy, 1.40), (0, math.radians(90), 0), mat_skin)
    create_cylinder(f"Mesh_L_{fname}3", 0.009, 0.008, 0.02, (0.815, fy, 1.40), (0, math.radians(90), 0), mat_skin)

# Right Arm
r_sleeve_obj = create_cylinder("Mesh_R_Sleeve", 0.07, 0.065, 0.14, (-0.24, 0, 1.40), (0, math.radians(-90), 0), mat=mat_shirt)
r_uarm_obj = create_cylinder("Mesh_R_UpperArm", 0.055, 0.05, 0.24, (-0.30, 0, 1.40), (0, math.radians(-90), 0), mat=mat_skin)
r_farm_obj = create_cylinder("Mesh_R_ForeArm", 0.05, 0.04, 0.24, (-0.54, 0, 1.40), (0, math.radians(-90), 0), mat_skin)
r_palm_obj = create_box("Mesh_R_Palm", (0.09, 0.075, 0.026), (-0.705, 0, 1.40), mat=mat_skin)

# Right Fingers
create_cylinder("Mesh_R_T1", 0.012, 0.011, 0.03, (-0.705, 0.04, 1.40), (math.radians(45), 0, math.radians(-30)), mat_skin)
create_cylinder("Mesh_R_T2", 0.011, 0.010, 0.025, (-0.73, 0.06, 1.40), (math.radians(45), 0, math.radians(-30)), mat_skin)
create_cylinder("Mesh_R_T3", 0.010, 0.009, 0.02, (-0.75, 0.075, 1.40), (math.radians(45), 0, math.radians(-30)), mat_skin)

for idx, (fy, fname) in enumerate(zip(finger_y_offsets, finger_names)):
    create_cylinder(f"Mesh_R_{fname}1", 0.011, 0.010, 0.03, (-0.765, fy, 1.40), (0, math.radians(-90), 0), mat_skin)
    create_cylinder(f"Mesh_R_{fname}2", 0.010, 0.009, 0.025, (-0.795, fy, 1.40), (0, math.radians(-90), 0), mat_skin)
    create_cylinder(f"Mesh_R_{fname}3", 0.009, 0.008, 0.02, (-0.815, fy, 1.40), (0, math.radians(-90), 0), mat_skin)

# Join All Mesh Parts into Single Production Mesh
bpy.ops.object.select_all(action='DESELECT')
for p in mesh_parts:
    p.select_set(True)
bpy.context.view_layer.objects.active = mesh_parts[0]
bpy.ops.object.join()
avatar_mesh = bpy.context.active_object
avatar_mesh.name = "SBV_Signing_Avatar_Mesh"

# Move mesh to col_mesh
for c in avatar_mesh.users_collection:
    c.objects.unlink(avatar_mesh)
col_mesh.objects.link(avatar_mesh)

# 6. Automatic Weighting & Bone Weight Vertex Groups Assignment
avatar_mesh.parent = armature_obj
mod = avatar_mesh.modifiers.new(name="Armature", type='ARMATURE')
mod.object = armature_obj
mod.use_vertex_groups = True

# Create Vertex Groups for each bone and assign weights based on vertex spatial proximity
bone_names = [b.name for b in armature_data.bones]
for bname in bone_names:
    avatar_mesh.vertex_groups.new(name=bname)

vgroup_map = {vg.name: vg for vg in avatar_mesh.vertex_groups}

# Compute Spatial Weights
mesh_data = avatar_mesh.data
for v in mesh_data.vertices:
    co = v.co
    x, y, z = co.x, co.y, co.z
    
    # Head & Neck
    if z >= 1.58:
        vgroup_map["head"].add([v.index], 1.0, 'REPLACE')
    elif z >= 1.48:
        vgroup_map["neck"].add([v.index], 1.0, 'REPLACE')
    elif z >= 1.34:
        vgroup_map["chest"].add([v.index], 0.8, 'REPLACE')
        vgroup_map["spine_02"].add([v.index], 0.2, 'REPLACE')
    elif z >= 1.20:
        vgroup_map["spine_01"].add([v.index], 0.7, 'REPLACE')
        vgroup_map["spine_02"].add([v.index], 0.3, 'REPLACE')
    elif z >= 1.02:
        vgroup_map["spine"].add([v.index], 0.8, 'REPLACE')
        vgroup_map["pelvis"].add([v.index], 0.2, 'REPLACE')
    elif z >= 0.88:
        vgroup_map["pelvis"].add([v.index], 1.0, 'REPLACE')
    
    # Left Arm & Hand
    if x > 0.15 and z >= 1.25:
        if x < 0.28:
            vgroup_map["left_shoulder"].add([v.index], 1.0, 'REPLACE')
        elif x < 0.45:
            vgroup_map["left_upper_arm"].add([v.index], 1.0, 'REPLACE')
        elif x < 0.65:
            vgroup_map["left_forearm"].add([v.index], 1.0, 'REPLACE')
        elif x < 0.74:
            vgroup_map["left_hand"].add([v.index], 1.0, 'REPLACE')
        else:
            # Finger assignment based on y offset
            if y > 0.04:
                if x < 0.72: vgroup_map["left_thumb_01"].add([v.index], 1.0, 'REPLACE')
                elif x < 0.74: vgroup_map["left_thumb_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["left_thumb_03"].add([v.index], 1.0, 'REPLACE')
            elif y > 0.015:
                if x < 0.78: vgroup_map["left_index_01"].add([v.index], 1.0, 'REPLACE')
                elif x < 0.805: vgroup_map["left_index_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["left_index_03"].add([v.index], 1.0, 'REPLACE')
            elif y > -0.002:
                if x < 0.785: vgroup_map["left_middle_01"].add([v.index], 1.0, 'REPLACE')
                elif x < 0.815: vgroup_map["left_middle_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["left_middle_03"].add([v.index], 1.0, 'REPLACE')
            elif y > -0.018:
                if x < 0.78: vgroup_map["left_ring_01"].add([v.index], 1.0, 'REPLACE')
                elif x < 0.805: vgroup_map["left_ring_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["left_ring_03"].add([v.index], 1.0, 'REPLACE')
            else:
                if x < 0.775: vgroup_map["left_pinky_01"].add([v.index], 1.0, 'REPLACE')
                elif x < 0.795: vgroup_map["left_pinky_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["left_pinky_03"].add([v.index], 1.0, 'REPLACE')

    # Right Arm & Hand
    if x < -0.15 and z >= 1.25:
        if x > -0.28:
            vgroup_map["right_shoulder"].add([v.index], 1.0, 'REPLACE')
        elif x > -0.45:
            vgroup_map["right_upper_arm"].add([v.index], 1.0, 'REPLACE')
        elif x > -0.65:
            vgroup_map["right_forearm"].add([v.index], 1.0, 'REPLACE')
        elif x > -0.74:
            vgroup_map["right_hand"].add([v.index], 1.0, 'REPLACE')
        else:
            if y > 0.04:
                if x > -0.72: vgroup_map["right_thumb_01"].add([v.index], 1.0, 'REPLACE')
                elif x > -0.74: vgroup_map["right_thumb_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["right_thumb_03"].add([v.index], 1.0, 'REPLACE')
            elif y > 0.015:
                if x > -0.78: vgroup_map["right_index_01"].add([v.index], 1.0, 'REPLACE')
                elif x > -0.805: vgroup_map["right_index_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["right_index_03"].add([v.index], 1.0, 'REPLACE')
            elif y > -0.002:
                if x > -0.785: vgroup_map["right_middle_01"].add([v.index], 1.0, 'REPLACE')
                elif x > -0.815: vgroup_map["right_middle_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["right_middle_03"].add([v.index], 1.0, 'REPLACE')
            elif y > -0.018:
                if x > -0.78: vgroup_map["right_ring_01"].add([v.index], 1.0, 'REPLACE')
                elif x > -0.805: vgroup_map["right_ring_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["right_ring_03"].add([v.index], 1.0, 'REPLACE')
            else:
                if x > -0.775: vgroup_map["right_pinky_01"].add([v.index], 1.0, 'REPLACE')
                elif x > -0.795: vgroup_map["right_pinky_02"].add([v.index], 1.0, 'REPLACE')
                else: vgroup_map["right_pinky_03"].add([v.index], 1.0, 'REPLACE')

    # Legs
    if z < 0.88:
        if x >= 0:
            if z > 0.50: vgroup_map["left_thigh"].add([v.index], 1.0, 'REPLACE')
            elif z > 0.08: vgroup_map["left_shin"].add([v.index], 1.0, 'REPLACE')
            else: vgroup_map["left_foot"].add([v.index], 1.0, 'REPLACE')
        else:
            if z > 0.50: vgroup_map["right_thigh"].add([v.index], 1.0, 'REPLACE')
            elif z > 0.08: vgroup_map["right_shin"].add([v.index], 1.0, 'REPLACE')
            else: vgroup_map["right_foot"].add([v.index], 1.0, 'REPLACE')

# 7. Lighting Setup
def add_light(name, light_type, energy, color, location):
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    light_data.color = color
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    light_obj.location = location
    col_lights.objects.link(light_obj)
    return light_obj

add_light("SBV_KeyLight", 'AREA', 600, (1.0, 0.95, 0.9), (1.5, -2.2, 2.2))
add_light("SBV_FillLight", 'AREA', 300, (0.7, 0.85, 1.0), (-1.8, -1.8, 1.8))
add_light("SBV_RimLight", 'SPOT', 800, (0.6, 0.85, 1.0), (0.0, 2.5, 2.4))

# 8. Cameras Setup
def add_camera(name, location, rotation_euler, lens=50):
    cam_data = bpy.data.cameras.new(name=name)
    cam_data.lens = lens
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)
    cam_obj.location = location
    cam_obj.rotation_euler = [math.radians(r) for r in rotation_euler]
    col_cameras.objects.link(cam_obj)
    return cam_obj

cam_main = add_camera("SBV_Camera", (0, -2.4, 1.35), (85, 0, 0), lens=45)
cam_wide = add_camera("SBV_Wide_Camera", (0, -3.8, 1.0), (87, 0, 0), lens=32)
scene.camera = cam_main

# 9. Keyframed ISL Animation Actions
# Helper to insert keyframes on pose bones
def create_action(action_name, keyframes_func, total_frames=40):
    act = bpy.data.actions.new(name=action_name)
    act.use_fake_user = True
    armature_obj.animation_data_create()
    armature_obj.animation_data.action = act
    
    # Reset all pose bones
    for pb in armature_obj.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
    
    keyframes_func(act, total_frames)
    return act

# Rest / Base Pose Keyframes
def apply_rest_keys(frame):
    for pb in armature_obj.pose.bones:
        pb.keyframe_insert(data_path='rotation_euler', frame=frame)
        pb.keyframe_insert(data_path='location', frame=frame)

# 1) SBV_HELLO
def anim_hello(act, total):
    pb = armature_obj.pose.bones
    # Frame 1: Rest
    apply_rest_keys(1)
    # Frame 15: Arm up, waving at temple
    pb["right_upper_arm"].rotation_euler = (math.radians(-80), math.radians(20), math.radians(45))
    pb["right_forearm"].rotation_euler = (math.radians(-70), math.radians(30), math.radians(20))
    pb["right_hand"].rotation_euler = (0, 0, math.radians(25))
    pb["head"].rotation_euler = (math.radians(-5), 0, 0)
    for bone_name in ["right_upper_arm", "right_forearm", "right_hand", "head"]:
        pb[bone_name].keyframe_insert(data_path='rotation_euler', frame=15)
    
    # Frame 25: Wave back
    pb["right_hand"].rotation_euler = (0, 0, math.radians(-25))
    pb["right_hand"].keyframe_insert(data_path='rotation_euler', frame=25)
    
    # Frame 40: Return to Rest
    apply_rest_keys(total)

create_action("SBV_HELLO", anim_hello, 40)

# 2) SBV_THANK_YOU
def anim_thankyou(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    # Hand to chin
    pb["right_upper_arm"].rotation_euler = (math.radians(-65), math.radians(10), math.radians(20))
    pb["right_forearm"].rotation_euler = (math.radians(-110), math.radians(15), math.radians(10))
    pb["right_hand"].rotation_euler = (math.radians(25), 0, 0)
    for b in ["right_upper_arm", "right_forearm", "right_hand"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=15)
    # Hand moves forward to viewer
    pb["right_upper_arm"].rotation_euler = (math.radians(-45), math.radians(10), math.radians(10))
    pb["right_forearm"].rotation_euler = (math.radians(-40), math.radians(10), math.radians(10))
    pb["right_hand"].rotation_euler = (math.radians(10), 0, 0)
    for b in ["right_upper_arm", "right_forearm", "right_hand"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=28)
    apply_rest_keys(total)

create_action("SBV_THANK_YOU", anim_thankyou, 40)

# 3) SBV_PLEASE
def anim_please(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    # Hand on chest rubbing circularly
    pb["right_upper_arm"].rotation_euler = (math.radians(-45), math.radians(25), math.radians(30))
    pb["right_forearm"].rotation_euler = (math.radians(-100), math.radians(20), math.radians(20))
    for b in ["right_upper_arm", "right_forearm"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=12)
        pb[b].keyframe_insert(data_path='rotation_euler', frame=26)
    apply_rest_keys(total)

create_action("SBV_PLEASE", anim_please, 38)

# 4) SBV_HELP
def anim_help(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    # Left hand flat, right fist on it lifting
    pb["left_upper_arm"].rotation_euler = (math.radians(-40), math.radians(-15), math.radians(20))
    pb["left_forearm"].rotation_euler = (math.radians(-80), math.radians(-10), math.radians(15))
    pb["right_upper_arm"].rotation_euler = (math.radians(-50), math.radians(15), math.radians(20))
    pb["right_forearm"].rotation_euler = (math.radians(-95), math.radians(15), math.radians(20))
    for b in ["left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=18)
    apply_rest_keys(total)

create_action("SBV_HELP", anim_help, 40)

# 5) SBV_WATER
def anim_water(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    # W-handshape at chin tapping
    pb["right_upper_arm"].rotation_euler = (math.radians(-70), math.radians(20), math.radians(20))
    pb["right_forearm"].rotation_euler = (math.radians(-115), math.radians(25), math.radians(20))
    for b in ["right_upper_arm", "right_forearm"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=14)
        pb[b].keyframe_insert(data_path='rotation_euler', frame=24)
    apply_rest_keys(total)

create_action("SBV_WATER", anim_water, 36)

# 6) SBV_FOOD
def anim_food(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    pb["right_upper_arm"].rotation_euler = (math.radians(-75), math.radians(15), math.radians(15))
    pb["right_forearm"].rotation_euler = (math.radians(-120), math.radians(20), math.radians(20))
    for b in ["right_upper_arm", "right_forearm"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=14)
        pb[b].keyframe_insert(data_path='rotation_euler', frame=24)
    apply_rest_keys(total)

create_action("SBV_FOOD", anim_food, 36)

# 7) SBV_WHERE
def anim_where(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    pb["left_upper_arm"].rotation_euler = (math.radians(-35), math.radians(-25), math.radians(-10))
    pb["left_forearm"].rotation_euler = (math.radians(-70), math.radians(-15), math.radians(-10))
    pb["right_upper_arm"].rotation_euler = (math.radians(-35), math.radians(25), math.radians(10))
    pb["right_forearm"].rotation_euler = (math.radians(-70), math.radians(15), math.radians(10))
    pb["head"].rotation_euler = (0, 0, math.radians(12))
    for b in ["left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm", "head"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=18)
    apply_rest_keys(total)

create_action("SBV_WHERE", anim_where, 40)

# 8) SBV_YES
def anim_yes(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    pb["right_upper_arm"].rotation_euler = (math.radians(-40), math.radians(15), math.radians(10))
    pb["right_forearm"].rotation_euler = (math.radians(-75), math.radians(15), math.radians(10))
    pb["right_hand"].rotation_euler = (math.radians(35), 0, 0)
    pb["head"].rotation_euler = (math.radians(15), 0, 0)
    for b in ["right_upper_arm", "right_forearm", "right_hand", "head"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=12)
        pb[b].keyframe_insert(data_path='rotation_euler', frame=24)
    apply_rest_keys(total)

create_action("SBV_YES", anim_yes, 36)

# 9) SBV_NO
def anim_no(act, total):
    pb = armature_obj.pose.bones
    apply_rest_keys(1)
    pb["right_upper_arm"].rotation_euler = (math.radians(-45), math.radians(20), math.radians(15))
    pb["right_forearm"].rotation_euler = (math.radians(-85), math.radians(20), math.radians(15))
    pb["head"].rotation_euler = (0, math.radians(20), 0)
    for b in ["right_upper_arm", "right_forearm", "head"]:
        pb[b].keyframe_insert(data_path='rotation_euler', frame=12)
    pb["head"].rotation_euler = (0, math.radians(-20), 0)
    pb["head"].keyframe_insert(data_path='rotation_euler', frame=22)
    apply_rest_keys(total)

create_action("SBV_NO", anim_no, 36)

# 10) SBV_DEMO_SEQUENCE (Combined full test sequence: HELLO -> THANK YOU -> WATER -> HELP)
def anim_demo(act, total):
    apply_rest_keys(1)
    anim_hello(act, 35)
    anim_thankyou(act, 70)
    anim_water(act, 105)
    anim_help(act, 140)
    apply_rest_keys(total)

create_action("SBV_DEMO_SEQUENCE", anim_demo, 150)

# Set Default Active Action to SBV_HELLO in a clear signing pose
armature_obj.animation_data.action = bpy.data.actions.get("SBV_HELLO")
scene.frame_set(15)

# 10. Save .blend & Export .glb
out_dir = r"C:\Users\drx20\.gemini\antigravity\scratch\signbridge2vision\apps\web\public\models\sign-avatar"
os.makedirs(out_dir, exist_ok=True)

blend_path = os.path.join(out_dir, "sign-avatar.blend")
glb_path = os.path.join(out_dir, "sign-avatar.glb")

bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"Saved Blender Scene to: {blend_path}")

bpy.ops.export_scene.gltf(
    filepath=glb_path,
    export_format='GLB',
    export_apply=True,
    export_animations=True,
    export_all_influences=True,
    export_def_bones=True,
    export_skins=True,
    export_morph=True,
    export_lights=True,
    export_cameras=True,
)
print(f"Exported Production GLB to: {glb_path}")

print("=" * 60)
print("PRODUCTION SIGNING AVATAR GENERATION COMPLETE!")
print(f"Bones: {total_bones} | Hand Bones: {hand_bone_count}")
print(f"Actions Created: {len(bpy.data.actions)}")
print("=" * 60)
