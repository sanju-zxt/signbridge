import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"C:\Users\drx20\Downloads\3d model\Standing Greeting.fbx")
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
act = arm.animation_data.action

print("Layers count:", len(act.layers))
for layer in act.layers:
    print(f"Layer: {layer.name}, strips: {len(layer.strips)}")
    for strip in layer.strips:
        print("Strip attrs:", [a for a in dir(strip) if not a.startswith('__')])
        for cb in strip.channelbags:
            print("Channelbag attrs:", [a for a in dir(cb) if not a.startswith('__')])
            print("CB slot:", cb.slot)
            for fc in cb.fcurves[:5]:
                print(f"  fcurve data_path: {fc.data_path}")
