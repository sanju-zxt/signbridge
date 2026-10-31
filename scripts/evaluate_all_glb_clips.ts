import * as fs from 'fs';
import * as path from 'path';
import * as THREE from 'three';
// @ts-ignore
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

(global as any).self = global;
(global as any).window = global;
if (typeof (global as any).ProgressEvent === 'undefined') {
  (global as any).ProgressEvent = class ProgressEvent {
    lengthComputable = false;
    loaded = 0;
    total = 0;
  };
}
if (typeof (global as any).createImageBitmap === 'undefined') {
  (global as any).createImageBitmap = async () => ({
    width: 1,
    height: 1,
    close: () => {},
  });
}

const glbPath = path.resolve('apps/web/public/models/sign-avatar/sign-avatar.glb');
const buffer = fs.readFileSync(glbPath);
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

const loader = new GLTFLoader();
loader.parse(arrayBuffer, '', (gltf: any) => {
  const scene = gltf.scene;
  const cleanName = (n: string) => n.replace(/^([a-zA-Z0-9_-]+:)?mixamorig\d*:?/i, '').replace(/^[a-zA-Z0-9_-]+:/, '');

  const bones: Record<string, THREE.Bone> = {};
  scene.traverse((obj: any) => {
    if (obj.isBone) {
      bones[cleanName(obj.name)] = obj;
    }
  });

  const mixer = new THREE.AnimationMixer(scene);

  console.log('=== EVALUATING ALL GLB CLIPS AT MIDPOINT ===');
  for (const anim of gltf.animations) {
    const act = mixer.clipAction(anim);
    act.play();
    mixer.setTime(anim.duration * 0.5);
    scene.updateMatrixWorld(true);

    const lWrist = new THREE.Vector3();
    const rWrist = new THREE.Vector3();
    bones['LeftHand'].getWorldPosition(lWrist);
    bones['RightHand'].getWorldPosition(rWrist);

    // If wrist is behind the back: z < -0.05
    if (lWrist.z < -0.05 || rWrist.z < -0.05) {
      console.log(`⚠️  ${anim.name.padEnd(20)}: LeftWrist=[${lWrist.x.toFixed(2)}, ${lWrist.y.toFixed(2)}, ${lWrist.z.toFixed(2)}]  RightWrist=[${rWrist.x.toFixed(2)}, ${rWrist.y.toFixed(2)}, ${rWrist.z.toFixed(2)}]`);
    } else {
      console.log(`✓  ${anim.name.padEnd(20)}: LeftWrist=[${lWrist.x.toFixed(2)}, ${lWrist.y.toFixed(2)}, ${lWrist.z.toFixed(2)}]  RightWrist=[${rWrist.x.toFixed(2)}, ${rWrist.y.toFixed(2)}, ${rWrist.z.toFixed(2)}]`);
    }
    act.stop();
  }
});
