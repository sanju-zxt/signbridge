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

import { AnimationController } from '../apps/web/src/components/avatar/AnimationController';

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

  const controller = new AnimationController(scene);
  controller.registerClips(gltf.animations);

  const status = controller.getStatus();
  console.log(`Loaded ${status.loadedClips.length} clips in controller.`);

  const proceduralClips = status.loadedClips.filter((c: string) => c.startsWith('PROC_'));
  console.log(`\nEvaluating ${proceduralClips.length} procedural clips:`);

  for (const pClipName of proceduralClips) {
    const action = (controller as any).actions.get(pClipName);
    if (!action) continue;

    action.reset();
    action.play();
    (controller as any).mixer.setTime(1.0); // mid-point
    scene.updateMatrixWorld(true);

    const rWrist = new THREE.Vector3();
    const lWrist = new THREE.Vector3();
    bones['RightHand'].getWorldPosition(rWrist);
    bones['LeftHand'].getWorldPosition(lWrist);

    const isBackward = rWrist.z < -0.05 || lWrist.z < -0.05;
    const mark = isBackward ? '❌ BACKWARD' : '✓ OK';
    console.log(`${mark.padEnd(12)} ${pClipName.padEnd(20)}: RightWrist=[${rWrist.x.toFixed(2)}, ${rWrist.y.toFixed(2)}, ${rWrist.z.toFixed(2)}]  LeftWrist=[${lWrist.x.toFixed(2)}, ${lWrist.y.toFixed(2)}, ${lWrist.z.toFixed(2)}]`);

    action.stop();
  }
});
