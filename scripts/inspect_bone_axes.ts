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

  const printBonesAxes = (name: string) => {
    const b = bones[name];
    scene.updateMatrixWorld(true);
    const pos = new THREE.Vector3();
    b.getWorldPosition(pos);
    const rot = new THREE.Quaternion();
    b.getWorldQuaternion(rot);

    const xDir = new THREE.Vector3(1, 0, 0).applyQuaternion(rot);
    const yDir = new THREE.Vector3(0, 1, 0).applyQuaternion(rot);
    const zDir = new THREE.Vector3(0, 0, 1).applyQuaternion(rot);

    console.log(`Bone: ${name}`);
    console.log(`  World Pos: [${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}]`);
    console.log(`  World +X:  [${xDir.x.toFixed(2)}, ${xDir.y.toFixed(2)}, ${xDir.z.toFixed(2)}]`);
    console.log(`  World +Y:  [${yDir.x.toFixed(2)}, ${yDir.y.toFixed(2)}, ${yDir.z.toFixed(2)}] (bone vector)`);
    console.log(`  World +Z:  [${zDir.x.toFixed(2)}, ${zDir.y.toFixed(2)}, ${zDir.z.toFixed(2)}]`);
  };

  // Check in native SBV_HELP pose where the arm is nicely in front of chest!
  const helpClip = gltf.animations.find((a: any) => a.name === 'SBV_HELP');
  const mixer = new THREE.AnimationMixer(scene);
  const act = mixer.clipAction(helpClip);
  act.play();
  mixer.setTime(1.5);
  scene.updateMatrixWorld(true);

  console.log('=== In SBV_HELP Pose (t=1.5s) ===');
  printBonesAxes('RightArm');
  printBonesAxes('RightForeArm');
  printBonesAxes('RightHand');

  console.log('\n=== In SBV_IDLE Pose ===');
  act.stop();
  const idleClip = gltf.animations.find((a: any) => a.name === 'SBV_IDLE');
  const actIdle = mixer.clipAction(idleClip);
  actIdle.play();
  mixer.setTime(1.0);
  scene.updateMatrixWorld(true);
  printBonesAxes('RightArm');
  printBonesAxes('RightForeArm');
  printBonesAxes('RightHand');
});
