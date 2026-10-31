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

  const helloClip = gltf.animations.find((a: any) => a.name === 'SBV_HELLO');
  const mixer = new THREE.AnimationMixer(scene);
  const act = mixer.clipAction(helloClip);
  act.play();

  console.log('=== Checking Left Arm across SBV_HELLO ===');
  for (let t = 0; t <= helloClip.duration; t += 1.0) {
    mixer.setTime(t);
    scene.updateMatrixWorld(true);

    const lShoulder = new THREE.Vector3();
    const lElbow = new THREE.Vector3();
    const lWrist = new THREE.Vector3();

    bones['LeftShoulder'].getWorldPosition(lShoulder);
    bones['LeftForeArm'].getWorldPosition(lElbow);
    bones['LeftHand'].getWorldPosition(lWrist);

    console.log(`t=${t.toFixed(1)}s:`);
    console.log(`  Left Elbow: [${lElbow.x.toFixed(2)}, ${lElbow.y.toFixed(2)}, ${lElbow.z.toFixed(2)}]`);
    console.log(`  Left Wrist: [${lWrist.x.toFixed(2)}, ${lWrist.y.toFixed(2)}, ${lWrist.z.toFixed(2)}]`);
  }
});
