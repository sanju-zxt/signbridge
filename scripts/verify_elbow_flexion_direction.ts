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

  const rArm = bones['RightArm'];
  const rFore = bones['RightForeArm'];
  const rHand = bones['RightHand'];

  // Test 1: Arm down at side (Rest Pose)
  rArm.quaternion.set(0.3932, -0.2587, -0.0417, 0.8813);

  // Compare positive X vs negative X on ForeArm:
  for (const angleDeg of [-60, 0, 60, 90]) {
    rFore.quaternion.setFromEuler(new THREE.Euler(THREE.MathUtils.degToRad(angleDeg), 0, 0, 'XYZ'));
    scene.updateMatrixWorld(true);

    const elbow = new THREE.Vector3();
    const wrist = new THREE.Vector3();
    rFore.getWorldPosition(elbow);
    rHand.getWorldPosition(wrist);

    console.log(`ForeArm Euler X = ${angleDeg > 0 ? '+' : ''}${angleDeg}°:`);
    console.log(`  Elbow: [${elbow.x.toFixed(2)}, ${elbow.y.toFixed(2)}, ${elbow.z.toFixed(2)}]`);
    console.log(`  Wrist: [${wrist.x.toFixed(2)}, ${wrist.y.toFixed(2)}, ${wrist.z.toFixed(2)}] -> deltaZ = ${(wrist.z - elbow.z).toFixed(2)} (positive = forward)`);
    console.log(`  Quat:  [${rFore.quaternion.x.toFixed(4)}, ${rFore.quaternion.y.toFixed(4)}, ${rFore.quaternion.z.toFixed(4)}, ${rFore.quaternion.w.toFixed(4)}]`);
  }
});
