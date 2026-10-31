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

  const testProceduralPose = (name: string, rArm: number[], rForeArm: number[], rHand: number[], lArm?: number[], lForeArm?: number[], lHand?: number[]) => {
    bones['RightArm'].quaternion.fromArray(rArm);
    bones['RightForeArm'].quaternion.fromArray(rForeArm);
    bones['RightHand'].quaternion.fromArray(rHand);

    if (lArm) bones['LeftArm'].quaternion.fromArray(lArm);
    if (lForeArm) bones['LeftForeArm'].quaternion.fromArray(lForeArm);
    if (lHand) bones['LeftHand'].quaternion.fromArray(lHand);

    scene.updateMatrixWorld(true);

    const rElbow = new THREE.Vector3();
    const rWrist = new THREE.Vector3();
    const rIndex = new THREE.Vector3();
    const lElbow = new THREE.Vector3();
    const lWrist = new THREE.Vector3();

    bones['RightForeArm'].getWorldPosition(rElbow);
    bones['RightHand'].getWorldPosition(rWrist);
    if (bones['RightHandIndex3']) bones['RightHandIndex3'].getWorldPosition(rIndex);
    bones['LeftForeArm'].getWorldPosition(lElbow);
    bones['LeftHand'].getWorldPosition(lWrist);

    const rUpperArmVec = new THREE.Vector3().subVectors(rElbow, bones['RightArm'].getWorldPosition(new THREE.Vector3())).normalize();
    const rForeArmVec = new THREE.Vector3().subVectors(rWrist, rElbow).normalize();
    const rElbowFlexDeg = 180 - THREE.MathUtils.radToDeg(rUpperArmVec.angleTo(rForeArmVec));

    console.log(`\n--- Procedural Pose: ${name} ---`);
    console.log(`  Right Elbow: x=${rElbow.x.toFixed(2)}, y=${rElbow.y.toFixed(2)}, z=${rElbow.z.toFixed(2)}`);
    console.log(`  Right Wrist: x=${rWrist.x.toFixed(2)}, y=${rWrist.y.toFixed(2)}, z=${rWrist.z.toFixed(2)} (Elbow flex=${rElbowFlexDeg.toFixed(1)}°)`);
    if (bones['RightHandIndex3']) {
      console.log(`  Right Index Tip: x=${rIndex.x.toFixed(2)}, y=${rIndex.y.toFixed(2)}, z=${rIndex.z.toFixed(2)}`);
    }
    console.log(`  Left  Elbow: x=${lElbow.x.toFixed(2)}, y=${lElbow.y.toFixed(2)}, z=${lElbow.z.toFixed(2)}`);
    console.log(`  Left  Wrist: x=${lWrist.x.toFixed(2)}, y=${lWrist.y.toFixed(2)}, z=${lWrist.z.toFixed(2)}`);
  };

  const REST_R_ARM = [0.3932, -0.2587, -0.0417, 0.8813];
  const REST_R_FOREARM = [0.016, 0.0021, -0.152, 0.9882];
  const REST_R_HAND = [0.023, 0.2889, 0.1613, 0.9434];

  const REST_L_ARM = [0.4763, 0.1736, 0.0317, 0.8614];
  const REST_L_FOREARM = [0.0127, -0.0021, 0.1213, 0.9925];
  const REST_L_HAND = [-0.001, -0.0707, -0.1795, 0.9812];

  // Test current I_ME
  testProceduralPose('Current I_ME',
    [-0.3, 0.4, -0.3, 0.86],
    [-0.52, 0.58, 0.1, 0.61],
    [-0.3, 0.15, 0.0, 0.94],
    REST_L_ARM, REST_L_FOREARM, REST_L_HAND
  );

  // Test current MY
  testProceduralPose('Current MY',
    [-0.35, 0.35, -0.2, 0.85],
    [-0.42, 0.5, 0.2, 0.74],
    [0.0, -0.3, 0.0, 0.95],
    REST_L_ARM, REST_L_FOREARM, REST_L_HAND
  );

  // Test current YOU
  testProceduralPose('Current YOU',
    [-0.48, 0.12, 0.1, 0.86],
    [0.0, 0.32, 0.0, 0.94],
    [0.0, 0.15, 0.0, 0.98],
    REST_L_ARM, REST_L_FOREARM, REST_L_HAND
  );

  // Test current YOUR
  testProceduralPose('Current YOUR',
    [-0.52, 0.1, 0.0, 0.85],
    [-0.1, 0.1, -0.1, 0.98],
    REST_R_HAND,
    REST_L_ARM, REST_L_FOREARM, REST_L_HAND
  );

  // Test current WHAT
  testProceduralPose('Current WHAT',
    [-0.38, 0.3, 0.15, 0.86],
    [-0.25, 0.1, -0.35, 0.86],
    REST_R_HAND,
    [-0.38, -0.3, -0.15, 0.86],
    [-0.25, -0.1, 0.35, 0.86],
    REST_L_HAND
  );
});
