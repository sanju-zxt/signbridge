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

  const REST_R_ARM = [0.3932, -0.2587, -0.0417, 0.8813];
  const REST_R_FOREARM = [0.016, 0.0021, -0.152, 0.9882];
  const REST_R_HAND = [0.023, 0.2889, 0.1613, 0.9434];

  const REST_L_ARM = [0.4763, 0.1736, 0.0317, 0.8614];
  const REST_L_FOREARM = [0.0127, -0.0021, 0.1213, 0.9925];
  const REST_L_HAND = [-0.001, -0.0707, -0.1795, 0.9812];

  let passedCount = 0;
  let totalCount = 0;

  function verifyPose(
    name: string,
    rArm: number[],
    rFore: number[],
    rHand: number[],
    lArm: number[] = REST_L_ARM,
    lFore: number[] = REST_L_FOREARM,
    lHand: number[] = REST_L_HAND
  ) {
    totalCount++;
    bones['RightArm'].quaternion.fromArray(rArm);
    bones['RightForeArm'].quaternion.fromArray(rFore);
    bones['RightHand'].quaternion.fromArray(rHand);
    bones['LeftArm'].quaternion.fromArray(lArm);
    bones['LeftForeArm'].quaternion.fromArray(lFore);
    bones['LeftHand'].quaternion.fromArray(lHand);

    scene.updateMatrixWorld(true);

    const rElbow = new THREE.Vector3();
    const rWrist = new THREE.Vector3();
    const lElbow = new THREE.Vector3();
    const lWrist = new THREE.Vector3();

    bones['RightForeArm'].getWorldPosition(rElbow);
    bones['RightHand'].getWorldPosition(rWrist);
    bones['LeftForeArm'].getWorldPosition(lElbow);
    bones['LeftHand'].getWorldPosition(lWrist);

    const rUpperArmVec = new THREE.Vector3().subVectors(rElbow, bones['RightArm'].getWorldPosition(new THREE.Vector3())).normalize();
    const rForeArmVec = new THREE.Vector3().subVectors(rWrist, rElbow).normalize();
    const rElbowFlexDeg = 180 - THREE.MathUtils.radToDeg(rUpperArmVec.angleTo(rForeArmVec));

    const wristEuler = new THREE.Euler().setFromQuaternion(bones['RightHand'].quaternion, 'XYZ');
    const wristRollDeg = THREE.MathUtils.radToDeg(wristEuler.z);

    // Criteria:
    // 1. Right wrist must be in front of torso (z >= 0.05)
    // 2. Left wrist must not be tucked unnaturally behind back (z >= -0.05)
    // 3. Elbow flexion must be anatomically valid (between 15° and 150°)
    // 4. Wrist twist roll must be natural (< 45°)
    const pass = rWrist.z >= 0.05 && lWrist.z >= -0.05 && rElbowFlexDeg >= 15 && rElbowFlexDeg <= 150 && Math.abs(wristRollDeg) <= 45;

    if (pass) {
      passedCount++;
      console.log(`[PASS] ${name.padEnd(16)}: rWrist=[${rWrist.x.toFixed(2)}, ${rWrist.y.toFixed(2)}, ${rWrist.z.toFixed(2)}] flex=${rElbowFlexDeg.toFixed(1)}° roll=${wristRollDeg.toFixed(1)}°`);
    } else {
      console.log(`[FAIL] ${name.padEnd(16)}: rWrist=[${rWrist.x.toFixed(2)}, ${rWrist.y.toFixed(2)}, ${rWrist.z.toFixed(2)}] lWrist=[${lWrist.x.toFixed(2)}, ${lWrist.y.toFixed(2)}, ${lWrist.z.toFixed(2)}] flex=${rElbowFlexDeg.toFixed(1)}° roll=${wristRollDeg.toFixed(1)}°`);
    }
  }

  console.log('=== V15 PROCEDURAL GESTURE POSE VERIFICATION ===');

  verifyPose('I_ME', [0.28, 0.20, -0.20, 0.92], [0.72, 0.12, 0.05, 0.68], [0.15, 0.05, 0.15, 0.97]);
  verifyPose('MY', [0.30, 0.18, -0.22, 0.91], [0.68, 0.15, -0.01, 0.72], [0.15, 0.10, 0.20, 0.96]);
  verifyPose('YOU', [0.25, 0.10, -0.15, 0.95], [0.45, 0.05, -0.05, 0.89], [0.08, 0.00, 0.02, 0.99]);
  verifyPose('YOUR', [0.25, 0.10, -0.15, 0.95], [0.45, 0.05, -0.05, 0.89], [0.22, 0.05, -0.15, 0.96]);
  verifyPose('WHAT', [0.25, -0.05, -0.15, 0.96], [0.60, 0.12, -0.02, 0.79], [-0.08, -0.02, -0.17, 0.98], [0.25, 0.05, 0.15, 0.96], [0.60, -0.12, 0.02, 0.79], [-0.08, 0.02, 0.17, 0.98]);
  verifyPose('HELP', [0.20, 0.09, -0.15, 0.96], [0.68, 0.17, -0.06, 0.71], [0.10, 0.05, 0.05, 0.99], [0.26, -0.17, 0.09, 0.95], [0.68, -0.19, 0.10, 0.70], [-0.05, 0.15, 0.25, 0.95]);
  verifyPose('NAME', [0.24, 0.08, -0.15, 0.96], [0.64, 0.15, -0.07, 0.75], [0.05, -0.10, -0.20, 0.97], [0.28, -0.12, 0.10, 0.94], [0.62, -0.15, 0.08, 0.76], [0.05, 0.10, 0.20, 0.97]);
  verifyPose('HOW', [0.22, 0.08, -0.12, 0.96], [0.62, 0.15, -0.05, 0.76], [0.00, 0.00, 0.00, 1.00], [0.22, -0.08, 0.12, 0.96], [0.62, -0.15, 0.05, 0.76], [0.00, 0.00, 0.00, 1.00]);
  verifyPose('HOW_ARE_YOU', [0.26, 0.10, -0.12, 0.95], [0.55, 0.10, -0.04, 0.83], [0.10, 0.05, 0.05, 0.99]);
  verifyPose('DOCTOR', [0.24, 0.10, -0.14, 0.96], [0.70, 0.16, -0.06, 0.69], [0.12, 0.05, 0.05, 0.99], [0.26, -0.15, 0.10, 0.95], [0.68, -0.18, 0.10, 0.70], [-0.05, 0.10, 0.18, 0.97]);
  verifyPose('WANT', [0.28, 0.12, -0.15, 0.94], [0.68, 0.15, -0.02, 0.71], [0.10, 0.05, 0.05, 0.99], [0.28, -0.12, 0.15, 0.94], [0.68, -0.15, 0.02, 0.71], [0.10, -0.05, -0.05, 0.99]);
  verifyPose('WORK', [0.22, 0.10, -0.12, 0.96], [0.68, 0.16, -0.05, 0.71], [0.05, -0.05, -0.05, 0.99], [0.24, -0.15, 0.08, 0.95], [0.65, -0.15, 0.08, 0.74], [0.05, 0.05, 0.05, 0.99]);
  verifyPose('CALL', [0.35, 0.25, -0.25, 0.86], [0.78, 0.20, 0.15, 0.57], [0.20, 0.15, 0.10, 0.96]);
  verifyPose('GIVE', [0.25, -0.02, -0.10, 0.96], [0.52, 0.08, -0.04, 0.85], [-0.05, -0.02, -0.15, 0.98], [0.25, 0.02, 0.10, 0.96], [0.52, -0.08, 0.04, 0.85], [-0.05, 0.02, 0.15, 0.98]);
  verifyPose('BAD', [0.22, 0.08, -0.12, 0.96], [0.58, 0.12, -0.08, 0.80], [0.15, 0.05, -0.25, 0.95]);
  verifyPose('DANGER', [0.28, 0.15, -0.18, 0.93], [0.72, 0.18, 0.12, 0.65], [0.05, 0.05, 0.05, 0.99], [0.28, -0.15, 0.18, 0.93], [0.72, -0.18, -0.12, 0.65], [0.05, -0.05, -0.05, 0.99]);
  verifyPose('PAIN', [0.24, 0.12, -0.14, 0.95], [0.68, 0.15, 0.05, 0.71], [0.05, 0.05, 0.05, 0.99], [0.24, -0.12, 0.14, 0.95], [0.68, -0.15, -0.05, 0.71], [0.05, -0.05, -0.05, 0.99]);
  verifyPose('MEDICINE', [0.22, 0.10, -0.12, 0.96], [0.68, 0.16, -0.04, 0.71], [0.10, 0.05, 0.05, 0.99], [0.25, -0.14, 0.08, 0.95], [0.65, -0.16, 0.06, 0.73], [-0.05, 0.10, 0.20, 0.97]);
  verifyPose('AMBULANCE', [0.35, 0.15, -0.20, 0.90], [0.65, 0.10, -0.05, 0.75], [0.10, 0.10, 0.10, 0.98], [0.35, -0.15, 0.20, 0.90], [0.65, -0.10, 0.05, 0.75], [0.10, -0.10, -0.10, 0.98]);
  verifyPose('FRIEND', [0.24, 0.10, -0.14, 0.96], [0.66, 0.15, -0.04, 0.73], [0.05, 0.05, 0.05, 0.99], [0.24, -0.10, 0.14, 0.96], [0.66, -0.15, 0.04, 0.73], [0.05, -0.05, -0.05, 0.99]);
  verifyPose('NICE_TO_MEET_YOU', [0.25, 0.10, -0.15, 0.95], [0.62, 0.12, -0.05, 0.77], [0.05, 0.05, 0.05, 0.99], [0.25, -0.10, 0.15, 0.95], [0.62, -0.12, 0.05, 0.77], [0.05, -0.05, -0.05, 0.99]);
  verifyPose('DEAF', [0.32, 0.22, -0.22, 0.89], [0.75, 0.18, 0.10, 0.62], [0.15, 0.10, 0.10, 0.98]);
  verifyPose('HEARING', [0.30, 0.20, -0.20, 0.91], [0.72, 0.16, 0.08, 0.66], [0.10, 0.05, 0.05, 0.99]);
  verifyPose('LEFT', [0.26, 0.22, -0.25, 0.90], [0.70, 0.25, 0.10, 0.65], [0.10, 0.15, 0.25, 0.95]);
  verifyPose('RIGHT', [0.24, -0.05, -0.10, 0.96], [0.60, 0.10, -0.05, 0.79], [0.05, -0.10, -0.10, 0.98]);
  verifyPose('HERE', [0.22, -0.04, -0.12, 0.96], [0.56, 0.10, -0.04, 0.82], [-0.10, -0.02, -0.18, 0.97], [0.22, 0.04, 0.12, 0.96], [0.56, -0.10, 0.04, 0.82], [-0.10, 0.02, 0.18, 0.97]);
  verifyPose('REPEAT', [0.24, 0.08, -0.14, 0.96], [0.60, 0.12, -0.05, 0.79], [0.05, 0.05, 0.05, 0.99]);
  verifyPose('NOW', [0.25, 0.05, -0.12, 0.96], [0.58, 0.10, -0.05, 0.80], [0.05, 0.05, 0.05, 0.99], [0.25, -0.05, 0.12, 0.96], [0.58, -0.10, 0.05, 0.80], [0.05, -0.05, -0.05, 0.99]);
  verifyPose('DO', [0.24, 0.06, -0.12, 0.96], [0.60, 0.10, -0.04, 0.79], [0.05, 0.05, 0.05, 0.99], [0.24, -0.06, 0.12, 0.96], [0.60, -0.10, 0.04, 0.79], [0.05, -0.05, -0.05, 0.99]);
  verifyPose('LETTER_SIGNING', [0.25, 0.10, -0.15, 0.95], [0.65, 0.15, -0.05, 0.74], [0.10, 0.00, 0.00, 0.99]);

  console.log(`\nVerification Score: ${passedCount} / ${totalCount} (${((passedCount / totalCount) * 100).toFixed(1)}%) Passed.`);
});
