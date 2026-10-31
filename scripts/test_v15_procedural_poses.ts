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

  function testPose(
    signName: string,
    rArmQ: number[],
    rForeQ: number[],
    rHandQ: number[],
    lArmQ: number[] = REST_L_ARM,
    lForeQ: number[] = REST_L_FOREARM,
    lHandQ: number[] = REST_L_HAND
  ) {
    bones['RightArm'].quaternion.fromArray(rArmQ);
    bones['RightForeArm'].quaternion.fromArray(rForeQ);
    bones['RightHand'].quaternion.fromArray(rHandQ);

    bones['LeftArm'].quaternion.fromArray(lArmQ);
    bones['LeftForeArm'].quaternion.fromArray(lForeQ);
    bones['LeftHand'].quaternion.fromArray(lHandQ);

    scene.updateMatrixWorld(true);

    const rElbow = new THREE.Vector3();
    const rWrist = new THREE.Vector3();
    const rTip = new THREE.Vector3();
    const lElbow = new THREE.Vector3();
    const lWrist = new THREE.Vector3();

    bones['RightForeArm'].getWorldPosition(rElbow);
    bones['RightHand'].getWorldPosition(rWrist);
    if (bones['RightHandIndex3']) bones['RightHandIndex3'].getWorldPosition(rTip);
    bones['LeftForeArm'].getWorldPosition(lElbow);
    bones['LeftHand'].getWorldPosition(lWrist);

    // Compute elbow flexion angle
    const rUpperArmVec = new THREE.Vector3().subVectors(rElbow, bones['RightArm'].getWorldPosition(new THREE.Vector3())).normalize();
    const rForeArmVec = new THREE.Vector3().subVectors(rWrist, rElbow).normalize();
    const rElbowFlexDeg = 180 - THREE.MathUtils.radToDeg(rUpperArmVec.angleTo(rForeArmVec));

    // Calculate twist angle on wrist (Euler Z or Y)
    const wristEuler = new THREE.Euler().setFromQuaternion(bones['RightHand'].quaternion, 'XYZ');
    const wristRollDeg = THREE.MathUtils.radToDeg(wristEuler.z);

    const isValid = rWrist.z > 0.05 && lWrist.z >= -0.05 && rElbowFlexDeg > 20 && rElbowFlexDeg < 150;

    console.log(`\n=== POSE: ${signName} [${isValid ? 'PASSED ✓' : 'FAILED ❌'}] ===`);
    console.log(`  Right Elbow: [${rElbow.x.toFixed(2)}, ${rElbow.y.toFixed(2)}, ${rElbow.z.toFixed(2)}]  (flexion=${rElbowFlexDeg.toFixed(1)}°)`);
    console.log(`  Right Wrist: [${rWrist.x.toFixed(2)}, ${rWrist.y.toFixed(2)}, ${rWrist.z.toFixed(2)}]  (roll=${wristRollDeg.toFixed(1)}°)`);
    console.log(`  Right Tip:   [${rTip.x.toFixed(2)}, ${rTip.y.toFixed(2)}, ${rTip.z.toFixed(2)}]`);
    console.log(`  Left  Elbow: [${lElbow.x.toFixed(2)}, ${lElbow.y.toFixed(2)}, ${lElbow.z.toFixed(2)}]`);
    console.log(`  Left  Wrist: [${lWrist.x.toFixed(2)}, ${lWrist.y.toFixed(2)}, ${lWrist.z.toFixed(2)}]`);

    return isValid;
  }

  // Test Proposed Quaternions for Target Signs:

  // 1. I_ME: Pointing directly to sternum (center chest)
  testPose('I_ME',
    [0.28, 0.20, -0.20, 0.92],
    [0.72, 0.12, 0.05, 0.68],
    [0.15, 0.05, 0.15, 0.97]
  );

  // 2. MY: Flat palm placed on sternum
  testPose('MY',
    [0.30, 0.18, -0.22, 0.91],
    [0.68, 0.15, -0.01, 0.72],
    [0.15, 0.10, 0.20, 0.96]
  );

  // 3. YOU: Index pointing forward toward partner
  testPose('YOU',
    [0.25, 0.10, -0.15, 0.95],
    [0.45, 0.05, -0.05, 0.89],
    [0.08, 0.00, 0.02, 0.99]
  );

  // 4. YOUR: Flat open palm pushing forward toward partner
  testPose('YOUR',
    [0.25, 0.10, -0.15, 0.95],
    [0.45, 0.05, -0.05, 0.89],
    [0.22, 0.05, -0.15, 0.96]
  );

  // 5. WHAT: Both hands forward, palms up, gentle inquiry
  testPose('WHAT',
    [0.25, -0.05, -0.15, 0.96],
    [0.60, 0.12, -0.02, 0.79],
    [-0.08, -0.02, -0.17, 0.98],
    [0.25, 0.05, 0.15, 0.96],
    [0.60, -0.12, 0.02, 0.79],
    [-0.08, 0.02, 0.17, 0.98]
  );

  // 6. HELP: Left palm flat up, Right hand resting on left, lifting
  testPose('HELP',
    [0.20, 0.09, -0.15, 0.96],
    [0.68, 0.17, -0.06, 0.71],
    [0.10, 0.05, 0.05, 0.99],
    [0.26, -0.17, 0.09, 0.95],
    [0.68, -0.19, 0.10, 0.70],
    [-0.05, 0.15, 0.25, 0.95]
  );

  // 7. NAME: Both hands tapping at chest level
  testPose('NAME',
    [0.24, 0.08, -0.15, 0.96],
    [0.64, 0.15, -0.07, 0.75],
    [0.05, -0.10, -0.20, 0.97],
    [0.28, -0.12, 0.10, 0.94],
    [0.62, -0.15, 0.08, 0.76],
    [0.05, 0.10, 0.20, 0.97]
  );
});
