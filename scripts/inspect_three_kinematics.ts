import * as THREE from 'three';
import * as fs from 'fs';

// Let's create an Object3D hierarchy matching the GLB to inspect the local and world positions/vectors!
const RightShoulder = new THREE.Object3D();
RightShoulder.name = 'RightShoulder';
RightShoulder.quaternion.set(0.5385, -0.4559, 0.5507, 0.4460);

const RightArm = new THREE.Object3D();
RightArm.name = 'RightArm';
RightArm.position.set(0, 14.596, 0); // local trans from GLB
RightShoulder.add(RightArm);

const RightForeArm = new THREE.Object3D();
RightForeArm.name = 'RightForeArm';
RightForeArm.position.set(0, 26.106, 0);
RightArm.add(RightForeArm);

const RightHand = new THREE.Object3D();
RightHand.name = 'RightHand';
RightHand.position.set(0, 21.124, 0);
RightForeArm.add(RightHand);

// Left side
const LeftShoulder = new THREE.Object3D();
LeftShoulder.name = 'LeftShoulder';
LeftShoulder.quaternion.set(0.5394, 0.4551, -0.5499, 0.4466);

const LeftArm = new THREE.Object3D();
LeftArm.name = 'LeftArm';
LeftArm.position.set(0, 14.596, 0);
LeftShoulder.add(LeftArm);

const LeftForeArm = new THREE.Object3D();
LeftForeArm.name = 'LeftForeArm';
LeftForeArm.position.set(0, 26.107, 0);
LeftArm.add(LeftForeArm);

const LeftHand = new THREE.Object3D();
LeftHand.name = 'LeftHand';
LeftHand.position.set(0, 21.123, 0);
LeftForeArm.add(LeftHand);

const root = new THREE.Object3D();
root.add(RightShoulder);
root.add(LeftShoulder);

function testPose(name: string, rArmQ: number[], rForeQ: number[], rHandQ: number[], lArmQ: number[], lForeQ: number[], lHandQ: number[]) {
  RightArm.quaternion.fromArray(rArmQ);
  RightForeArm.quaternion.fromArray(rForeQ);
  RightHand.quaternion.fromArray(rHandQ);

  LeftArm.quaternion.fromArray(lArmQ);
  LeftForeArm.quaternion.fromArray(lForeQ);
  LeftHand.quaternion.fromArray(lHandQ);

  root.updateMatrixWorld(true);

  const rElbowPos = new THREE.Vector3();
  const rWristPos = new THREE.Vector3();
  const lElbowPos = new THREE.Vector3();
  const lWristPos = new THREE.Vector3();

  RightForeArm.getWorldPosition(rElbowPos);
  RightHand.getWorldPosition(rWristPos);
  LeftForeArm.getWorldPosition(lElbowPos);
  LeftHand.getWorldPosition(lWristPos);

  console.log(`\n=== POSE: ${name} ===`);
  console.log(`Right Elbow World Pos: x=${rElbowPos.x.toFixed(2)}, y=${rElbowPos.y.toFixed(2)}, z=${rElbowPos.z.toFixed(2)}`);
  console.log(`Right Wrist World Pos: x=${rWristPos.x.toFixed(2)}, y=${rWristPos.y.toFixed(2)}, z=${rWristPos.z.toFixed(2)}`);
  console.log(`Left  Elbow World Pos: x=${lElbowPos.x.toFixed(2)}, y=${lElbowPos.y.toFixed(2)}, z=${lElbowPos.z.toFixed(2)}`);
  console.log(`Left  Wrist World Pos: x=${lWristPos.x.toFixed(2)}, y=${lWristPos.y.toFixed(2)}, z=${lWristPos.z.toFixed(2)}`);
}

// Test SBV_IDLE
testPose('SBV_IDLE', 
  [0.3932, -0.2587, -0.0417, 0.8813], [0.016, 0.0021, -0.152, 0.9882], [0.023, 0.2889, 0.1613, 0.9434],
  [0.4763, 0.1736, 0.0317, 0.8614], [0.0127, -0.0021, 0.1213, 0.9925], [-0.001, -0.0707, -0.1795, 0.9812]
);

// Test SBV_HELLO midpoint
testPose('SBV_HELLO Midpoint', 
  [0.0593, 0.3688, -0.2310, 0.8984], [0.0931, 0.0010, -0.8827, 0.4607], [-0.3212, -0.1481, 0.1339, 0.9257],
  [0.4917, 0.1505, 0.0802, 0.8539], [0.0052, -0.0021, 0.0498, 0.9987], [-0.0905, -0.0613, -0.1116, 0.9877]
);

// Test current procedural I_ME midpoint
testPose('Current I_ME Midpoint',
  [-0.3, 0.4, -0.3, 0.86], [-0.52, 0.58, 0.1, 0.61], [-0.3, 0.15, 0.0, 0.94],
  [0.4763, 0.1736, 0.0317, 0.8614], [0.0127, -0.0021, 0.1213, 0.9925], [-0.001, -0.0707, -0.1795, 0.9812]
);

// Test current procedural MY midpoint
testPose('Current MY Midpoint',
  [-0.35, 0.35, -0.2, 0.85], [-0.42, 0.5, 0.2, 0.74], [0.0, -0.3, 0.0, 0.95],
  [0.4763, 0.1736, 0.0317, 0.8614], [0.0127, -0.0021, 0.1213, 0.9925], [-0.001, -0.0707, -0.1795, 0.9812]
);
