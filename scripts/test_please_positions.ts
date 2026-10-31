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

  const testQ = (name: string, rArm: number[], rForeArm: number[], rHand: number[]) => {
    bones['RightArm'].quaternion.fromArray(rArm);
    bones['RightForeArm'].quaternion.fromArray(rForeArm);
    bones['RightHand'].quaternion.fromArray(rHand);
    scene.updateMatrixWorld(true);

    const elbow = new THREE.Vector3();
    const wrist = new THREE.Vector3();
    const tip = new THREE.Vector3();

    bones['RightForeArm'].getWorldPosition(elbow);
    bones['RightHand'].getWorldPosition(wrist);
    if (bones['RightHandIndex3']) bones['RightHandIndex3'].getWorldPosition(tip);

    console.log(`\n=== ${name} ===`);
    console.log(`  Elbow: [${elbow.x.toFixed(3)}, ${elbow.y.toFixed(3)}, ${elbow.z.toFixed(3)}]`);
    console.log(`  Wrist: [${wrist.x.toFixed(3)}, ${wrist.y.toFixed(3)}, ${wrist.z.toFixed(3)}]`);
    console.log(`  Index: [${tip.x.toFixed(3)}, ${tip.y.toFixed(3)}, ${tip.z.toFixed(3)}]`);
  };

  testQ('Native PLEASE mid', 
    [0.3048, 0.1715, -0.2369, 0.9064],
    [0.6672, 0.1685, -0.0050, 0.7256],
    [0.1729, 0.1404, 0.2195, 0.9499]
  );

  testQ('Native THANK_YOU mid',
    [0.2721, 0.1601, -0.2090, 0.9256],
    [0.5255, 0.2064, 0.0349, 0.8246],
    [0.2009, 0.1577, 0.2639, 0.9301]
  );
});
