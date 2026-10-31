import * as fs from 'fs';
import * as path from 'path';
import * as THREE from 'three';
// @ts-ignore
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// Polyfills for headless Node environment
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
loader.parse(
  arrayBuffer,
  '',
  (gltf: any) => {
    console.log('GLB successfully parsed!');
    const scene = gltf.scene;

    const bones: Map<string, THREE.Bone> = new Map();
    scene.traverse((obj: any) => {
      if (obj.isBone) {
        bones.set(obj.name, obj);
      }
    });

    console.log(`Found ${bones.size} bones in skeleton:`);

    const cleanName = (n: string) => n.replace(/^([a-zA-Z0-9_-]+:)?mixamorig\d*:?/i, '').replace(/^[a-zA-Z0-9_-]+:/, '');

    const interestingBones = [
      'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
      'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
      'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
    ];

    scene.updateMatrixWorld(true);

    for (const [rawName, bone] of bones.entries()) {
      const cName = cleanName(rawName);
      if (interestingBones.includes(cName)) {
        const parentName = bone.parent ? cleanName(bone.parent.name) : 'none';
        const q = bone.quaternion;
        const e = new THREE.Euler().setFromQuaternion(q);
        const wp = new THREE.Vector3();
        bone.getWorldPosition(wp);

        // Find bone vector (direction to child or along bone length)
        const localDirY = new THREE.Vector3(0, 1, 0).applyQuaternion(bone.quaternion);
        const localDirX = new THREE.Vector3(1, 0, 0).applyQuaternion(bone.quaternion);
        const localDirZ = new THREE.Vector3(0, 0, 1).applyQuaternion(bone.quaternion);

        console.log(`\nBone: ${cName.padEnd(15)} (raw: ${rawName}) <- parent: ${parentName}`);
        console.log(`  Local Pos:   [${bone.position.x.toFixed(2)}, ${bone.position.y.toFixed(2)}, ${bone.position.z.toFixed(2)}]`);
        console.log(`  World Pos:   [${wp.x.toFixed(2)}, ${wp.y.toFixed(2)}, ${wp.z.toFixed(2)}]`);
        console.log(`  Bind Quat:   [${q.x.toFixed(4)}, ${q.y.toFixed(4)}, ${q.z.toFixed(4)}, ${q.w.toFixed(4)}]`);
        console.log(`  Bind Euler:  deg=[${THREE.MathUtils.radToDeg(e.x).toFixed(1)}, ${THREE.MathUtils.radToDeg(e.y).toFixed(1)}, ${THREE.MathUtils.radToDeg(e.z).toFixed(1)}]`);
        console.log(`  Local Y (bone axis in parent): [${localDirY.x.toFixed(3)}, ${localDirY.y.toFixed(3)}, ${localDirY.z.toFixed(3)}]`);
      }
    }
  },
  (err: any) => {
    console.error('Error parsing GLB:', err);
  }
);
