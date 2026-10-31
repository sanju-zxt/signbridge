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

  function evalPose(
    side: 'Right' | 'Left',
    armEuler: [number, number, number],
    foreEuler: [number, number, number],
    handEuler: [number, number, number]
  ) {
    const arm = bones[`${side}Arm`];
    const fore = bones[`${side}ForeArm`];
    const hand = bones[`${side}Hand`];

    arm.quaternion.setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(armEuler[0]),
      THREE.MathUtils.degToRad(armEuler[1]),
      THREE.MathUtils.degToRad(armEuler[2]),
      'XYZ'
    ));

    fore.quaternion.setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(foreEuler[0]),
      THREE.MathUtils.degToRad(foreEuler[1]),
      THREE.MathUtils.degToRad(foreEuler[2]),
      'XYZ'
    ));

    hand.quaternion.setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(handEuler[0]),
      THREE.MathUtils.degToRad(handEuler[1]),
      THREE.MathUtils.degToRad(handEuler[2]),
      'XYZ'
    ));

    scene.updateMatrixWorld(true);

    const elbow = new THREE.Vector3();
    const wrist = new THREE.Vector3();
    const tip = new THREE.Vector3();

    fore.getWorldPosition(elbow);
    hand.getWorldPosition(wrist);
    if (bones[`${side}HandIndex3`]) {
      bones[`${side}HandIndex3`].getWorldPosition(tip);
    }

    const pointingDir = new THREE.Vector3().subVectors(tip, wrist).normalize();

    return {
      elbow,
      wrist,
      tip,
      pointingDir,
      armQ: [arm.quaternion.x, arm.quaternion.y, arm.quaternion.z, arm.quaternion.w],
      foreQ: [fore.quaternion.x, fore.quaternion.y, fore.quaternion.z, fore.quaternion.w],
      handQ: [hand.quaternion.x, hand.quaternion.y, hand.quaternion.z, hand.quaternion.w],
    };
  }

  // 1. Solve I_ME (index points to chest at x ≈ -0.05, y ≈ 1.25, z ≈ 0.12)
  console.log('--- SOLVING I_ME ---');
  let bestDistI = 999;
  let bestI: any = null;
  const targetChest = new THREE.Vector3(-0.05, 1.25, 0.12);

  for (let ax = 15; ax <= 45; ax += 10) {
    for (let ay = -10; ay <= 30; ay += 10) {
      for (let az = -35; az <= -5; az += 10) {
        for (let fx = 70; fx <= 110; fx += 10) {
          for (let fy = -10; fy <= 30; fy += 10) {
            for (let fz = 0; fz <= 50; fz += 10) {
              for (let hx = -15; hx <= 25; hx += 15) {
                for (let hy = -20; hy <= 20; hy += 20) {
                  for (let hz = -10; hz <= 30; hz += 20) {
                    const res = evalPose('Right', [ax, ay, az], [fx, fy, fz], [hx, hy, hz]);
                    const d = res.tip.distanceTo(targetChest);
                    if (d < bestDistI) {
                      bestDistI = d;
                      bestI = { ax, ay, az, fx, fy, fz, hx, hy, hz, res };
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  console.log(`Best I_ME dist=${bestDistI.toFixed(3)}m:`);
  console.log(`  Arm: [${bestI.ax}, ${bestI.ay}, ${bestI.az}] deg -> quat=[${bestI.res.armQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Fore:[${bestI.fx}, ${bestI.fy}, ${bestI.fz}] deg -> quat=[${bestI.res.foreQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Hand:[${bestI.hx}, ${bestI.hy}, ${bestI.hz}] deg -> quat=[${bestI.res.handQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Tip: [${bestI.res.tip.x.toFixed(3)}, ${bestI.res.tip.y.toFixed(3)}, ${bestI.res.tip.z.toFixed(3)}]`);
  console.log(`  Wrist: [${bestI.res.wrist.x.toFixed(3)}, ${bestI.res.wrist.y.toFixed(3)}, ${bestI.res.wrist.z.toFixed(3)}]`);

  // 2. Solve MY (flat palm on sternum at x ≈ -0.04, y ≈ 1.25, z ≈ 0.10)
  console.log('\n--- SOLVING MY ---');
  let bestDistMY = 999;
  let bestMY: any = null;
  const targetChestMY = new THREE.Vector3(-0.04, 1.25, 0.10);

  for (let ax = 15; ax <= 45; ax += 10) {
    for (let ay = -10; ay <= 30; ay += 10) {
      for (let az = -35; az <= -5; az += 10) {
        for (let fx = 70; fx <= 110; fx += 10) {
          for (let fy = 0; fy <= 40; fy += 10) {
            for (let fz = 10; fz <= 50; fz += 10) {
              for (let hx = -15; hx <= 25; hx += 15) {
                for (let hy = -20; hy <= 20; hy += 20) {
                  for (let hz = -10; hz <= 30; hz += 20) {
                    const res = evalPose('Right', [ax, ay, az], [fx, fy, fz], [hx, hy, hz]);
                    const d = res.wrist.distanceTo(targetChestMY);
                    if (d < bestDistMY) {
                      bestDistMY = d;
                      bestMY = { ax, ay, az, fx, fy, fz, hx, hy, hz, res };
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  console.log(`Best MY dist=${bestDistMY.toFixed(3)}m:`);
  console.log(`  Arm: [${bestMY.ax}, ${bestMY.ay}, ${bestMY.az}] deg -> quat=[${bestMY.res.armQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Fore:[${bestMY.fx}, ${bestMY.fy}, ${bestMY.fz}] deg -> quat=[${bestMY.res.foreQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Hand:[${bestMY.hx}, ${bestMY.hy}, ${bestMY.hz}] deg -> quat=[${bestMY.res.handQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Wrist: [${bestMY.res.wrist.x.toFixed(3)}, ${bestMY.res.wrist.y.toFixed(3)}, ${bestMY.res.wrist.z.toFixed(3)}]`);

  // 3. Solve YOU (arm extended forward, index pointing at conversational partner)
  console.log('\n--- SOLVING YOU ---');
  let bestDistYOU = 999;
  let bestYOU: any = null;
  const targetYOU = new THREE.Vector3(-0.15, 1.25, 0.40);

  for (let ax = 0; ax <= 40; ax += 10) {
    for (let ay = -20; ay <= 10; ay += 10) {
      for (let az = -20; az <= 10; az += 10) {
        for (let fx = 30; fx <= 80; fx += 10) {
          for (let fy = -10; fy <= 10; fy += 10) {
            for (let fz = -20; fz <= 20; fz += 10) {
              const res = evalPose('Right', [ax, ay, az], [fx, fy, fz], [0, 0, 0]);
              const d = res.tip.distanceTo(targetYOU);
              if (d < bestDistYOU && res.pointingDir.z > 0.5) {
                bestDistYOU = d;
                bestYOU = { ax, ay, az, fx, fy, fz, res };
              }
            }
          }
        }
      }
    }
  }

  console.log(`Best YOU dist=${bestDistYOU.toFixed(3)}m:`);
  console.log(`  Arm: [${bestYOU.ax}, ${bestYOU.ay}, ${bestYOU.az}] deg -> quat=[${bestYOU.res.armQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Fore:[${bestYOU.fx}, ${bestYOU.fy}, ${bestYOU.fz}] deg -> quat=[${bestYOU.res.foreQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Hand: [0, 0, 0] deg -> quat=[${bestYOU.res.handQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  Tip: [${bestYOU.res.tip.x.toFixed(3)}, ${bestYOU.res.tip.y.toFixed(3)}, ${bestYOU.res.tip.z.toFixed(3)}]`);
  console.log(`  Pointing dir: [${bestYOU.res.pointingDir.x.toFixed(2)}, ${bestYOU.res.pointingDir.y.toFixed(2)}, ${bestYOU.res.pointingDir.z.toFixed(2)}]`);

  // 4. Solve WHAT (both arms forward, palms up)
  console.log('\n--- SOLVING WHAT ---');
  const resWhatR = evalPose('Right', [25, -10, -15], [75, 10, -10], [-10, 0, -20]);
  const resWhatL = evalPose('Left', [25, 10, 15], [75, -10, 10], [-10, 0, 20]);
  console.log(`WHAT Right: Elbow=[${resWhatR.elbow.x.toFixed(2)}, ${resWhatR.elbow.y.toFixed(2)}, ${resWhatR.elbow.z.toFixed(2)}] Wrist=[${resWhatR.wrist.x.toFixed(2)}, ${resWhatR.wrist.y.toFixed(2)}, ${resWhatR.wrist.z.toFixed(2)}]`);
  console.log(`WHAT Left:  Elbow=[${resWhatL.elbow.x.toFixed(2)}, ${resWhatL.elbow.y.toFixed(2)}, ${resWhatL.elbow.z.toFixed(2)}] Wrist=[${resWhatL.wrist.x.toFixed(2)}, ${resWhatL.wrist.y.toFixed(2)}, ${resWhatL.wrist.z.toFixed(2)}]`);
  console.log(`  rArm:     [${resWhatR.armQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  rForeArm: [${resWhatR.foreQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  rHand:    [${resWhatR.handQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  lArm:     [${resWhatL.armQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  lForeArm: [${resWhatL.foreQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
  console.log(`  lHand:    [${resWhatL.handQ.map((v: number) => v.toFixed(4)).join(', ')}]`);
});
