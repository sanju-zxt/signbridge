import * as fs from 'fs';
import * as path from 'path';
import { Landmark3D, NormalizedFrame } from '@signbridge2vision/types';
import { SIGN_FEATURE_TEMPLATES } from './temporalClassifier';

export interface BenchmarkSample {
  id: string;
  signId: string;
  subjectId: string;
  speed: 'slow' | 'normal' | 'fast';
  noiseLevel: number;
  frames: NormalizedFrame[];
}

export function generateSyntheticSequence(
  signId: string,
  subjectId: string,
  speed: 'slow' | 'normal' | 'fast' = 'normal',
  noise: number = 0.02
): BenchmarkSample {
  const template = SIGN_FEATURE_TEMPLATES[signId];
  const frameCount = speed === 'slow' ? 36 : speed === 'fast' ? 24 : 30;
  const frames: NormalizedFrame[] = [];

  const hash = (subjectId + signId).split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
  const subjectScale = 0.92 + (hash % 16) * 0.01;

  for (let f = 0; f < frameCount; f++) {
    const progress = f / (frameCount - 1);
    const easeProgress = Math.sin(progress * Math.PI / 2);

    const mx = template.motionVector[0] * easeProgress * subjectScale + (Math.random() - 0.5) * noise;
    const my = template.motionVector[1] * easeProgress * subjectScale + (Math.random() - 0.5) * noise;
    const mz = template.motionVector[2] * easeProgress * subjectScale + (Math.random() - 0.5) * noise;

    const rightHand: Landmark3D[] = [];
    const wrist: Landmark3D = { x: mx, y: my, z: mz, visibility: 1.0 };
    rightHand.push(wrist);

    const fingerExtensions = template.fingerStates;
    const fingerBaseAngles = [-0.4, -0.2, 0.0, 0.2, 0.4];

    for (let fingerIdx = 0; fingerIdx < 5; fingerIdx++) {
      const ext = fingerExtensions[fingerIdx];
      const spread = fingerBaseAngles[fingerIdx];

      for (let joint = 1; joint <= 4; joint++) {
        const jointDist = (joint * 0.25) * (0.4 + 0.6 * ext) * subjectScale;
        const jx = wrist.x + Math.sin(spread) * jointDist + (Math.random() - 0.5) * noise * 0.5;
        const jy = wrist.y - Math.cos(spread) * jointDist + (Math.random() - 0.5) * noise * 0.5;
        const jz = wrist.z + (ext < 0.5 ? (joint * 0.1) : 0) + (Math.random() - 0.5) * noise * 0.5;
        rightHand.push({ x: jx, y: jy, z: jz, visibility: 1.0 });
      }
    }

    let leftHand: Landmark3D[] | null = null;
    if (template.isTwoHanded) {
      leftHand = rightHand.map(lm => ({
        x: -lm.x + (Math.random() - 0.5) * noise,
        y: lm.y + (Math.random() - 0.5) * noise,
        z: lm.z + (Math.random() - 0.5) * noise,
        visibility: 1.0
      }));
    }

    frames.push({
      timestamp: Date.now() + f * 33,
      rightHand,
      leftHand,
      pose: null,
      features: []
    });
  }

  return {
    id: signId + '_' + subjectId + '_' + speed,
    signId,
    subjectId,
    speed,
    noiseLevel: noise,
    frames
  };
}

export function generateFullBenchmarkDataset(): BenchmarkSample[] {
  const signs = Object.keys(SIGN_FEATURE_TEMPLATES);
  const subjects = ['user_01', 'user_02', 'user_03', 'user_04', 'user_05', 'user_06', 'user_07', 'user_08', 'user_09', 'user_10'];
  const speeds: Array<'slow' | 'normal' | 'fast'> = ['normal', 'slow', 'fast'];
  const dataset: BenchmarkSample[] = [];

  for (const sign of signs) {
    for (const subject of subjects) {
      for (const speed of speeds) {
        dataset.push(generateSyntheticSequence(sign, subject, speed, 0.015));
      }
    }
  }

  return dataset;
}

if (require.main === module) {
  const dataset = generateFullBenchmarkDataset();
  const outDir = path.resolve(__dirname, '../datasets');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  const outFile = path.join(outDir, 'isl_benchmark_dataset.json');
  fs.writeFileSync(outFile, JSON.stringify(dataset, null, 2), 'utf8');
  console.log('Successfully generated ISL benchmark dataset with ' + dataset.length + ' samples at ' + outFile);
}
