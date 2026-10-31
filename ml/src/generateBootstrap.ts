/**
 * generateBootstrap.ts
 * Creates and saves a bootstrap GRU model using pure TensorFlow.js with custom file IO handler.
 * Run: npm run generate:bootstrap
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-var-requires */

import * as fs from 'fs';
import * as path from 'path';
import { FEATURE_DIM, buildFeatureMatrix, jitterLandmarks, timeWarp } from './features/landmarkFeatures';
import { ISL_VOCABULARY } from '@signbridge/types';

// Dynamic require
// eslint-disable-next-line @typescript-eslint/no-require-imports
const tf: any = require('@tensorflow/tfjs');

const WINDOW_SIZE = 30;
const NUM_CLASSES = ISL_VOCABULARY.length;
const CLASS_LABELS = ISL_VOCABULARY.map(v => v.sign);
const MODELS_DIR = path.join(__dirname, '..', 'models', 'sbv-gru-v1');
const EPOCHS = 8;
const SAMPLES_PER_CLASS = 25;

interface HandshapeTemplate {
  extensions: [number, number, number, number, number];
  motionX: number;
  motionY: number;
  isTwoHanded: boolean;
}

const HANDSHAPE_TEMPLATES: Record<string, HandshapeTemplate> = {
  HELLO:           { extensions: [0.9,1.0,1.0,1.0,1.0], motionX:  0.04, motionY: -0.03, isTwoHanded: false },
  THANK_YOU:       { extensions: [0.85,1.0,1.0,1.0,1.0], motionX: 0.00, motionY:  0.03, isTwoHanded: false },
  PLEASE:          { extensions: [0.75,1.0,1.0,1.0,1.0], motionX:  0.02, motionY:  0.02, isTwoHanded: false },
  GOODBYE:         { extensions: [0.8,1.0,1.0,1.0,1.0], motionX:  0.06, motionY:  0.00, isTwoHanded: false },
  HELP:            { extensions: [0.9,0.05,0.05,0.05,0.05], motionX: 0.00, motionY: -0.05, isTwoHanded: true  },
  WATER:           { extensions: [0.05,1.0,1.0,1.0,0.05], motionX: 0.00, motionY:  0.02, isTwoHanded: false },
  FOOD:            { extensions: [0.35,0.35,0.35,0.35,0.35], motionX: 0.00, motionY:  0.01, isTwoHanded: false },
  HOSPITAL:        { extensions: [0.05,1.0,1.0,0.05,0.05], motionX:  0.02, motionY:  0.03, isTwoHanded: true  },
  BATHROOM:        { extensions: [0.65,0.1,0.05,0.05,0.05], motionX:  0.04, motionY:  0.00, isTwoHanded: false },
  WHERE:           { extensions: [0.95,1.0,1.0,1.0,1.0], motionX:  0.04, motionY:  0.00, isTwoHanded: true  },
  HERE:            { extensions: [0.1,1.0,0.05,0.05,0.05], motionX:  0.00, motionY:  0.04, isTwoHanded: true  },
  LEFT:            { extensions: [0.1,1.0,0.05,0.05,0.05], motionX: -0.06, motionY:  0.00, isTwoHanded: false },
  RIGHT:           { extensions: [0.1,1.0,0.05,0.05,0.05], motionX:  0.06, motionY:  0.00, isTwoHanded: false },
  YES:             { extensions: [0.6,0.0,0.0,0.0,0.0], motionX:  0.00, motionY:  0.06, isTwoHanded: false },
  NO:              { extensions: [0.75,0.9,0.9,0.0,0.0], motionX:  0.03, motionY:  0.00, isTwoHanded: false },
  DONT_UNDERSTAND: { extensions: [0.1,1.0,0.05,0.05,0.05], motionX:  0.04, motionY: -0.02, isTwoHanded: false },
  REPEAT:          { extensions: [0.55,0.85,0.85,0.85,0.85], motionX:  0.00, motionY:  0.03, isTwoHanded: true  },
};

function noise(scale: number): number {
  return (Math.random() - 0.5) * 2 * scale;
}

function syntheticLandmarks(
  extensions: [number,number,number,number,number],
  noiseLevel = 0.05
): Array<{x:number;y:number;z:number}> {
  const lm: Array<{x:number;y:number;z:number}> = [];
  lm.push({ x: 0, y: 0, z: 0 });

  for (let f = 0; f < 5; f++) {
    const angle = ((f - 2) * Math.PI) / 8;
    const palmX = 0.2 * Math.cos(angle);
    const palmY = -0.1 + 0.05 * f;
    const ext = extensions[f];

    lm.push({ x: palmX + noise(noiseLevel), y: palmY + noise(noiseLevel), z: noise(noiseLevel * 0.5) });
    lm.push({ x: palmX + 0.2 * ext * Math.cos(angle) + noise(noiseLevel), y: palmY - 0.3 * ext + noise(noiseLevel), z: noise(noiseLevel * 0.5) });
    lm.push({ x: palmX + 0.35 * ext * Math.cos(angle) + noise(noiseLevel), y: palmY - 0.5 * ext + noise(noiseLevel), z: noise(noiseLevel * 0.5) });
    lm.push({ x: palmX + 0.5 * ext * Math.cos(angle) + noise(noiseLevel), y: palmY - 0.65 * ext + noise(noiseLevel), z: noise(noiseLevel * 0.5) });
  }
  return lm;
}

function generateSyntheticSequence(
  template: HandshapeTemplate,
  windowSize: number
): Array<{ rightHand: Array<{x:number;y:number;z:number}> | null; leftHand: Array<{x:number;y:number;z:number}> | null }> {
  const frames = [];
  const speedVariant = 0.7 + Math.random() * 0.6;

  for (let t = 0; t < windowSize; t++) {
    const progress = t / (windowSize - 1);
    const offsetX = template.motionX * progress * speedVariant;
    const offsetY = template.motionY * progress * speedVariant;

    const rightLMs = syntheticLandmarks(template.extensions, 0.04);
    rightLMs.forEach(lm => { lm.x += offsetX; lm.y += offsetY; });

    const leftLMs = template.isTwoHanded
      ? syntheticLandmarks(template.extensions, 0.05).map(lm => ({
          x: -lm.x + offsetX * 0.5,
          y:  lm.y + offsetY * 0.5,
          z:  lm.z,
        }))
      : null;

    frames.push({ rightHand: rightLMs, leftHand: leftLMs });
  }
  return frames;
}

function buildDataset(): { xs: number[][][]; ys: number[] } {
  const xs: number[][][] = [];
  const ys: number[] = [];

  for (let ci = 0; ci < CLASS_LABELS.length; ci++) {
    const sign = CLASS_LABELS[ci];
    const template = HANDSHAPE_TEMPLATES[sign];
    if (!template) { continue; }

    for (let s = 0; s < SAMPLES_PER_CLASS; s++) {
      const rawFrames = generateSyntheticSequence(template, WINDOW_SIZE);
      const matrix = buildFeatureMatrix(rawFrames, WINDOW_SIZE);

      let aug = matrix;
      if (Math.random() < 0.3) {
        const srcSize = Math.floor(WINDOW_SIZE * (0.8 + Math.random() * 0.4));
        const srcMatrix = buildFeatureMatrix(generateSyntheticSequence(template, srcSize), srcSize);
        aug = timeWarp(srcMatrix, srcSize, WINDOW_SIZE);
      }
      const jittered = jitterLandmarks(aug, 0.025);

      const sequence: number[][] = [];
      for (let t = 0; t < WINDOW_SIZE; t++) {
        const row: number[] = [];
        for (let f = 0; f < FEATURE_DIM; f++) row.push(jittered[t * FEATURE_DIM + f]);
        sequence.push(row);
      }
      xs.push(sequence);
      ys.push(ci);
    }
  }

  return { xs, ys };
}

async function main(): Promise<void> {
  console.log('=============================================');
  console.log('SIGNBRIDGE2VISION — Bootstrap GRU Generator');
  console.log('=============================================');
  console.log(`Generating ${NUM_CLASSES} classes x ${SAMPLES_PER_CLASS} samples...`);

  const { xs, ys } = buildDataset();

  // Shuffle
  const indices = Array.from({ length: xs.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const sxs = indices.map(i => xs[i]);
  const sys = indices.map(i => ys[i]);

  const splitIdx = Math.floor(sxs.length * 0.85);
  const trainXs = tf.tensor3d(sxs.slice(0, splitIdx));
  const trainYs = tf.tensor1d(sys.slice(0, splitIdx)).toFloat();
  const valXs   = tf.tensor3d(sxs.slice(splitIdx));
  const valYs   = tf.tensor1d(sys.slice(splitIdx)).toFloat();

  console.log(`Train: ${splitIdx} | Val: ${sxs.length - splitIdx}`);
  console.log(`Input shape: [${WINDOW_SIZE}, ${FEATURE_DIM}] | Classes: ${NUM_CLASSES}`);

  const model = tf.sequential();
  model.add(tf.layers.masking({ maskValue: 0, inputShape: [WINDOW_SIZE, FEATURE_DIM] }));
  model.add(tf.layers.gru({ units: 64, returnSequences: true, dropout: 0.3, recurrentDropout: 0.2 }));
  model.add(tf.layers.gru({ units: 64, returnSequences: false, dropout: 0.3, recurrentDropout: 0.2 }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });

  console.log(`\nTraining for ${EPOCHS} epochs...\n`);

  await model.fit(trainXs, trainYs, {
    epochs: EPOCHS,
    batchSize: 64,
    validationData: [valXs, valYs],
    callbacks: {
      onEpochEnd: async (epoch: number, logs: any) => {
        const acc = ((logs?.acc ?? logs?.accuracy ?? 0) * 100).toFixed(1);
        const va  = ((logs?.val_acc ?? logs?.val_accuracy ?? 0) * 100).toFixed(1);
        console.log(`  Epoch ${String(epoch + 1).padStart(2)} / ${EPOCHS}  loss=${(logs?.loss ?? 0).toFixed(4)}  acc=${acc}%  val_acc=${va}%`);
      },
    },
  });

  const evalResult: any = model.evaluate(valXs, valYs);
  const valAccData: Float32Array = await evalResult[1].data();
  const valAcc = valAccData[0];
  console.log(`\nFinal val accuracy (synthetic): ${(valAcc * 100).toFixed(1)}%`);

  // Custom Node IOHandler for saving
  fs.mkdirSync(MODELS_DIR, { recursive: true });

  const customSaveHandler = tf.io.withSaveHandler(async (artifacts: any) => {
    const weightsManifest = [{
      paths: ['./weights.bin'],
      weights: artifacts.weightSpecs || [],
    }];
    const modelJSON = {
      modelTopology: artifacts.modelTopology,
      format: artifacts.format || 'layers-model',
      generatedBy: 'TensorFlow.js Layers',
      convertedBy: null,
      weightsManifest,
    };
    fs.writeFileSync(path.join(MODELS_DIR, 'model.json'), JSON.stringify(modelJSON, null, 2));
    if (artifacts.weightData) {
      fs.writeFileSync(path.join(MODELS_DIR, 'weights.bin'), Buffer.from(artifacts.weightData));
    }
    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: JSON.stringify(artifacts.modelTopology).length,
        weightSpecsBytes: JSON.stringify(artifacts.weightSpecs || []).length,
        weightDataBytes: artifacts.weightData ? artifacts.weightData.byteLength : 0,
      }
    };
  });

  await model.save(customSaveHandler);

  const metadata = {
    id: 'sbv-gru-v1',
    version: '1.0.0-bootstrap',
    featureDim: FEATURE_DIM,
    windowSize: WINDOW_SIZE,
    numClasses: NUM_CLASSES,
    classes: CLASS_LABELS,
    accuracy: Math.round(valAcc * 1000) / 1000,
    f1Score: null,
    trainedAt: new Date().toISOString(),
    isBootstrap: true,
    description: 'Bootstrap GRU — synthetic data only. Run npm run collect:data then npm run train for real accuracy.',
  };
  fs.writeFileSync(path.join(MODELS_DIR, 'metadata.json'), JSON.stringify(metadata, null, 2));

  console.log(`\nBootstrap model saved to: ${MODELS_DIR}`);
  fs.readdirSync(MODELS_DIR).forEach(f => console.log(`  - ${f}`));

  // Copy to web public directory
  const webModelsDir = path.join(__dirname, '..', '..', 'apps', 'web', 'public', 'models', 'sbv-gru-v1');
  fs.mkdirSync(webModelsDir, { recursive: true });
  fs.cpSync(MODELS_DIR, webModelsDir, { recursive: true });
  console.log(`Copied model files to web public folder: ${webModelsDir}`);

  trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
}

main().catch(e => { console.error(e); process.exit(1); });