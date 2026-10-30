/**
 * train.ts
 * â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
 * Real GRU training pipeline.
 *
 * Reads landmark sequences from ml/data/<SIGN>/<session_id>/sequence_*.json,
 * extracts features, augments, splits by session to prevent data leakage,
 * trains a 2-layer GRU, evaluates, and exports to ml/models/sbv-gru-v1/.
 *
 * Usage: npm run train
 *
 * Prerequisites: Run `npm run collect:data` first to gather real hand sequences.
 */

// eslint-disable-next-line @typescript-eslint/no-require-imports
const tf = require('@tensorflow/tfjs') as typeof import('@tensorflow/tfjs');

import * as fs from 'fs';
import * as path from 'path';
import {
  FEATURE_DIM,
  buildFeatureMatrix,
  jitterLandmarks,
  timeWarp,
} from './features/landmarkFeatures';
import { ISL_VOCABULARY, CollectionSequence } from '@signbridge2vision/types';

const WINDOW_SIZE  = 30;
const NUM_CLASSES  = ISL_VOCABULARY.length;
const CLASS_LABELS = ISL_VOCABULARY.map(v => v.sign);
const DATA_DIR     = path.join(__dirname, '..', 'data');
const MODELS_DIR   = path.join(__dirname, '..', 'models', 'sbv-gru-v1');
const REPORTS_DIR  = path.join(__dirname, '..', 'reports');
const EPOCHS       = 80;
const BATCH_SIZE   = 32;
const PATIENCE     = 15;

// â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface LabelledSample {
  classIndex: number;
  sessionId: string;
  matrix: Float32Array;
}

function loadDataset(): LabelledSample[] {
  const samples: LabelledSample[] = [];

  for (let ci = 0; ci < CLASS_LABELS.length; ci++) {
    const sign = CLASS_LABELS[ci];
    const signDir = path.join(DATA_DIR, sign.toLowerCase());
    if (!fs.existsSync(signDir)) {
      console.warn(`  âš  No data directory for sign: ${sign}`);
      continue;
    }

    const files = fs.readdirSync(signDir).filter(f => f.endsWith('.json'));
    console.log(`  ${sign}: ${files.length} sequences`);

    for (const file of files) {
      try {
        const seq: CollectionSequence = JSON.parse(
          fs.readFileSync(path.join(signDir, file), 'utf-8')
        );
        const matrix = buildFeatureMatrix(
          seq.frames.map(f => ({ rightHand: f.rightHand, leftHand: f.leftHand })),
          WINDOW_SIZE
        );
        samples.push({ classIndex: ci, sessionId: seq.sessionId, matrix });
      } catch (err) {
        console.warn(`  âš  Failed to load ${file}:`, err);
      }
    }
  }

  return samples;
}

// â”€â”€ Session-safe split (no leakage) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function splitBySession(
  samples: LabelledSample[],
  trainRatio = 0.70,
  valRatio = 0.15
): { train: LabelledSample[]; val: LabelledSample[]; test: LabelledSample[] } {
  // Group samples by session
  const sessionMap = new Map<string, LabelledSample[]>();
  for (const s of samples) {
    if (!sessionMap.has(s.sessionId)) sessionMap.set(s.sessionId, []);
    sessionMap.get(s.sessionId)!.push(s);
  }

  const sessions = Array.from(sessionMap.keys());
  // Shuffle sessions
  for (let i = sessions.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [sessions[i], sessions[j]] = [sessions[j], sessions[i]];
  }

  const trainEnd = Math.floor(sessions.length * trainRatio);
  const valEnd   = Math.floor(sessions.length * (trainRatio + valRatio));

  const train: LabelledSample[] = [];
  const val:   LabelledSample[] = [];
  const test:  LabelledSample[] = [];

  sessions.slice(0, trainEnd).forEach(sid => train.push(...sessionMap.get(sid)!));
  sessions.slice(trainEnd, valEnd).forEach(sid => val.push(...sessionMap.get(sid)!));
  sessions.slice(valEnd).forEach(sid => test.push(...sessionMap.get(sid)!));

  return { train, val, test };
}

// â”€â”€ Augmentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function augmentSamples(samples: LabelledSample[], numAugments = 3): LabelledSample[] {
  const augmented: LabelledSample[] = [...samples];

  // Signs where mirroring is NOT valid (left/right directional)
  const noMirror = new Set(['LEFT', 'RIGHT']);

  for (const s of samples) {
    for (let a = 0; a < numAugments; a++) {
      // Jitter
      const jittered = jitterLandmarks(s.matrix, 0.025);
      augmented.push({ ...s, matrix: jittered });

      // Time warp
      if (Math.random() < 0.5) {
        const srcSize = Math.floor(WINDOW_SIZE * (0.75 + Math.random() * 0.5));
        const warped  = timeWarp(s.matrix, WINDOW_SIZE, srcSize);
        const rewarped = timeWarp(warped, srcSize, WINDOW_SIZE);
        augmented.push({ ...s, matrix: rewarped });
      }
    }
  }

  return augmented;
}

// â”€â”€ Shuffle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function shuffle<T>(arr: T[]): T[] {
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async function buildModel() {
  const model = tf.sequential();

  model.add(tf.layers.masking({ maskValue: 0, inputShape: [WINDOW_SIZE, FEATURE_DIM] }));

  model.add(tf.layers.gru({
    units: 64,
    returnSequences: true,
    dropout: 0.3,
    recurrentDropout: 0.2,
  }));

  model.add(tf.layers.gru({
    units: 64,
    returnSequences: false,
    dropout: 0.3,
    recurrentDropout: 0.2,
  }));

  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async function main() {
  console.log('===================================================');
  console.log('SIGNBRIDGE2VISION â€” Real GRU Training Pipeline');
  console.log('===================================================');
  console.log('Loading dataset from ml/data/â€¦');

  const allSamples = loadDataset();

  if (allSamples.length < NUM_CLASSES * 5) {
    console.error('\nâŒ INSUFFICIENT TRAINING DATA');
    console.error(`Found ${allSamples.length} samples across ${new Set(allSamples.map(s => s.classIndex)).size} classes.`);
    console.error('You need at least 5 samples per class (30+ recommended).');
    console.error('Run `npm run collect:data` to collect real training data.\n');
    process.exit(1);
  }

  console.log(`\nLoaded ${allSamples.length} samples total.`);

  const { train, val, test } = splitBySession(allSamples);
  console.log(`Session-safe split: train=${train.length} | val=${val.length} | test=${test.length}`);

  const trainAug = shuffle(augmentSamples(train, 3));
  console.log(`After augmentation: train=${trainAug.length}`);

  const toTensors = (samples: LabelledSample[]) => ({
    xs: tf.tensor3d(samples.map(s => {
      const rows: number[][] = [];
      for (let t = 0; t < WINDOW_SIZE; t++) {
        const row: number[] = [];
        for (let f = 0; f < FEATURE_DIM; f++) row.push(s.matrix[t * FEATURE_DIM + f]);
        rows.push(row);
      }
      return rows;
    })),
    ys: tf.tensor1d(samples.map(s => s.classIndex)).toFloat(),
  });

  const { xs: trainXs, ys: trainYs } = toTensors(trainAug);
  const { xs: valXs,   ys: valYs   } = toTensors(val);
  const { xs: testXs,  ys: testYs  } = toTensors(test);

  const model = await buildModel();
  model.summary();

  let bestValAcc = 0;
  let patience   = 0;

  console.log(`\nTraining for up to ${EPOCHS} epochs (early stop patience=${PATIENCE})â€¦\n`);

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    const h = await model.fit(trainXs, trainYs, {
      epochs: 1,
      batchSize: BATCH_SIZE,
      validationData: [valXs, valYs],
      verbose: 0,
    });

    const acc    = (h.history.acc?.[0] ?? h.history.accuracy?.[0] ?? 0) as number;
    const valAcc = (h.history.val_acc?.[0] ?? h.history.val_accuracy?.[0] ?? 0) as number;

    if ((epoch + 1) % 5 === 0 || epoch === 0) {
      console.log(`  Epoch ${String(epoch + 1).padStart(3)} / ${EPOCHS}  acc=${(acc * 100).toFixed(1)}%  val_acc=${(valAcc * 100).toFixed(1)}%`);
    }

    if (valAcc > bestValAcc) {
      bestValAcc = valAcc;
      patience   = 0;
    } else {
      patience++;
      if (patience >= PATIENCE) {
        console.log(`\n  Early stopping at epoch ${epoch + 1} (best val_acc=${(bestValAcc * 100).toFixed(1)}%)`);
        break;
      }
    }
  }

  // Test evaluation
  const evalResult: any = model.evaluate(testXs, testYs); const testAccData = await evalResult[1].data();
  const testAcc = testAccData[0];
  console.log(`\nTest accuracy: ${(testAcc * 100).toFixed(1)}%`);

  // Save
  fs.mkdirSync(MODELS_DIR, { recursive: true });
  fs.mkdirSync(REPORTS_DIR, { recursive: true });

  await model.save(`file://${MODELS_DIR}`);

  const metadata = {
    id: 'sbv-gru-v1',
    version: '1.0.0-real',
    featureDim: FEATURE_DIM,
    windowSize: WINDOW_SIZE,
    numClasses: NUM_CLASSES,
    classes: CLASS_LABELS,
    accuracy: Math.round(testAcc * 1000) / 1000,
    f1Score: null,
    trainedAt: new Date().toISOString(),
    isBootstrap: false,
    description: 'GRU model trained on real human hand landmark sequences collected with the SignBridge2Vision data collection tool.',
    totalSamples: allSamples.length,
    trainSamples: trainAug.length,
    valSamples: val.length,
    testSamples: test.length,
  };

  fs.writeFileSync(path.join(MODELS_DIR, 'metadata.json'), JSON.stringify(metadata, null, 2));
  fs.writeFileSync(path.join(REPORTS_DIR, 'training_summary.json'), JSON.stringify({
    ...metadata,
    bestValAcc,
    testAcc,
    generatedAt: new Date().toISOString(),
  }, null, 2));

  console.log(`\nâœ… Model saved to: ${MODELS_DIR}`);
  console.log('Next: Run `npm run eval:ml` for full evaluation, then copy model to apps/web/public/models/sbv-gru-v1/');

  trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
  testXs.dispose(); testYs.dispose();
  tf.disposeVariables();
}

main().catch(err => { console.error(err); process.exit(1); });
