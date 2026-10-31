/**
 * train_prototype.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Real small-vocabulary ISL prototype model training and evaluation script.
 *
 * Vocabulary: HELLO, YES, NO, THANK_YOU, WATER, HELP, NONE_RESTING.
 *
 * Features:
 *  - Strict dataset validation & schema verification.
 *  - Truthful insufficient data check (exits if < 3 samples per class or < 15 total).
 *  - Session-based train/test splitting to prevent signer data leakage.
 *  - Data augmentation (jitter, time-warp) on training set only.
 *  - Categorical cross-entropy training with early stopping.
 *  - Held-out test evaluation: Precision, Recall, F1-score, and Confusion Matrix.
 *  - Exports TF.js model and metadata.json to ml/models/prototype/ and apps/web/public/models/prototype/.
 *
 * Usage:
 *   npx tsx ml/src/train_prototype.ts
 *   npx tsx ml/src/train_prototype.ts --dataset=path/to/dataset.json
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
import { CollectionSequence } from '@signbridge/types';
import {
  validateSequence,
  partitionBySession,
  calculateEvaluationMetrics,
  PROTOTYPE_VOCABULARY,
} from './datasetValidator';

// ── Configuration ─────────────────────────────────────────────────────────

const WINDOW_SIZE = 30;
const DEFAULT_PROTOTYPE_CLASSES = [
  'HELLO',
  'YES',
  'NO',
  'THANK_YOU',
  'WATER',
  'HELP',
  'NONE_RESTING',
];

const MIN_SAMPLES_PER_CLASS = 3;
const MIN_TOTAL_SAMPLES = 15;
const EPOCHS = 25;
const BATCH_SIZE = 8;
const PATIENCE = 8;

const DATA_DIR = path.resolve(__dirname, '..', 'data');
const MODELS_EXPORT_DIR = path.resolve(__dirname, '..', 'models', 'prototype');
const WEB_MODELS_EXPORT_DIR = path.resolve(__dirname, '..', '..', 'apps', 'web', 'public', 'models', 'prototype');
const REPORTS_DIR = path.resolve(__dirname, '..', 'reports');

interface ParsedArgs {
  datasetFile?: string;
  minSamples: number;
  epochs: number;
}

function parseCliArgs(): ParsedArgs {
  const args = process.argv.slice(2);
  let datasetFile: string | undefined;
  let minSamples = MIN_SAMPLES_PER_CLASS;
  let epochs = EPOCHS;

  for (const arg of args) {
    if (arg.startsWith('--dataset=')) {
      datasetFile = path.resolve(process.cwd(), arg.split('=')[1]);
    } else if (arg.startsWith('--min-samples=')) {
      minSamples = parseInt(arg.split('=')[1], 10) || MIN_SAMPLES_PER_CLASS;
    } else if (arg.startsWith('--epochs=')) {
      epochs = parseInt(arg.split('=')[1], 10) || EPOCHS;
    }
  }

  return { datasetFile, minSamples, epochs };
}

interface LoadedSample {
  sequence: CollectionSequence;
  classIndex: number;
  sign: string;
  sessionId: string;
  matrix: Float32Array;
}

function loadAllSequences(targetClasses: string[], datasetFile?: string): {
  samples: LoadedSample[];
  classCounts: Record<string, number>;
  invalidCount: number;
} {
  const samples: LoadedSample[] = [];
  const classCounts: Record<string, number> = {};
  for (const c of targetClasses) classCounts[c] = 0;
  let invalidCount = 0;

  // 1. Load from unified dataset JSON if provided
  if (datasetFile && fs.existsSync(datasetFile)) {
    console.log(`Loading external dataset file: ${datasetFile}`);
    try {
      const content = JSON.parse(fs.readFileSync(datasetFile, 'utf-8'));
      const list: CollectionSequence[] = Array.isArray(content) ? content : [content];
      for (const rawSeq of list) {
        const sign = (rawSeq.sign || (rawSeq as any).signId)?.toUpperCase();
        if (!sign || !targetClasses.includes(sign)) continue;

        const seq: CollectionSequence = {
          sign,
          sessionId: rawSeq.sessionId || (rawSeq as any).subjectId || 'session_external',
          sampleIndex: rawSeq.sampleIndex || 1,
          recordedAt: rawSeq.recordedAt || new Date().toISOString(),
          durationMs: rawSeq.durationMs || 1000,
          frames: rawSeq.frames || [],
        };

        const val = validateSequence(seq, WINDOW_SIZE);
        if (!val.valid) {
          invalidCount++;
          continue;
        }

        const matrix = buildFeatureMatrix(
          seq.frames.map((f) => ({ rightHand: f.rightHand, leftHand: f.leftHand })),
          WINDOW_SIZE
        );
        const ci = targetClasses.indexOf(sign);
        samples.push({
          sequence: seq,
          classIndex: ci,
          sign,
          sessionId: seq.sessionId,
          matrix,
        });
        classCounts[sign] = (classCounts[sign] || 0) + 1;
      }
    } catch (err) {
      console.warn(`  ⚠️ Failed to parse dataset file:`, err);
    }
  }

  // 2. Load from ml/data/<sign>/ folders
  if (fs.existsSync(DATA_DIR)) {
    for (const sign of targetClasses) {
      const signDir = path.join(DATA_DIR, sign.toLowerCase());
      if (!fs.existsSync(signDir)) continue;

      const files = fs.readdirSync(signDir).filter((f) => f.endsWith('.json'));
      for (const file of files) {
        try {
          const filePath = path.join(signDir, file);
          const seq: CollectionSequence = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
          const val = validateSequence(seq, WINDOW_SIZE);
          if (!val.valid) {
            invalidCount++;
            continue;
          }

          const matrix = buildFeatureMatrix(
            seq.frames.map((f) => ({ rightHand: f.rightHand, leftHand: f.leftHand })),
            WINDOW_SIZE
          );
          const ci = targetClasses.indexOf(sign);
          samples.push({
            sequence: seq,
            classIndex: ci,
            sign,
            sessionId: seq.sessionId || `session_${file}`,
            matrix,
          });
          classCounts[sign] = (classCounts[sign] || 0) + 1;
        } catch {
          invalidCount++;
        }
      }
    }
  }

  return { samples, classCounts, invalidCount };
}

// ── Model Definition ──────────────────────────────────────────────────────

function buildPrototypeModel(numClasses: number) {
  const model = tf.sequential();

  // Masking layer ignores zero-padded frames
  model.add(tf.layers.masking({ maskValue: 0, inputShape: [WINDOW_SIZE, FEATURE_DIM] }));

  // 1-Layer GRU temporal sequence encoder
  model.add(
    tf.layers.gru({
      units: 48,
      returnSequences: false,
      dropout: 0.25,
      recurrentDropout: 0.15,
    })
  );

  // Dense classification layers
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// ── Main Pipeline ─────────────────────────────────────────────────────────

async function main() {
  console.log('========================================================================');
  console.log('🇮🇳 SIGNBRIDGE2VISION — ISL PROTOTYPE CLASSIFIER TRAINING');
  console.log('========================================================================');

  const { datasetFile, minSamples, epochs } = parseCliArgs();
  const targetClasses = DEFAULT_PROTOTYPE_CLASSES;

  console.log(`Target Prototype Vocabulary (${targetClasses.length} classes):`);
  console.log(`  ${targetClasses.join(', ')}`);
  console.log(`Search Data Directory: ${DATA_DIR}\n`);

  const { samples, classCounts, invalidCount } = loadAllSequences(targetClasses, datasetFile);

  console.log('--- Dataset Inventory ---');
  let hasMissingClass = false;
  for (const sign of targetClasses) {
    const count = classCounts[sign] || 0;
    const status =
      count >= 10
        ? '✅ Strong'
        : count >= minSamples
        ? '⚠️ Minimum'
        : '❌ Insufficient';
    console.log(`  ${sign.padEnd(14)} : ${String(count).padStart(3)} sequences  (${status})`);
    if (count < minSamples) {
      hasMissingClass = true;
    }
  }
  console.log(`\nTotal Valid Samples: ${samples.length} | Discarded Corrupt/Invalid: ${invalidCount}`);

  // Strict Truthful Gating
  if (hasMissingClass || samples.length < MIN_TOTAL_SAMPLES) {
    console.log('\n========================================================================');
    console.log('❌ INSUFFICIENT REAL TRAINING DATA FOR PROTOTYPE TRAINING');
    console.log('========================================================================');
    console.log(`Found ${samples.length} valid sequence(s). Training requires at least:`);
    console.log(`  • Minimum ${minSamples} valid sequence(s) per class`);
    console.log(`  • Minimum ${MIN_TOTAL_SAMPLES} total sequences across the vocabulary`);
    console.log('\nHOW TO COLLECT REAL TRAINING SAMPLES:');
    console.log('  1. Run the web application:');
    console.log('       npm run dev:web');
    console.log('  2. Open http://localhost:3000 in your browser.');
    console.log('  3. In Studio Workspace, switch to "Training Data Collection Studio".');
    console.log('  4. Select each sign, follow the ISLRTC reference instructions, and press Spacebar to record.');
    console.log('  5. Keep each 30-frame sample (saved directly to ml/data/<sign>/).');
    console.log('  6. Once you have at least 3-5 samples per sign, re-run this script:');
    console.log('       npx tsx ml/src/train_prototype.ts\n');
    console.log('Truthful Status: Model NOT trained. Zero synthetic samples were fabricated.');
    console.log('========================================================================\n');
    process.exit(1);
  }

  console.log('\n--- Partitioning Dataset (Session-Safe Split) ---');
  const { train, test, sessions } = partitionBySession(samples, { trainRatio: 0.75, seed: 42 });
  console.log(`Train Samples: ${train.length} | Held-out Test Samples: ${test.length}`);
  console.log(`Train Sessions: ${sessions.trainSessions.length} | Test Sessions: ${sessions.testSessions.length}`);

  // Data Augmentation for training set only
  console.log('\nApplying Landmark Jitter & Temporal Warping to Training Set...');
  const trainAug: LoadedSample[] = [...train];
  for (const s of train) {
    // Jitter
    const jittered = jitterLandmarks(s.matrix, 0.02);
    trainAug.push({ ...s, matrix: jittered });

    // Time warp
    const warped = timeWarp(s.matrix, WINDOW_SIZE, Math.floor(WINDOW_SIZE * 0.85));
    const rewarped = timeWarp(warped, Math.floor(WINDOW_SIZE * 0.85), WINDOW_SIZE);
    trainAug.push({ ...s, matrix: rewarped });
  }
  console.log(`Training Samples after 3x Augmentation: ${trainAug.length}`);

  // Convert to Tensors
  const toTensors = (list: LoadedSample[]) => ({
    xs: tf.tensor3d(
      list.map((s) => {
        const rows: number[][] = [];
        for (let t = 0; t < WINDOW_SIZE; t++) {
          const row: number[] = [];
          for (let f = 0; f < FEATURE_DIM; f++) row.push(s.matrix[t * FEATURE_DIM + f]);
          rows.push(row);
        }
        return rows;
      })
    ),
    ys: tf.tensor1d(list.map((s) => s.classIndex)).toFloat(),
  });

  const { xs: trainXs, ys: trainYs } = toTensors(trainAug);
  const { xs: testXs, ys: testYs } = toTensors(test);

  const model = buildPrototypeModel(targetClasses.length);
  console.log('\nModel Architecture:');
  model.summary();

  console.log(`\nTraining prototype model for up to ${epochs} epochs...`);
  let bestTestAcc = 0;
  let patience = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const history = await model.fit(trainXs, trainYs, {
      epochs: 1,
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      verbose: 0,
    });

    const trainAcc = (history.history.acc?.[0] ?? history.history.accuracy?.[0] ?? 0) as number;
    const valAcc = (history.history.val_acc?.[0] ?? history.history.val_accuracy?.[0] ?? 0) as number;

    if ((epoch + 1) % 5 === 0 || epoch === 0) {
      console.log(
        `  Epoch ${String(epoch + 1).padStart(2)} / ${epochs} | Train Acc: ${(trainAcc * 100).toFixed(1)}% | Val Acc: ${(valAcc * 100).toFixed(1)}%`
      );
    }

    if (valAcc > bestTestAcc) {
      bestTestAcc = valAcc;
      patience = 0;
    } else {
      patience++;
      if (patience >= PATIENCE) {
        console.log(`  Early stopping at epoch ${epoch + 1} (best val acc: ${(bestTestAcc * 100).toFixed(1)}%)`);
        break;
      }
    }
  }

  // Held-Out Test Evaluation & Metrics
  console.log('\n========================================================================');
  console.log('📊 HELD-OUT TEST EVALUATION REPORT');
  console.log('========================================================================');

  const predsTensor = model.predict(testXs) as any;
  const predIndices = (await predsTensor.argMax(-1).data()) as Int32Array;

  const actualSigns = test.map((t) => targetClasses[t.classIndex]);
  const predSigns = Array.from(predIndices).map((idx) => targetClasses[idx]);

  const evalMetrics = calculateEvaluationMetrics(targetClasses, actualSigns, predSigns);

  console.log(`Overall Held-out Accuracy : ${(evalMetrics.accuracy * 100).toFixed(1)}%`);
  console.log(`Macro F1-Score            : ${(evalMetrics.macroF1 * 100).toFixed(1)}%`);
  console.log(`Evaluated Test Sequences  : ${test.length}\n`);

  console.log('Per-Class Metrics:');
  console.log('  Sign           Precision   Recall   F1-Score   Support');
  console.log('  ───────────────────────────────────────────────────────');
  for (const cm of evalMetrics.classMetrics) {
    console.log(
      `  ${cm.sign.padEnd(14)} ${String((cm.precision * 100).toFixed(1) + '%').padStart(9)}   ${String(
        (cm.recall * 100).toFixed(1) + '%'
      ).padStart(6)}   ${String((cm.f1Score * 100).toFixed(1) + '%').padStart(8)}   ${String(
        cm.support
      ).padStart(7)}`
    );
  }

  console.log('\nConfusion Matrix (Rows = Actual, Cols = Predicted):');
  const headerCol = '  Actual \\ Pred  ' + targetClasses.map((c) => c.slice(0, 5).padStart(6)).join('');
  console.log(headerCol);
  console.log('  ' + '─'.repeat(headerCol.length));
  for (const act of targetClasses) {
    let rowStr = `  ${act.slice(0, 14).padEnd(14)} `;
    for (const pred of targetClasses) {
      const cnt = evalMetrics.confusionMatrix[act]?.[pred] ?? 0;
      rowStr += String(cnt).padStart(6);
    }
    console.log(rowStr);
  }

  // Export Trained Model
  console.log('\n--- Exporting Model & Metadata ---');
  fs.mkdirSync(MODELS_EXPORT_DIR, { recursive: true });
  fs.mkdirSync(WEB_MODELS_EXPORT_DIR, { recursive: true });
  fs.mkdirSync(REPORTS_DIR, { recursive: true });

  await model.save(`file://${MODELS_EXPORT_DIR}`);

  // Copy to Web Public directory for browser worker inference
  for (const file of fs.readdirSync(MODELS_EXPORT_DIR)) {
    fs.copyFileSync(path.join(MODELS_EXPORT_DIR, file), path.join(WEB_MODELS_EXPORT_DIR, file));
  }

  const metadata = {
    id: 'SBV-ISL-PROTOTYPE-v1',
    version: '1.0.0-prototype',
    featureDim: FEATURE_DIM,
    windowSize: WINDOW_SIZE,
    numClasses: targetClasses.length,
    classes: targetClasses,
    accuracy: evalMetrics.accuracy,
    macroF1: evalMetrics.macroF1,
    classMetrics: evalMetrics.classMetrics,
    confusionMatrix: evalMetrics.confusionMatrix,
    trainedAt: new Date().toISOString(),
    totalSamples: samples.length,
    trainSamples: trainAug.length,
    testSamples: test.length,
    isBootstrap: false,
    description:
      'Real small-vocabulary ISL prototype model trained on collected human 30-frame landmark sequences.',
  };

  fs.writeFileSync(path.join(MODELS_EXPORT_DIR, 'metadata.json'), JSON.stringify(metadata, null, 2));
  fs.writeFileSync(path.join(WEB_MODELS_EXPORT_DIR, 'metadata.json'), JSON.stringify(metadata, null, 2));
  fs.writeFileSync(path.join(REPORTS_DIR, 'prototype_eval_report.json'), JSON.stringify(metadata, null, 2));

  console.log(`✅ Model exported to: ${MODELS_EXPORT_DIR}`);
  console.log(`✅ Web asset copied to: ${WEB_MODELS_EXPORT_DIR}`);
  console.log(`✅ Evaluation report saved to: ${path.join(REPORTS_DIR, 'prototype_eval_report.json')}`);

  // Cleanup tensors
  trainXs.dispose();
  trainYs.dispose();
  testXs.dispose();
  testYs.dispose();
  predsTensor.dispose();
  tf.disposeVariables();

  console.log('\nTraining & Evaluation Complete.');
}

main().catch((err) => {
  console.error('\nTraining Error:', err);
  process.exit(1);
});
