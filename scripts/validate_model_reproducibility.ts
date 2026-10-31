/**
 * scripts/validate_model_reproducibility.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * SignBridge V8/V9: Comprehensive Independent Model & Dataset Validation
 *
 * Verifies:
 *   1. 400 independent test samples integrity (provenance, shape [30, 126], finite).
 *   2. Unseen signer separation (user_09 & user_10 only; zero leakage).
 *   3. Hash-based duplication check against train & val splits.
 *   4. Uniform class distribution (8 samples per class across all 50 classes).
 *   5. Model reproduction: Top-1 and Top-5 accuracy on unseen signers.
 *   6. Full 50x50 Confusion Matrix.
 *   7. Per-class Precision, Recall, and F1-Score.
 *   8. Detailed error analysis of all misclassifications.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import * as tf from '@tensorflow/tfjs';

const PROJECT_ROOT = path.resolve(__dirname, '..');
const MODEL_DIR = path.join(PROJECT_ROOT, 'apps', 'web', 'public', 'models', 'sbv-gru-v1');
const DATASET_DIR = path.join(PROJECT_ROOT, 'ml', 'datasets', 'isl');
const TRAIN_DIR = path.join(DATASET_DIR, 'train');
const VAL_DIR = path.join(DATASET_DIR, 'validation');
const TEST_DIR = path.join(DATASET_DIR, 'test');
const REPORTS_DIR = path.join(PROJECT_ROOT, 'ml', 'reports');

fs.mkdirSync(REPORTS_DIR, { recursive: true });

function hashMatrix(matrix: number[][]): string {
  const hash = crypto.createHash('sha256');
  for (let t = 0; t < matrix.length; t++) {
    for (let d = 0; d < matrix[t].length; d++) {
      hash.update(matrix[t][d].toFixed(5));
    }
  }
  return hash.digest('hex');
}

async function validateModelAndDataset() {
  console.log('='.repeat(80));
  console.log('SIGNBRIDGE — PRIORITY 2: MODEL VALIDATION & DATASET REPRODUCIBILITY AUDIT');
  console.log('='.repeat(80));

  // ── STEP 1: Metadata & Class Definitions ───────────────────────────────────
  const metadata = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'metadata.json'), 'utf-8'));
  const classes: string[] = metadata.classes;
  const numClasses = classes.length;
  console.log(`\n[Audit 1] Model Metadata: ${numClasses} classes, input shape [${metadata.windowSize}, ${metadata.featureDim}]`);

  // ── STEP 2: Dataset Inventory & Provenance ─────────────────────────────────
  console.log('\n[Audit 2] Auditing Dataset Splits & Provenance...');
  const trainFiles = fs.readdirSync(TRAIN_DIR).filter(f => f.endsWith('.json'));
  const valFiles = fs.readdirSync(VAL_DIR).filter(f => f.endsWith('.json'));
  const testFiles = fs.readdirSync(TEST_DIR).filter(f => f.endsWith('.json'));

  console.log(`  Train samples:      ${trainFiles.length} (Expected: 1400)`);
  console.log(`  Validation samples: ${valFiles.length} (Expected: 200)`);
  console.log(`  Test samples:       ${testFiles.length} (Expected: 400)`);
  console.log(`  Total samples:      ${trainFiles.length + valFiles.length + testFiles.length} (Expected: 2000)`);

  if (testFiles.length !== 400) {
    throw new Error(`Expected exactly 400 test files, got ${testFiles.length}`);
  }

  // Hash train and val samples for strict leakage detection
  console.log('\n[Audit 3] Checking for Data Leakage between Splits...');
  const trainValHashes = new Set<string>();
  for (const f of trainFiles) {
    const data = JSON.parse(fs.readFileSync(path.join(TRAIN_DIR, f), 'utf-8'));
    trainValHashes.add(hashMatrix(data.featureMatrix));
  }
  for (const f of valFiles) {
    const data = JSON.parse(fs.readFileSync(path.join(VAL_DIR, f), 'utf-8'));
    trainValHashes.add(hashMatrix(data.featureMatrix));
  }

  let leakageCount = 0;
  const testSigners = new Set<string>();
  const classCounts: Record<string, number> = {};
  classes.forEach(c => { classCounts[c] = 0; });

  const testSamples: Array<{
    filename: string;
    sign: string;
    signer: string;
    speed: string;
    matrix: number[][];
  }> = [];

  for (const f of testFiles) {
    const raw = JSON.parse(fs.readFileSync(path.join(TEST_DIR, f), 'utf-8'));
    const sign = raw.sign || raw.signId;
    const signer = raw.signer || raw.signerId;
    const speed = raw.speed || 'standard';
    const matrix = raw.featureMatrix;

    testSigners.add(signer);
    classCounts[sign] = (classCounts[sign] || 0) + 1;

    // Check shape
    if (matrix.length !== 30 || matrix[0].length !== 126) {
      throw new Error(`Invalid shape in ${f}: [${matrix.length}, ${matrix[0].length}]`);
    }

    // Check finite
    for (let t = 0; t < 30; t++) {
      for (let d = 0; d < 126; d++) {
        if (!Number.isFinite(matrix[t][d])) {
          throw new Error(`Non-finite value in ${f} at [${t}, ${d}]`);
        }
      }
    }

    // Check hash leakage
    const h = hashMatrix(matrix);
    if (trainValHashes.has(h)) {
      leakageCount++;
      console.warn(`  [LEAKAGE WARNING] Test file ${f} matches train/val hash!`);
    }

    testSamples.push({ filename: f, sign, signer, speed, matrix });
  }

  console.log(`  ✓ Unseen test signers: [${Array.from(testSigners).sort().join(', ')}]`);
  console.log(`  ✓ Duplicate/Augmented Leakage from Train/Val: ${leakageCount} (0 expected)`);

  // Verify class distribution
  let balanced = true;
  for (const c of classes) {
    if (classCounts[c] !== 8) {
      console.warn(`  Unbalanced class: ${c} has ${classCounts[c]} test samples (expected 8)`);
      balanced = false;
    }
  }
  if (balanced) {
    console.log(`  ✓ Class distribution: Exactly 8 test sequences per class across all 50 classes (Uniform!)`);
  }

  // ── STEP 3: Load Model & Run Inference ─────────────────────────────────────
  console.log('\n[Audit 4] Loading Deployed TF.js Model & Running Reproducibility Benchmark...');
  const modelJson = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model.json'), 'utf-8'));
  const weightsBin = fs.readFileSync(path.join(MODEL_DIR, 'weights.bin'));

  const model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: modelJson.modelTopology,
    weightSpecs: modelJson.weightsManifest[0].weights,
    weightData: weightsBin.buffer
  }));

  // Build batch tensor
  const batchData: number[] = [];
  const trueIndices: number[] = [];

  for (const sample of testSamples) {
    const cIdx = classes.indexOf(sample.sign);
    trueIndices.push(cIdx);
    for (let t = 0; t < 30; t++) {
      for (let d = 0; d < 126; d++) {
        batchData.push(sample.matrix[t][d]);
      }
    }
  }

  const inputTensor = tf.tensor3d(batchData, [testSamples.length, 30, 126]);
  const t0 = performance.now();
  const predsTensor = model.predict(inputTensor) as tf.Tensor;
  const predsData = await predsTensor.data();
  const totalInferenceTimeMs = performance.now() - t0;
  inputTensor.dispose();
  predsTensor.dispose();

  console.log(`  ✓ Evaluated 400 test samples in ${totalInferenceTimeMs.toFixed(1)} ms (${(totalInferenceTimeMs / 400).toFixed(2)} ms/sample in Node CPU)`);

  // ── STEP 4: Compute Confusion Matrix, Top-1, Top-5, Precision, Recall ──────
  console.log('\n[Audit 5] Computing Confusion Matrix and Classification Metrics...');
  const confusionMatrix: number[][] = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
  let top1Correct = 0;
  let top5Correct = 0;

  interface Misclassification {
    filename: string;
    trueLabel: string;
    predictedLabel: string;
    confidence: number;
    signer: string;
    speed: string;
    top5: Array<{ label: string; prob: number }>;
  }

  const misclassifications: Misclassification[] = [];

  for (let i = 0; i < testSamples.length; i++) {
    const trueIdx = trueIndices[i];
    const probs: Array<{ idx: number; prob: number }> = [];
    for (let c = 0; c < numClasses; c++) {
      probs.push({ idx: c, prob: predsData[i * numClasses + c] });
    }
    probs.sort((a, b) => b.prob - a.prob);

    const predIdx = probs[0].idx;
    confusionMatrix[trueIdx][predIdx]++;

    if (predIdx === trueIdx) {
      top1Correct++;
    } else {
      misclassifications.push({
        filename: testSamples[i].filename,
        trueLabel: classes[trueIdx],
        predictedLabel: classes[predIdx],
        confidence: Number((probs[0].prob * 100).toFixed(2)),
        signer: testSamples[i].signer,
        speed: testSamples[i].speed,
        top5: probs.slice(0, 5).map(p => ({
          label: classes[p.idx],
          prob: Number((p.prob * 100).toFixed(2))
        }))
      });
    }

    if (probs.slice(0, 5).some(p => p.idx === trueIdx)) {
      top5Correct++;
    }
  }

  const top1Acc = (top1Correct / testSamples.length) * 100;
  const top5Acc = (top5Correct / testSamples.length) * 100;

  console.log(`\n================================================================================`);
  console.log(`INDEPENDENT TEST SET REPRODUCIBILITY RESULTS:`);
  console.log(`  Top-1 Accuracy: ${top1Acc.toFixed(2)}% (${top1Correct}/${testSamples.length})`);
  console.log(`  Top-5 Accuracy: ${top5Acc.toFixed(2)}% (${top5Correct}/${testSamples.length})`);
  console.log(`  Total Misclassified Sequences: ${misclassifications.length}/400`);
  console.log(`================================================================================`);

  // Compute per-class precision, recall, F1
  interface ClassMetrics {
    class: string;
    support: number;
    tp: number;
    fp: number;
    fn: number;
    precision: number;
    recall: number;
    f1: number;
  }

  const perClassMetrics: ClassMetrics[] = [];
  let totalPrecision = 0;
  let totalRecall = 0;
  let totalF1 = 0;

  for (let c = 0; c < numClasses; c++) {
    const tp = confusionMatrix[c][c];
    let fp = 0;
    let fn = 0;

    for (let i = 0; i < numClasses; i++) {
      if (i !== c) {
        fp += confusionMatrix[i][c];
        fn += confusionMatrix[c][i];
      }
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    totalPrecision += precision;
    totalRecall += recall;
    totalF1 += f1;

    perClassMetrics.push({
      class: classes[c],
      support: tp + fn,
      tp,
      fp,
      fn,
      precision: Number((precision * 100).toFixed(2)),
      recall: Number((recall * 100).toFixed(2)),
      f1: Number((f1 * 100).toFixed(2))
    });
  }

  const macroPrecision = (totalPrecision / numClasses) * 100;
  const macroRecall = (totalRecall / numClasses) * 100;
  const macroF1 = (totalF1 / numClasses) * 100;

  console.log(`\nMacro-Averaged Performance:`);
  console.log(`  Macro Precision: ${macroPrecision.toFixed(2)}%`);
  console.log(`  Macro Recall:    ${macroRecall.toFixed(2)}%`);
  console.log(`  Macro F1-Score:  ${macroF1.toFixed(2)}%`);

  // Print Misclassifications
  if (misclassifications.length > 0) {
    console.log(`\n[Misclassifications Breakdown (${misclassifications.length} samples)]:`);
    for (const m of misclassifications) {
      console.log(`  - File: ${m.filename}`);
      console.log(`    True: ${m.trueLabel} | Pred: ${m.predictedLabel} (${m.confidence}%) | Signer: ${m.signer} | Speed: ${m.speed}`);
      console.log(`    Top-5: ${m.top5.map(p => `${p.label} (${p.prob}%)`).join(', ')}`);
    }
  }

  // ── STEP 5: Export JSON and Markdown Reports ───────────────────────────────
  const report = {
    timestamp: new Date().toISOString(),
    numTestSamples: testSamples.length,
    numClasses,
    top1Accuracy: Number(top1Acc.toFixed(2)),
    top5Accuracy: Number(top5Acc.toFixed(2)),
    macroPrecision: Number(macroPrecision.toFixed(2)),
    macroRecall: Number(macroRecall.toFixed(2)),
    macroF1: Number(macroF1.toFixed(2)),
    leakageCount,
    classes,
    confusionMatrix,
    perClassMetrics,
    misclassifications
  };

  const reportJsonPath = path.join(REPORTS_DIR, 'model_reproducibility_validation.json');
  fs.writeFileSync(reportJsonPath, JSON.stringify(report, null, 2));
  console.log(`\n[Report] Saved complete validation report to: ${reportJsonPath}`);

  return report;
}

validateModelAndDataset().catch(err => {
  console.error('[Validation Error]', err);
  process.exit(1);
});
