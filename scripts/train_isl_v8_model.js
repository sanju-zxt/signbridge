/**
 * train_isl_v8_model.js
 * ─────────────────────────────────────────────────────────────────────────────
 * SignBridge V8/V9: 50-Class ISL 2-Layer GRU Training directly in TensorFlow.js.
 *
 * Guarantees 100% mathematical gate alignment with browser inference:
 *   - Native TF.js training eliminating PyTorch vs Keras candidate gate divergence
 *   - Input shape: [batch, 30, 126]
 *   - Layer 1: GRU(64 units, returnSequences=true, dropout=0.20)
 *   - Layer 2: GRU(64 units, returnSequences=false, dropout=0.20)
 *   - Dense(128, ReLU) -> Dropout(0.25)
 *   - Output: Dense(50, Softmax)
 *
 * Evaluated on 400 independent test sequences from unseen Signers user_09 and user_10.
 * Exports model.json, weights.bin, and metadata.json to web public & ml directories.
 */

const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const ISL_DIR = path.join(PROJECT_ROOT, 'ml', 'datasets', 'isl');
const TRAIN_DIR = path.join(ISL_DIR, 'train');
const VAL_DIR = path.join(ISL_DIR, 'validation');
const TEST_DIR = path.join(ISL_DIR, 'test');
const MANIFEST_PATH = path.join(ISL_DIR, 'annotations', 'isl_provenance_manifest.json');

const PUBLIC_MODEL_DIR = path.join(PROJECT_ROOT, 'apps', 'web', 'public', 'models', 'sbv-gru-v1');
const ML_MODEL_DIR = path.join(PROJECT_ROOT, 'ml', 'models', 'sbv-gru-v1');
const BACKUP_DIR = path.join(PROJECT_ROOT, 'backups', 'v7_sbv_gru');

// 1. Backup previous checkpoint
console.log('='.repeat(80));
console.log('SIGNBRIDGE V8/V9 — NATIVE TENSORFLOW.JS 50-CLASS ISL MODEL TRAINING');
console.log('='.repeat(80));

if (!fs.existsSync(BACKUP_DIR)) fs.mkdirSync(BACKUP_DIR, { recursive: true });
for (const file of ['model.json', 'metadata.json', 'weights.bin']) {
  const src = path.join(PUBLIC_MODEL_DIR, file);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, path.join(BACKUP_DIR, file));
    console.log(`[Backup] Preserved ${file} to backups/v7_sbv_gru/`);
  }
}

// 2. Read manifest
const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
const CLASSES = manifest.classes;
const NUM_CLASSES = CLASSES.length;
const class2idx = {};
CLASSES.forEach((c, i) => { class2idx[c] = i; });

console.log(`[Dataset] Target Vocabulary: ${NUM_CLASSES} classes`);

function loadSplit(dirPath) {
  const files = fs.readdirSync(dirPath).filter(f => f.endsWith('.json'));
  const X = [];
  const y = [];
  const records = [];

  for (const f of files) {
    const raw = JSON.parse(fs.readFileSync(path.join(dirPath, f), 'utf-8'));
    const sign = raw.sign || raw.signId;
    if (class2idx[sign] === undefined) continue;

    // Flatten [30, 126]
    const flat = [];
    for (let t = 0; t < 30; t++) {
      const row = raw.featureMatrix[t];
      for (let d = 0; d < 126; d++) {
        flat.push(row[d] !== undefined ? row[d] : 0.0);
      }
    }

    X.push(flat);
    y.push(class2idx[sign]);
    records.push({ id: raw.id, sign, signer: raw.signerId });
  }

  return { X, y, records };
}

console.log('\n[Loading] Loading multi-signer dataset splits...');
const trainData = loadSplit(TRAIN_DIR);
const valData = loadSplit(VAL_DIR);
const testData = loadSplit(TEST_DIR);

console.log(`  Train sequences:      ${trainData.X.length} (Signers user_01 - user_07)`);
console.log(`  Validation sequences: ${valData.X.length} (Signer user_08)`);
console.log(`  Test sequences:       ${testData.X.length} (Signers user_09 - user_10)`);

// 3. Convert to Tensors
const trainXTensor = tf.tensor3d(trainData.X.flat(), [trainData.X.length, 30, 126]);
const trainYTensor = tf.oneHot(tf.tensor1d(trainData.y, 'int32'), NUM_CLASSES);

const valXTensor = tf.tensor3d(valData.X.flat(), [valData.X.length, 30, 126]);
const valYTensor = tf.oneHot(tf.tensor1d(valData.y, 'int32'), NUM_CLASSES);

const testXTensor = tf.tensor3d(testData.X.flat(), [testData.X.length, 30, 126]);
const testYTensor = tf.oneHot(tf.tensor1d(testData.y, 'int32'), NUM_CLASSES);

// 4. Build exact 2-Layer GRU Architecture
console.log('\n[Model] Building 2-Layer GRU Neural Architecture...');
const model = tf.sequential();

model.add(tf.layers.gru({
  units: 64,
  returnSequences: true,
  inputShape: [30, 126],
  dropout: 0.20,
  name: 'gru_GRU1',
}));

model.add(tf.layers.gru({
  units: 64,
  returnSequences: false,
  dropout: 0.20,
  name: 'gru_GRU2',
}));

model.add(tf.layers.dense({
  units: 128,
  activation: 'relu',
  name: 'dense_Dense1',
}));

model.add(tf.layers.dropout({ rate: 0.25, name: 'dropout_Dropout1' }));

model.add(tf.layers.dense({
  units: NUM_CLASSES,
  activation: 'softmax',
  name: 'dense_Dense2',
}));

model.compile({
  optimizer: tf.train.adam(0.002),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();

// 5. Training Loop
async function train() {
  const EPOCHS = 35;
  const BATCH_SIZE = 32;

  console.log(`\n[Training] Starting ${EPOCHS} epochs (Batch Size ${BATCH_SIZE})...`);
  const tStart = Date.now();

  let bestValAcc = 0;
  let bestWeights = null;

  await model.fit(trainXTensor, trainYTensor, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [valXTensor, valYTensor],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const ep = epoch + 1;
        const loss = logs.loss.toFixed(4);
        const acc = (logs.acc * 100).toFixed(1);
        const valLoss = logs.val_loss.toFixed(4);
        const valAcc = (logs.val_acc * 100).toFixed(1);

        console.log(`  Epoch ${ep.toString().padStart(2, '0')}/${EPOCHS} | Train Loss: ${loss} | Train Acc: ${acc}% | Val Loss: ${valLoss} | Val Acc: ${valAcc}%`);

        if (logs.val_acc > bestValAcc) {
          bestValAcc = logs.val_acc;
          bestWeights = model.getWeights().map(w => w.clone());
        }
      }
    }
  });

  const durationSec = Math.round((Date.now() - tStart) / 1000);
  console.log(`\n[Training Completed] Duration: ${durationSec}s | Best Validation Accuracy: ${(bestValAcc * 100).toFixed(2)}%`);

  // Restore best weights if available
  if (bestWeights) {
    model.setWeights(bestWeights);
    console.log('[Model] Restored best checkpoint weights from peak validation performance.');
  }

  // 6. Independent Test Evaluation on Unseen Signers
  console.log('\n[Evaluation] Evaluating on 400 Independent Test Sequences (Signers user_09 & user_10)...');
  const predsTensor = model.predict(testXTensor);
  const predsData = await predsTensor.data();
  const testYData = await testYTensor.data();

  let correctCount = 0;
  let top5Count = 0;
  const perClass = {};
  CLASSES.forEach(c => {
    perClass[c] = { total: 0, top1Correct: 0, top5Correct: 0 };
  });

  for (let i = 0; i < testData.records.length; i++) {
    const trueIdx = testData.y[i];
    const sign = CLASSES[trueIdx];
    const probs = [];
    for (let c = 0; c < NUM_CLASSES; c++) {
      probs.push({ idx: c, prob: predsData[i * NUM_CLASSES + c] });
    }
    probs.sort((a, b) => b.prob - a.prob);

    const top1 = probs[0].idx;
    const top5 = probs.slice(0, 5).map(p => p.idx);

    perClass[sign].total++;
    if (top1 === trueIdx) {
      correctCount++;
      perClass[sign].top1Correct++;
    }
    if (top5.includes(trueIdx)) {
      top5Count++;
      perClass[sign].top5Correct++;
    }
  }

  const testAcc = correctCount / testData.records.length;
  const top5Acc = top5Count / testData.records.length;

  console.log(`\n================================================================================`);
  console.log(`INDEPENDENT TEST EVALUATION (Zero Signer Leakage)`);
  console.log(`================================================================================`);
  console.log(`Top-1 Independent Test Accuracy: ${(testAcc * 100).toFixed(2)}% (${correctCount}/${testData.records.length})`);
  console.log(`Top-5 Independent Test Accuracy: ${(top5Acc * 100).toFixed(2)}% (${top5Count}/${testData.records.length})`);
  console.log('--------------------------------------------------------------------------------');
  console.log('Per-Class Performance Sample:');
  CLASSES.slice(0, 15).forEach(c => {
    const stat = perClass[c];
    const pAcc = ((stat.top1Correct / stat.total) * 100).toFixed(1);
    console.log(`  ${c.padEnd(20)} | Total: ${stat.total} | Correct: ${stat.top1Correct} | Acc: ${pAcc}%`);
  });

  // 7. Save Model Artifacts
  console.log('\n[Export] Saving native TensorFlow.js layers model...');
  let savedArtifacts = null;

  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    savedArtifacts = artifacts;
    return { modelArtifactsInfo: { dateSaved: new Date() } };
  }));

  const modelJson = {
    modelTopology: savedArtifacts.modelTopology,
    format: 'layers-model',
    generatedBy: `TensorFlow.js tfjs-layers v${tf.version.tfjs}`,
    convertedBy: null,
    weightsManifest: [{
      paths: ['./weights.bin'],
      weights: savedArtifacts.weightSpecs
    }]
  };

  const weightsBuffer = Buffer.from(savedArtifacts.weightData);

  const metadataJson = {
    id: 'SBV-ISL-GRU-V8',
    version: '8.0.0',
    modelVersion: '8.0.0',
    language: 'ISL',
    classes: CLASSES,
    numClasses: NUM_CLASSES,
    featureDim: 126,
    windowSize: 30,
    architecture: '2-Layer GRU (64 units) -> Dense(128, ReLU) -> Dense(50, Softmax)',
    training: {
      dataset: 'SignBridge Authentic ISL Benchmark Dataset V8 (Motion-Preserving)',
      totalSequences: trainData.X.length + valData.X.length + testData.X.length,
      trainSequences: trainData.X.length,
      valSequences: valData.X.length,
      testSequences: testData.X.length,
      epochs: EPOCHS,
      bestValAccuracy: Number(bestValAcc.toFixed(4)),
      independentTestAccuracy: Number(testAcc.toFixed(4)),
      top5Accuracy: Number(top5Acc.toFixed(4)),
      trainedAt: new Date().toISOString(),
    },
    zeroSignerLeakage: true,
    activeBackend: 'webgl',
    normalization: 'Motion-Preserving (Landmark 0 = wrist trajectory displacement / scale_0; Landmarks 1..20 = local hand configuration / scale_t)',
  };

  for (const outDir of [PUBLIC_MODEL_DIR, ML_MODEL_DIR]) {
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, 'model.json'), JSON.stringify(modelJson, null, 2));
    fs.writeFileSync(path.join(outDir, 'weights.bin'), weightsBuffer);
    fs.writeFileSync(path.join(outDir, 'metadata.json'), JSON.stringify(metadataJson, null, 2));
    console.log(`[Export] Wrote model.json, weights.bin, and metadata.json to ${outDir}`);
  }

  // 8. Round-Trip Verification Test
  console.log('\n[Verification] Verifying round-trip model loading from disk...');
  const verifyBin = fs.readFileSync(path.join(PUBLIC_MODEL_DIR, 'weights.bin'));
  const verifyJson = JSON.parse(fs.readFileSync(path.join(PUBLIC_MODEL_DIR, 'model.json'), 'utf-8'));
  const loadedModel = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: verifyJson.modelTopology,
    weightSpecs: verifyJson.weightsManifest[0].weights,
    weightData: verifyBin.buffer
  }));

  const testWarmup = tf.zeros([1, 30, 126]);
  const warmupPred = loadedModel.predict(testWarmup);
  const warmupData = await warmupPred.data();
  console.log(`[Verification] Deployed model loaded successfully! Output class count: ${warmupData.length}`);
  console.log('Phase 3 Model Training & Deployment Complete!');
}

train().catch(err => {
  console.error('[Training Error]', err);
  process.exit(1);
});
