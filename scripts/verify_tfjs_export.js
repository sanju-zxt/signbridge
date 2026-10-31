const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

async function testExportedModel() {
  console.log('--- Testing Exported TF.js Model Parity & Loading ---');
  const modelDir = path.resolve(__dirname, '../apps/web/public/models/sbv-gru-v1');
  const modelJsonPath = path.join(modelDir, 'model.json');
  const metadataPath = path.join(modelDir, 'metadata.json');

  if (!fs.existsSync(modelJsonPath) || !fs.existsSync(metadataPath)) {
    console.error('Error: model.json or metadata.json not found in', modelDir);
    process.exit(1);
  }

  const meta = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  console.log('Metadata ID:', meta.id);
  console.log('Number of classes:', meta.numClasses);
  console.log('Feature dimensions:', meta.featureDim);
  console.log('Window size:', meta.windowSize);

  const t0 = performance.now();
  const ioHandler = {
    load: async () => {
      const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
      const weightsBin = fs.readFileSync(path.join(modelDir, 'weights.bin'));
      return {
        modelTopology: modelJson.modelTopology,
        weightSpecs: modelJson.weightsManifest[0].weights,
        weightData: weightsBin.buffer.slice(weightsBin.byteOffset, weightsBin.byteOffset + weightsBin.byteLength),
      };
    }
  };
  const model = await tf.loadLayersModel(ioHandler);
  const loadDuration = performance.now() - t0;
  console.log(`TF.js Model Loaded in ${loadDuration.toFixed(2)} ms.`);

  // Test inference with a sample from test set
  const testSamplePath = path.resolve(__dirname, '../ml/datasets/asl/test');
  const testFiles = fs.readdirSync(testSamplePath).filter(f => f.endsWith('.json'));
  const testData = JSON.parse(fs.readFileSync(path.join(testSamplePath, testFiles[0]), 'utf8'));

  const inputMatrix = testData.featureMatrix; // 30 x 126
  const tensor = tf.tensor3d([inputMatrix]);

  const tInfer0 = performance.now();
  const predsTensor = model.predict(tensor);
  const probs = await predsTensor.data();
  const inferDuration = performance.now() - tInfer0;

  console.log(`Inference executed in ${inferDuration.toFixed(2)} ms.`);
  console.log(`Output probabilities length: ${probs.length} (Expected: ${meta.numClasses})`);

  let maxProb = 0;
  let maxIdx = -1;
  let probSum = 0;
  for (let i = 0; i < probs.length; i++) {
    probSum += probs[i];
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      maxIdx = i;
    }
  }

  console.log(`Softmax sum: ${probSum.toFixed(4)} (Expected ~1.0)`);
  console.log(`Top Predicted Class Index: ${maxIdx} (${meta.classes[maxIdx]}), Confidence: ${(maxProb * 100).toFixed(2)}%`);
  console.log(`Ground Truth Class Index: ${testData.classIndex} (${testData.sign})`);

  if (probs.length === meta.numClasses && Math.abs(probSum - 1.0) < 0.01) {
    console.log('TF.JS MODEL VERIFICATION: SUCCESS!');
  } else {
    console.error('TF.JS MODEL VERIFICATION: FAILED!');
    process.exit(1);
  }
}

testExportedModel().catch(err => {
  console.error('Test error:', err);
  process.exit(1);
});
