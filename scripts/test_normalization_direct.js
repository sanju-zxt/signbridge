const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

async function main() {
  await tf.setBackend('cpu');
  await tf.ready();
  const model = await tf.loadLayersModel('http://localhost:3000/models/sbv-gru-v1/model.json');
  const meta = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../apps/web/public/models/sbv-gru-v1/metadata.json'), 'utf8'));

  const testDir = path.resolve(__dirname, '../ml/datasets/isl/test');
  const signsToTest = ['HELLO', 'YOU', 'NAME', 'WHAT', 'HELP', 'THANK_YOU', 'HOSPITAL', 'WATER', 'GO', 'FOOD'];

  console.log('='.repeat(80));
  console.log('SIGNBRIDGE V16: MODEL PREDICTION AUDIT (CORRECT VS ERRONEOUS NORMALIZATION)');
  console.log('='.repeat(80));

  for (const sign of signsToTest) {
    const file = `${sign}_user_09_expressive.json`;
    const filePath = path.join(testDir, file);
    if (!fs.existsSync(filePath)) {
      console.log(`File for ${sign} not found at ${filePath}`);
      continue;
    }
    const sample = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const trueSign = sample.sign;

    // 1. Correct feature matrix (as trained)
    const input1 = tf.tensor3d([sample.featureMatrix]);
    const p1 = model.predict(input1);
    const probs1 = await p1.data();
    const maxIdx1 = probs1.indexOf(Math.max(...probs1));
    const pred1 = meta.classes[maxIdx1];
    const conf1 = probs1[maxIdx1];

    // 2. Erroneous runtime feature matrix ((x - x0) * 2.0, zero wrist)
    const wrongMat = sample.featureMatrix.map(row => {
      const r = [...row];
      r[0] = 0; r[1] = 0; r[2] = 0;
      r[63] = 0; r[64] = 0; r[65] = 0;
      for (let i = 0; i < 126; i++) r[i] *= 0.15;
      return r;
    });
    const input2 = tf.tensor3d([wrongMat]);
    const p2 = model.predict(input2);
    const probs2 = await p2.data();
    const maxIdx2 = probs2.indexOf(Math.max(...probs2));
    const pred2 = meta.classes[maxIdx2];
    const conf2 = probs2[maxIdx2];

    const match = pred1 === trueSign ? 'MATCH' : 'MISMATCH';
    console.log(`True: ${trueSign.padEnd(12)} | Correct: ${pred1.padEnd(12)} (${(conf1*100).toFixed(1)}%) [${match}] | Wrong: ${pred2.padEnd(12)} (${(conf2*100).toFixed(1)}%)`);
  }
}
main().catch(console.error);
