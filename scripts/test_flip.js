const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

async function testFlip() {
  await tf.setBackend('cpu');
  await tf.ready();
  const model = await tf.loadLayersModel('http://localhost:3000/models/sbv-gru-v1/model.json');
  const meta = JSON.parse(fs.readFileSync('apps/web/public/models/sbv-gru-v1/metadata.json', 'utf8'));

  const signs = ['HELLO', 'YOU', 'WATER', 'HELP', 'HOSPITAL', 'NAME'];
  for (const sign of signs) {
    const testFile = `ml/datasets/isl/test/${sign}_user_09_expressive.json`;
    if (!fs.existsSync(testFile)) continue;
    const sample = JSON.parse(fs.readFileSync(testFile, 'utf8'));

    // Normal input
    const inputNorm = tf.tensor3d([sample.featureMatrix]);
    const pNorm = model.predict(inputNorm);
    const probsNorm = await pNorm.data();
    const maxNorm = probsNorm.indexOf(Math.max(...probsNorm));

    // Invert X
    const flippedMat = sample.featureMatrix.map(row => {
      const r = [...row];
      for (let i = 0; i < 42; i++) {
        r[i * 3 + 0] = -r[i * 3 + 0];
      }
      return r;
    });
    const inputFlip = tf.tensor3d([flippedMat]);
    const pFlip = model.predict(inputFlip);
    const probsFlip = await pFlip.data();
    const maxFlip = probsFlip.indexOf(Math.max(...probsFlip));

    console.log(`Sign: ${sign.padEnd(10)} | Original: ${meta.classes[maxNorm].padEnd(10)} (${(probsNorm[maxNorm]*100).toFixed(1)}%) | Flipped: ${meta.classes[maxFlip].padEnd(10)} (${(probsFlip[maxFlip]*100).toFixed(1)}%)`);
  }
}
testFlip().catch(console.error);
