const fs = require('fs');
const path = require('path');
const { ASL_VOCABULARY } = require('../packages/types/dist/index.js');

const classNames = ASL_VOCABULARY.map(v => v.sign);
console.log('Class names count:', classNames.length);

const targets = [
  path.resolve(__dirname, '../ml/models/sbv-gru-v1/metadata.json'),
  path.resolve(__dirname, '../ml/models/sbv-gru-v1-asl/metadata.json'),
  path.resolve(__dirname, '../apps/web/public/models/sbv-gru-v1/metadata.json'),
  path.resolve(__dirname, '../ml/reports/asl_training_summary.json'),
];

for (const p of targets) {
  if (fs.existsSync(p)) {
    const meta = JSON.parse(fs.readFileSync(p, 'utf8'));
    meta.classes = classNames;
    meta.numClasses = classNames.length;
    fs.writeFileSync(p, JSON.stringify(meta, null, 2), 'utf8');
    console.log('Updated:', p);
  }
}
