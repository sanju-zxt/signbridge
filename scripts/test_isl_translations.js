const { translateISLGlossSequenceToEnglish } = require('../packages/types/dist/index.js');

const testCases = [
  ['HELLO'],
  ['YOU'],
  ['NAME'],
  ['WHAT'],
  ['HELP'],
  ['THANK_YOU'],
  ['HOSPITAL'],
  ['WATER'],
  ['YOU', 'DO', 'WHAT'],
  ['YOUR', 'NAME', 'WHAT'],
  ['YOU', 'NAME', 'WHAT'],
  ['I_ME', 'YOU', 'HELP', 'HOW'],
  ['YOU', 'HELP', 'HOW'],
  ['I_ME', 'HOSPITAL', 'GO'],
  ['YOU', 'WANT', 'WATER']
];

console.log('='.repeat(80));
console.log('ISL GLOSS SEQUENCE -> NATURAL ENGLISH TRANSLATIONS');
console.log('='.repeat(80));

for (const tc of testCases) {
  const result = translateISLGlossSequenceToEnglish(tc);
  console.log(tc.join(' ').padEnd(25), '->', result);
}
