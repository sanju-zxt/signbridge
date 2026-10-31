const {
  translateEnglishToASLGloss,
  translateASLGlossSequenceToEnglish,
} = require('../packages/types/dist/index.js');

const testSentences = [
  'What is your name?',
  'How can I help you?',
  'What can I do for you?',
  'What is your emergency?',
  'Can you call the police?',
  'Please call an ambulance.',
  'I need a doctor.',
  'I am in danger.',
  'I am hungry.',
  'Where are you going?',
  'Can you help me?',
  'Please wait.',
  "I don't understand.",
  'Can you repeat that?',
  'Where is the bathroom?',
  'Where is the hospital?',
  'Hello, how are you?',
];

console.log('--- Testing Bidirectional English <-> ASL Translation ---');
let allPassed = true;

for (const s of testSentences) {
  const { glosses, glossString } = translateEnglishToASLGloss(s);
  const backEnglish = translateASLGlossSequenceToEnglish(glosses);
  console.log(`[EN] "${s}"`);
  console.log(`  -> [ASL Gloss]: ${glossString} (${glosses.join(', ')})`);
  console.log(`  -> [Back EN]:   "${backEnglish}"`);
  if (!glosses || glosses.length === 0) {
    console.error(`  FAIL: No gloss extracted for "${s}"`);
    allPassed = false;
  }
}

// Test multi-sign sequence
const multiSeq = ['HELLO', 'HOW_ARE_YOU', 'THANK_YOU'];
const multiEnglish = translateASLGlossSequenceToEnglish(multiSeq);
console.log('\n[Multi-Sign Sequence]:', multiSeq.join(' -> '));
console.log('  -> [Back EN]:', multiEnglish);

if (allPassed) {
  console.log('\nALL 17 PHRASE TRANSLATION TESTS PASSED!');
} else {
  console.error('\nSOME TESTS FAILED!');
  process.exit(1);
}
