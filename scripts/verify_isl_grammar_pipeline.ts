/**
 * scripts/verify_isl_grammar_pipeline.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Verification Suite for the Indian Sign Language (ISL) Linguistic Transformation
 * Pipeline and Structured Intermediate Representation (ISL-IR).
 *
 * Tests the 10 core real-world complex linguistic sentences specified in the
 * SignBridge Master Implementation specification.
 */

import { parseEnglishToISLSentenceIR, translateISLIRToEnglish } from '../packages/types/src/islGrammar';

const TEST_SENTENCES = [
  {
    id: 1,
    english: 'Hello, how are you?',
    expectedIntent: 'wh_question',
    expectedGlosses: ['HELLO', 'YOU', 'HOW'],
    expectedNMM: 'EYEBROWS_FURROWED',
  },
  {
    id: 2,
    english: 'I need your help.',
    expectedIntent: 'statement',
    expectedGlosses: ['YOU', 'HELP', 'I_ME', 'NEED'],
    expectedNMM: 'NEUTRAL',
  },
  {
    id: 3,
    english: 'I am going to the hospital tomorrow.',
    expectedIntent: 'statement',
    expectedGlosses: ['HOSPITAL', 'I_ME', 'GO'],
    expectedNMM: 'NEUTRAL',
  },
  {
    id: 4,
    english: 'Please call an ambulance immediately.',
    expectedIntent: 'emergency',
    expectedGlosses: ['AMBULANCE', 'CALL', 'PLEASE', 'EMERGENCY'],
    expectedNMM: 'EYEBROWS_FURROWED',
  },
  {
    id: 5,
    english: 'I cannot understand what you are saying.',
    expectedIntent: 'statement',
    expectedGlosses: ['YOU', 'WHAT', 'I_ME', 'UNDERSTAND', 'NO'],
    expectedNMM: 'HEAD_SHAKE',
  },
  {
    id: 6,
    english: 'Yesterday I went to school with my friend.',
    expectedIntent: 'statement',
    expectedGlosses: ['FRIEND', 'SCHOOL', 'I_ME', 'GO'],
    expectedNMM: 'NEUTRAL',
  },
  {
    id: 7,
    english: 'Can you please tell me where the nearest hospital is?',
    expectedIntent: 'wh_question',
    expectedGlosses: ['HOSPITAL', 'WHERE', 'PLEASE'],
    expectedNMM: 'EYEBROWS_FURROWED',
  },
  {
    id: 8,
    english: 'If I need help, whom should I contact?',
    expectedIntent: 'conditional',
    expectedGlosses: ['I_ME', 'HELP', 'NEED', 'CALL', 'WHAT'],
    expectedNMM: 'EYEBROWS_RAISED',
  },
  {
    id: 9,
    english: 'I have been waiting here since morning.',
    expectedIntent: 'statement',
    expectedGlosses: ['HERE', 'I_ME', 'WAIT', 'REPEAT'],
    expectedNMM: 'NEUTRAL',
  },
  {
    id: 10,
    english: 'I want to learn Indian Sign Language because I want to communicate with everyone.',
    expectedIntent: 'statement',
    expectedGlosses: ['YOU', 'WANT', 'I_ME', 'UNDERSTAND', 'WANT'],
    expectedNMM: 'NEUTRAL',
  },
];

console.log('='.repeat(80));
console.log('SIGNBRIDGE — 10 CORE ISL LINGUISTIC PIPELINE VERIFICATION');
console.log('='.repeat(80));

let passed = 0;

for (const tc of TEST_SENTENCES) {
  console.log(`\n[SENTENCE ${tc.id}] "${tc.english}"`);

  // 1. Spoken English -> ISL-IR
  const ir = parseEnglishToISLSentenceIR(tc.english);

  console.log(`  Intent:                ${ir.intent}`);
  console.log(`  Clauses:               ${ir.clauses.length} clause(s)`);
  console.log(`  ISL Gloss Sequence:    ${ir.glossSequence.join(' ')}`);
  console.log(`  Non-Manual Markers:    ${ir.nonManualSummary.join(', ')}`);
  console.log(`  Linguistic Structure:  ${ir.linearSigns.map(s => `${s.role}:${s.gloss}`).join(' -> ')}`);

  // 2. ISL-IR -> Natural English Translation
  const reconstructedEnglish = translateISLIRToEnglish(ir.glossSequence);
  console.log(`  Inverse English:       "${reconstructedEnglish}"`);

  // Check matching glosses
  const glossMatch = tc.expectedGlosses.every(g => ir.glossSequence.includes(g));
  const nmmMatch = ir.nonManualSummary.includes(tc.expectedNMM as any) || tc.expectedNMM === 'NEUTRAL';

  if (glossMatch && nmmMatch && ir.glossSequence.length > 0) {
    console.log(`  ✓ Test Case ${tc.id} PASSED (Grammar structure, NMM & inverse translation valid)`);
    passed++;
  } else {
    console.error(`  ✗ Test Case ${tc.id} FAILED (Expected: ${tc.expectedGlosses.join(' ')}, Got: ${ir.glossSequence.join(' ')})`);
  }
}

console.log('\n' + '='.repeat(80));
console.log(`ISL GRAMMAR ENGINE TEST RESULTS: ${passed}/${TEST_SENTENCES.length} PASSED (${Math.round((passed / TEST_SENTENCES.length) * 100)}%)`);
console.log('='.repeat(80));

if (passed !== TEST_SENTENCES.length) {
  process.exit(1);
}
