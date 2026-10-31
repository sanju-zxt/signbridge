/**
 * verify_v10_advanced_bidirectional.ts
 * Comprehensive verification of the SignBridge Advanced Bidirectional ISL Communication Engine.
 * Tests:
 * 1. 8 Mandatory Acceptance Test Sentences (Speech-to-ISL IR & ISL-to-Speech)
 * 2. Extended Vocabulary (Categories A through I + Alphabet) and honest classification
 * 3. Avatar Animation Controller & Sequence Queuing
 * 4. Neural Model Metadata & Provenance
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  parseEnglishToISLSentenceIR,
  translateISLIRToEnglish,
  transformClauseToISL,
} from '../packages/types/src/islGrammar';
import {
  ISL_EXPANDED_VOCABULARY,
  ISL_TRAINED_MODEL_CLASSES,
  translateISLGlossSequenceToEnglish,
  translateEnglishToISLGloss,
} from '../packages/types/src/islVocabulary';

interface TestCase {
  id: number;
  english: string;
  expectedTokens: string[];
  expectedIntent?: string;
  expectedNMM?: string;
  expectedEnglishOutput: string;
}

const MANDATORY_TEST_CASES: TestCase[] = [
  {
    id: 1,
    english: 'What are you doing?',
    expectedTokens: ['YOU', 'DO', 'WHAT'],
    expectedIntent: 'wh_question',
    expectedNMM: 'EYEBROWS_FURROWED',
    expectedEnglishOutput: 'What are you doing?',
  },
  {
    id: 2,
    english: 'What is your name?',
    expectedTokens: ['YOUR', 'NAME', 'WHAT'],
    expectedIntent: 'wh_question',
    expectedNMM: 'EYEBROWS_FURROWED',
    expectedEnglishOutput: 'What is your name?',
  },
  {
    id: 3,
    english: 'How can I help you?',
    expectedTokens: ['I_ME', 'YOU', 'HELP', 'HOW'],
    expectedIntent: 'wh_question',
    expectedNMM: 'EYEBROWS_FURROWED',
    expectedEnglishOutput: 'How can I help you?',
  },
  {
    id: 4,
    english: 'My name is Daniel.',
    expectedTokens: ['MY', 'NAME', 'D', 'A', 'N', 'I', 'E', 'L'],
    expectedIntent: 'statement',
    expectedEnglishOutput: 'My name is Daniel.',
  },
  {
    id: 5,
    english: 'What are you doing right now?',
    expectedTokens: ['NOW', 'YOU', 'DO', 'WHAT'],
    expectedIntent: 'wh_question',
    expectedNMM: 'EYEBROWS_FURROWED',
    expectedEnglishOutput: 'What are you doing right now?',
  },
  {
    id: 6,
    english: 'I need your help.',
    expectedTokens: ['I_ME', 'YOUR', 'HELP', 'NEED'],
    expectedIntent: 'statement',
    expectedEnglishOutput: 'I need your help.',
  },
  {
    id: 7,
    english: 'I am going to the hospital.',
    expectedTokens: ['I_ME', 'HOSPITAL', 'GO'],
    expectedIntent: 'statement',
    expectedEnglishOutput: 'I am going to the hospital.',
  },
  {
    id: 8,
    english: 'Thank you for helping me.',
    expectedTokens: ['YOU', 'I_ME', 'HELP', 'THANK_YOU'],
    expectedIntent: 'statement',
    expectedEnglishOutput: 'Thank you for helping me.',
  },
];

async function runVerification() {
  console.log('================================================================');
  console.log(' SIGNBRIDGE V10 — ADVANCED BIDIRECTIONAL ISL COMM ENGINE VERIFICATION');
  console.log('================================================================\n');

  let passedTests = 0;
  let totalTests = 0;

  // --------------------------------------------------------------------------
  // TEST SECTION 1: 8 Mandatory Acceptance Sentences (Speech to ISL-IR)
  // --------------------------------------------------------------------------
  console.log('--- TEST SECTION 1: SPEECH TO ISL-IR & TOKEN SEQUENCES ---');
  for (const tc of MANDATORY_TEST_CASES) {
    totalTests++;
    const ir = parseEnglishToISLSentenceIR(tc.english);
    const tokens = ir.glossSequence;

    const tokenMatch = JSON.stringify(tokens) === JSON.stringify(tc.expectedTokens);
    const intentMatch = tc.expectedIntent ? ir.intent === tc.expectedIntent : true;
    const nmmMatch = tc.expectedNMM ? ir.nonManualSummary.includes(tc.expectedNMM as any) : true;

    if (tokenMatch && intentMatch && nmmMatch) {
      passedTests++;
      console.log(`[PASS] Test ${tc.id}: "${tc.english}"`);
      console.log(`       Tokens: [${tokens.join(', ')}]`);
      console.log(`       Intent: ${ir.intent} | NMMs: [${ir.nonManualSummary.join(', ')}]`);
    } else {
      console.error(`[FAIL] Test ${tc.id}: "${tc.english}"`);
      console.error(`       Expected: [${tc.expectedTokens.join(', ')}]`);
      console.error(`       Actual:   [${tokens.join(', ')}]`);
      console.error(`       IR:`, JSON.stringify(ir, null, 2));
    }
  }

  // --------------------------------------------------------------------------
  // TEST SECTION 2: 8 Mandatory Acceptance Sentences (ISL to English Speech)
  // --------------------------------------------------------------------------
  console.log('\n--- TEST SECTION 2: ISL GLOSS SEQUENCE TO NATURAL ENGLISH SPEECH ---');
  for (const tc of MANDATORY_TEST_CASES) {
    totalTests++;
    const englishSynth = translateISLGlossSequenceToEnglish(tc.expectedTokens);
    const match = englishSynth.toLowerCase().replace(/[.?!]/g, '') === tc.expectedEnglishOutput.toLowerCase().replace(/[.?!]/g, '');

    if (match) {
      passedTests++;
      console.log(`[PASS] Test ${tc.id}: [${tc.expectedTokens.join(', ')}] -> "${englishSynth}"`);
    } else {
      console.error(`[FAIL] Test ${tc.id}: [${tc.expectedTokens.join(', ')}]`);
      console.error(`       Expected: "${tc.expectedEnglishOutput}"`);
      console.error(`       Actual:   "${englishSynth}"`);
    }
  }

  // --------------------------------------------------------------------------
  // TEST SECTION 3: Extended Vocabulary Coverage & Category Distribution
  // --------------------------------------------------------------------------
  console.log('\n--- TEST SECTION 3: EXTENDED VOCABULARY AUDIT ---');
  totalTests++;
  const vocabCount = ISL_EXPANDED_VOCABULARY.length;
  const trainedCount = ISL_TRAINED_MODEL_CLASSES.length;
  console.log(`Total Extended Vocabulary Concepts: ${vocabCount}`);
  console.log(`Trained Neural Model Classes:        ${trainedCount}`);

  const categoryBreakdown: Record<string, number> = {};
  const statusBreakdown: Record<string, number> = {};

  for (const item of ISL_EXPANDED_VOCABULARY) {
    const cat = item.categoryGroup ?? 'Group';
    const status = item.isTrainedModelClass ? 'trained-neural-model' : item.animationStatus;
    categoryBreakdown[cat] = (categoryBreakdown[cat] || 0) + 1;
    statusBreakdown[status] = (statusBreakdown[status] || 0) + 1;
  }

  console.log('Category Distribution (Groups A through I):');
  for (const [cat, count] of Object.entries(categoryBreakdown)) {
    console.log(`  - Category Group ${cat}: ${count} concepts`);
  }
  console.log('Implementation Status Breakdown:');
  for (const [st, count] of Object.entries(statusBreakdown)) {
    console.log(`  - ${st}: ${count} concepts`);
  }

  if (vocabCount >= 80 && trainedCount === 50) {
    passedTests++;
    console.log('[PASS] Vocabulary coverage target met (>=80 concepts, 50 neural classes)');
  } else {
    console.error(`[FAIL] Vocabulary coverage target not met: count=${vocabCount}, trained=${trainedCount}`);
  }

  // --------------------------------------------------------------------------
  // TEST SECTION 4: Animation Controller Procedural & Queuing Capabilities
  // --------------------------------------------------------------------------
  console.log('\n--- TEST SECTION 4: ANIMATION CONTROLLER VERIFICATION ---');
  totalTests++;
  const animControllerPath = path.resolve(__dirname, '../apps/web/src/components/avatar/AnimationController.ts');
  const animControllerSrc = fs.readFileSync(animControllerPath, 'utf-8');

  const hasNow = animControllerSrc.includes("registerGesture('NOW',") || animControllerSrc.includes("PROC_NOW");
  const hasYour = animControllerSrc.includes("registerGesture('YOUR',") || animControllerSrc.includes("PROC_YOUR");
  const hasDo = animControllerSrc.includes("registerGesture('DO',") || animControllerSrc.includes("PROC_DO");
  const hasDanielLetters = ['D', 'A', 'N', 'I', 'E', 'L'].every((l) =>
    animControllerSrc.includes(`'LETTER_${l}'`) || animControllerSrc.includes(`'${l}'`)
  );
  const queueFix = animControllerSrc.includes('executionManager.getQueueLength() === 0');

  if (hasNow && hasYour && hasDo && hasDanielLetters && queueFix) {
    passedTests++;
    console.log('[PASS] AnimationController has NOW, YOUR, DO, D-A-N-I-E-L letters, and zero premature idle return.');
  } else {
    console.error('[FAIL] AnimationController missing procedural extensions or queue fix:');
    console.error(`       NOW: ${hasNow}, YOUR: ${hasYour}, DO: ${hasDo}, Letters: ${hasDanielLetters}, QueueFix: ${queueFix}`);
  }

  // --------------------------------------------------------------------------
  // TEST SECTION 5: Neural Model Metadata and Provenance
  // --------------------------------------------------------------------------
  console.log('\n--- TEST SECTION 5: NEURAL MODEL METADATA & INTEGRITY ---');
  totalTests++;
  const metadataPath = path.resolve(__dirname, '../apps/web/public/models/sbv-gru-v1/metadata.json');
  if (fs.existsSync(metadataPath)) {
    const meta = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
    const testAcc = meta.training?.independentTestAccuracy ?? meta.testAccuracy ?? 0;
    console.log(`Model ID:            ${meta.id}`);
    console.log(`Classes Count:       ${meta.numClasses}`);
    console.log(`Test Set Accuracy:   ${(testAcc * 100).toFixed(2)}%`);
    console.log(`Window Size:         ${meta.windowSize}`);
    console.log(`Feature Dimension:   ${meta.featureDim}`);

    if (meta.numClasses === 50 && testAcc >= 0.95 && meta.featureDim === 126) {
      passedTests++;
      console.log('[PASS] Model metadata matches production specifications.');
    } else {
      console.error('[FAIL] Model metadata unexpected values.');
    }
  } else {
    console.error(`[FAIL] metadata.json not found at ${metadataPath}`);
  }

  // --------------------------------------------------------------------------
  // SUMMARY
  // --------------------------------------------------------------------------
  console.log('\n================================================================');
  console.log(` VERIFICATION SUMMARY: ${passedTests}/${totalTests} TESTS PASSED`);
  console.log('================================================================\n');

  if (passedTests === totalTests) {
    process.exit(0);
  } else {
    process.exit(1);
  }
}

runVerification().catch((err) => {
  console.error('Fatal error running verification:', err);
  process.exit(1);
});
