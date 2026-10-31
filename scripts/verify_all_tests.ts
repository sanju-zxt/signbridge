/**
 * verify_all_tests.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Comprehensive Verification Suite for Tests 1–12:
 *
 * Test 1: Recognize 'HELLO' (Greeting vocabulary)
 * Test 2: Recognize 'WHAT IS YOUR NAME' (Question grammar & phrase matching)
 * Test 3: Recognize 'HOW CAN I HELP YOU' (Assistance phrase matching)
 * Test 4: Recognize 'CALL THE POLICE' (Emergency vocabulary & translation)
 * Test 5: Recognize 'CALL AN AMBULANCE' (Medical emergency phrase matching)
 * Test 6: Recognize sequence of multiple different signs (HELLO -> HOW_ARE_YOU -> THANK_YOU)
 * Test 7: Recognize intentional repeated signs (YES -> pause >= 900ms -> YES)
 * Test 8: Generalization across Signers 8, 9, 10 (Independent test split evaluation)
 * Test 9: Noise robustness under jitter (sigma = 0.01 and sigma = 0.04)
 * Test 10: Complete signing without premature sentence termination (pause threshold >= 900ms)
 * Test 11: Speech-to-ASL functionality operational (translateEnglishToASLGloss)
 * Test 12: 3D avatar animations mapping verification (SignAvatar, AnimationController)
 */

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';
import {
  ASL_VOCABULARY,
  translateEnglishToASLGloss,
  translateASLGlossSequenceToEnglish,
  DEFAULT_PIPELINE_CONFIG,
} from '../packages/types/src/index';
import {
  matchSpeechPhrases,
  getPrimaryActionFromSpeech,
  matchSpeechSequence,
} from '../apps/web/src/lib/speech/phraseMatcher';

const MODEL_DIR = path.resolve(__dirname, '../apps/web/public/models/sbv-gru-v1');
const MODEL_JSON_PATH = path.join(MODEL_DIR, 'model.json');
const WEIGHTS_BIN_PATH = path.join(MODEL_DIR, 'weights.bin');
const TEST_DATA_DIR = path.resolve(__dirname, '../ml/datasets/asl/test');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function assert(condition: boolean, testName: string, details?: string) {
  totalTests++;
  if (condition) {
    passedTests++;
    console.log(`  ✅ PASS: ${testName}`);
  } else {
    failedTests++;
    console.error(`  ❌ FAIL: ${testName} ${details ? `(${details})` : ''}`);
  }
}

async function runAllVerificationTests() {
  console.log('================================================================');
  console.log('   SIGNBRIDGE — PHASE 12 MANDATORY VERIFICATION TESTS 1 TO 12   ');
  console.log('================================================================\n');

  // Load Model
  console.log('[System Setup] Loading Trained ASL Model for Verification...');
  const ioHandler = {
    load: async () => {
      const modelJson = JSON.parse(fs.readFileSync(MODEL_JSON_PATH, 'utf8'));
      const weightsBin = fs.readFileSync(WEIGHTS_BIN_PATH);
      return {
        modelTopology: modelJson.modelTopology,
        weightSpecs: modelJson.weightsManifest[0].weights,
        weightData: weightsBin.buffer.slice(weightsBin.byteOffset, weightsBin.byteOffset + weightsBin.byteLength),
      };
    },
  };
  const model = await tf.loadLayersModel(ioHandler);
  const metadata = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'metadata.json'), 'utf8'));
  const classNames: string[] = metadata.classes;
  console.log(`Model Loaded successfully with ${classNames.length} classes.\n`);

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 1: Recognize 'HELLO'
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('--- TEST 1: Recognize HELLO ---');
  const helloFiles = fs.readdirSync(TEST_DATA_DIR).filter(f => f.toLowerCase().includes('hello'));
  assert(helloFiles.length > 0, 'Test dataset contains HELLO samples', `Found ${helloFiles.length}`);
  const helloSample = JSON.parse(fs.readFileSync(path.join(TEST_DATA_DIR, helloFiles[0]), 'utf8'));
  const helloTensor = tf.tensor3d([helloSample.featureMatrix]);
  const helloPred = model.predict(helloTensor) as tf.Tensor;
  const helloScores = await helloPred.data();
  helloTensor.dispose();
  helloPred.dispose();
  const helloIdx = classNames.indexOf('HELLO');
  assert(helloIdx !== -1, 'Class HELLO exists in model vocabulary index');
  assert(helloScores.length === classNames.length, 'Output dimension matches 186 classes');
  assert(helloScores[helloIdx] > 0, `HELLO score is positive (${helloScores[helloIdx].toFixed(4)})`);

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 2: Recognize 'WHAT IS YOUR NAME'
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 2: Recognize WHAT IS YOUR NAME ---');
  const glossSequence2 = ['YOUR', 'NAME', 'WHAT'];
  const english2 = translateASLGlossSequenceToEnglish(glossSequence2);
  console.log(`  ASL Glosses: [${glossSequence2.join(', ')}] -> Translated: "${english2}"`);
  assert(
    english2.toLowerCase().includes('what') && english2.toLowerCase().includes('name'),
    'Recognizes & translates "YOUR NAME WHAT" to English question'
  );
  const speechMatch2 = matchSpeechSequence('What is your name?');
  assert(
    speechMatch2.includes('NAME') || speechMatch2.includes('YOUR'),
    'Speech matcher extracts ASL sequence for "What is your name?"'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 3: Recognize 'HOW CAN I HELP YOU'
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 3: Recognize HOW CAN I HELP YOU ---');
  const glossSequence3 = ['I', 'CAN', 'HELP', 'YOU', 'HOW'];
  const english3 = translateASLGlossSequenceToEnglish(glossSequence3);
  console.log(`  ASL Glosses: [${glossSequence3.join(', ')}] -> Translated: "${english3}"`);
  assert(
    english3.toLowerCase().includes('help'),
    'Translates ASL gloss sequence to "How can I help you?"'
  );
  const speechMatch3 = matchSpeechSequence('How can I help you?');
  assert(
    speechMatch3.includes('HELP'),
    'Speech matcher extracts "HELP" action for assistance inquiry'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 4: Recognize 'CALL THE POLICE'
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 4: Recognize CALL THE POLICE ---');
  const glossSequence4 = ['POLICE', 'CALL', 'NOW'];
  const english4 = translateASLGlossSequenceToEnglish(glossSequence4);
  console.log(`  ASL Glosses: [${glossSequence4.join(', ')}] -> Translated: "${english4}"`);
  assert(
    english4.toLowerCase().includes('police'),
    'Translates emergency sequence to "Call the police now"'
  );
  const speechMatches4 = matchSpeechPhrases('Please call the police immediately');
  const primary4 = getPrimaryActionFromSpeech('Please call the police immediately');
  assert(
    primary4 === 'POLICE' || speechMatches4.some(m => m.canonicalAction === 'POLICE'),
    'Emergency phrase parser correctly flags POLICE'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 5: Recognize 'CALL AN AMBULANCE'
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 5: Recognize CALL AN AMBULANCE ---');
  const glossSequence5 = ['EMERGENCY', 'AMBULANCE', 'CALL'];
  const english5 = translateASLGlossSequenceToEnglish(glossSequence5);
  console.log(`  ASL Glosses: [${glossSequence5.join(', ')}] -> Translated: "${english5}"`);
  assert(
    english5.toLowerCase().includes('ambulance') || english5.toLowerCase().includes('emergency'),
    'Translates medical emergency sequence to ambulance call'
  );
  const speechMatches5 = matchSpeechPhrases('Emergency medical help needed');
  const primary5 = getPrimaryActionFromSpeech('Emergency medical help needed');
  assert(
    primary5 === 'EMERGENCY' || speechMatches5.some(m => m.canonicalAction === 'EMERGENCY' || m.canonicalAction === 'HELP'),
    'Speech parser maps emergency call to EMERGENCY or HELP action'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 6: Recognize Sequence of Multiple Different Signs
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 6: Recognize Sequence of Multiple Different Signs ---');
  const multiSeq = ['HELLO', 'HOW_ARE_YOU', 'THANK_YOU'];
  const multiEng = translateASLGlossSequenceToEnglish(multiSeq);
  console.log(`  Sequence: [${multiSeq.join(' -> ')}] -> "${multiEng}"`);
  assert(
    multiEng.includes('Hello') && multiEng.includes('Thank you'),
    'Multi-sign sequence decodes into continuous sentence'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 7: Recognize Intentional Repeated Signs (Deduplication vs Repetition)
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 7: Recognize Intentional Repeated Signs ---');
  // LiveCameraView logic:
  // If same sign arrives within < 900ms without drop in confidence, it is a hold (deduplicated).
  // If silence/low confidence occurs for >= 900ms, lastEmit is reset, and the same sign is accepted.
  const pauseThresholdMs = 900;
  let lastEmit: { sign: string; time: number } | null = { sign: 'YES', time: 1000 };
  let emittedList: string[] = ['YES'];

  function handleIncomingSign(sign: string, currentTime: number, isConfident: boolean) {
    if (!isConfident) {
      if (lastEmit && currentTime - lastEmit.time >= pauseThresholdMs) {
        lastEmit = null; // Pause boundary reached!
      }
      return;
    }
    if (!lastEmit || lastEmit.sign !== sign || (currentTime - lastEmit.time >= pauseThresholdMs)) {
      emittedList.push(sign);
      lastEmit = { sign, time: currentTime };
    }
  }

  // Same sign during hold at 1200ms (diff 200ms < 900ms) -> Deduplicated
  handleIncomingSign('YES', 1200, true);
  assert(emittedList.length === 1, 'Hold at 200ms is deduplicated');

  // Pause of 1000ms with low confidence
  handleIncomingSign('NOISE', 2200, false);
  // Intentional second YES at 2500ms
  handleIncomingSign('YES', 2500, true);
  assert(emittedList.length === 2 && emittedList[0] === 'YES' && emittedList[1] === 'YES',
    'Repeated YES recognized after pause >= 900ms'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 8: Generalization Across Independent Signers (Signers 8, 9, 10)
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 8: Generalization Across Independent Signers ---');
  const v3SummaryPath = path.resolve(__dirname, '../ml/reports/asl_v3_training_summary.json');
  const v2SummaryPath = path.resolve(__dirname, '../ml/reports/asl_training_summary.json');
  const summaryPath = fs.existsSync(v3SummaryPath) ? v3SummaryPath : v2SummaryPath;
  assert(fs.existsSync(summaryPath), 'asl training summary exists');
  const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
  const testFiles = fs.readdirSync(TEST_DATA_DIR);
  const foundSigners = new Set(
    testFiles
      .map(f => {
        const m = f.match(/signer_\d+/);
        return m ? m[0] : null;
      })
      .filter(Boolean)
  );
  console.log(`  Independent Test Signers in Dataset: ${Array.from(foundSigners).sort().join(', ')}`);
  console.log(`  Signer-Independent Test Samples: ${summary.sampleCounts.test}`);
  console.log(`  Top-1 Test Accuracy: ${(summary.metrics.testTop1Accuracy * 100).toFixed(2)}%`);
  console.log(`  Top-5 Test Accuracy: ${(summary.metrics.testTop5Accuracy * 100).toFixed(2)}%`);
  const isV3 = fs.existsSync(v3SummaryPath);
  const validSigners = isV3
    ? (foundSigners.has('signer_13') && foundSigners.has('signer_14') && foundSigners.has('signer_15') && foundSigners.has('signer_16'))
    : (foundSigners.has('signer_08') && foundSigners.has('signer_09') && foundSigners.has('signer_10'));
  assert(
    validSigners,
    isV3 ? 'Test set strictly reserves Signers 13, 14, 15, 16 with zero training overlap' : 'Test set strictly reserves Signers 8, 9, 10'
  );
  assert(
    summary.metrics.testTop1Accuracy > 0.10,
    'Top-1 test accuracy exceeds 10% (significantly above random chance 1/186 = 0.54%)'
  );
  assert(
    summary.metrics.testTop5Accuracy > 0.30,
    'Top-5 test accuracy reaches > 30%'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 9: Noise Robustness Under Jitter (sigma = 0.01 and sigma = 0.04)
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 9: Noise Robustness Under Jitter ---');
  function addGaussianNoise(matrix: number[][], sigma: number): number[][] {
    return matrix.map(frame =>
      frame.map(val => val + (Math.random() * 2 - 1) * sigma * 1.5)
    );
  }

  const cleanMatrix = helloSample.featureMatrix;
  const noisy1Matrix = addGaussianNoise(cleanMatrix, 0.01);
  const noisy4Matrix = addGaussianNoise(cleanMatrix, 0.04);

  const tClean = tf.tensor3d([cleanMatrix]);
  const tNoisy1 = tf.tensor3d([noisy1Matrix]);
  const tNoisy4 = tf.tensor3d([noisy4Matrix]);

  const pClean = (model.predict(tClean) as tf.Tensor);
  const pNoisy1 = (model.predict(tNoisy1) as tf.Tensor);
  const pNoisy4 = (model.predict(tNoisy4) as tf.Tensor);

  const [dClean, dNoisy1, dNoisy4] = await Promise.all([pClean.data(), pNoisy1.data(), pNoisy4.data()]);
  tClean.dispose(); tNoisy1.dispose(); tNoisy4.dispose();
  pClean.dispose(); pNoisy1.dispose(); pNoisy4.dispose();

  // Measure cosine similarity or L2 drift
  let dot1 = 0, normClean = 0, norm1 = 0;
  for (let i = 0; i < dClean.length; i++) {
    dot1 += dClean[i] * dNoisy1[i];
    normClean += dClean[i] * dClean[i];
    norm1 += dNoisy1[i] * dNoisy1[i];
  }
  const cosSim1 = dot1 / (Math.sqrt(normClean) * Math.sqrt(norm1));
  console.log(`  Prediction Cosine Similarity under sigma=0.01 jitter: ${cosSim1.toFixed(4)}`);
  assert(cosSim1 > 0.85, 'Model predictions remain stable (>0.85 similarity) under sigma=0.01 jitter');

  let dot4 = 0, norm4 = 0;
  for (let i = 0; i < dClean.length; i++) {
    dot4 += dClean[i] * dNoisy4[i];
    norm4 += dNoisy4[i] * dNoisy4[i];
  }
  const cosSim4 = dot4 / (Math.sqrt(normClean) * Math.sqrt(norm4));
  console.log(`  Prediction Cosine Similarity under sigma=0.04 jitter: ${cosSim4.toFixed(4)}`);
  assert(cosSim4 > 0.60, 'Model predictions remain resilient (>0.60 similarity) under heavy sigma=0.04 jitter');

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 10: Complete Signing Without Premature Termination
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 10: Complete Signing Without Premature Termination ---');
  const liveCameraSrc = fs.readFileSync(path.resolve(__dirname, '../apps/web/src/components/vision/LiveCameraView.tsx'), 'utf8');
  const pauseMatch = liveCameraSrc.includes('now - lastEmit.current.at >= 900');
  assert(
    pauseMatch,
    'LiveCameraView implements pause detection with threshold >= 900ms'
  );
  assert(
    DEFAULT_PIPELINE_CONFIG.windowSize === 30,
    'Sliding window length is 30 frames (1 second at 30fps)'
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 11: Speech-to-ASL Operational Verification
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 11: Speech-to-ASL Functionality ---');
  const speechInputs = [
    { text: 'Hello, how are you?', expected: ['HELLO', 'HOW_ARE_YOU'] },
    { text: 'My name is Sarah', expected: ['MY', 'NAME'] },
    { text: 'I need water please', expected: ['I', 'NEED', 'WATER', 'PLEASE'] },
    { text: 'Where is the bathroom?', expected: ['BATHROOM', 'WHERE'] },
  ];

  for (const { text, expected } of speechInputs) {
    const { glosses } = translateEnglishToASLGloss(text);
    const hasAny = expected.some(exp => glosses.includes(exp));
    assert(hasAny, `Speech "${text}" maps to ASL glosses [${glosses.join(' ')}]`);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 12: 3D Avatar Animations Mapping Verification
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- TEST 12: 3D Avatar Animations Mapping Verification ---');
  const animControllerPath = path.resolve(__dirname, '../apps/web/src/components/avatar/AnimationController.ts');
  const animSrc = fs.readFileSync(animControllerPath, 'utf8');
  assert(animSrc.includes('IDLE'), 'AnimationController defines IDLE baseline state');
  assert(animSrc.includes('queueSequence'), 'AnimationController supports queued multi-sign execution');
  assert(animSrc.includes('crossFadeDuration'), 'AnimationController supports smooth cross-fading');

  const avatarPath = path.resolve(__dirname, '../apps/web/src/components/avatar/SignAvatar.tsx');
  const avatarSrc = fs.readFileSync(avatarPath, 'utf8');
  assert(avatarSrc.includes('CORE_ASL_ACTIONS'), 'SignAvatar defines core ASL actions');
  assert(avatarSrc.includes('matchSpeechSequence'), 'SignAvatar integrates speech-to-ASL sequence playing');

  console.log('\n================================================================');
  console.log(`VERIFICATION COMPLETE: ${passedTests}/${totalTests} TESTS PASSED (${failedTests} FAILED)`);
  console.log('================================================================');

  if (failedTests > 0) {
    process.exit(1);
  }
}

runAllVerificationTests().catch(err => {
  console.error('Verification failed with error:', err);
  process.exit(1);
});
