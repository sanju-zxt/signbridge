/**
 * datasetValidator.test.ts
 * Automated unit tests verifying sequence schema validation, session-leakage prevention,
 * and multi-class evaluation metric calculations.
 */

import {
  validateSequence,
  partitionBySession,
  calculateEvaluationMetrics,
  PROTOTYPE_VOCABULARY,
} from './datasetValidator';

console.log('=== Running ISL Dataset Validator & Split Integrity Tests ===\n');

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  }
  console.log(`✅ PASS: ${msg}`);
}

// ---------------------------------------------------------------------------
// 1. Schema Validation Tests
// ---------------------------------------------------------------------------

// Test A: Null / Invalid object
const resNull = validateSequence(null);
assert(!resNull.valid && resNull.errors.length > 0, 'Rejects null input');

// Test B: Missing sign
const resNoSign = validateSequence({ frames: [] });
assert(!resNoSign.valid && resNoSign.errors.some(e => e.includes('sign')), 'Rejects sequence without sign');

// Test C: Valid 30-frame sequence
const dummyFrame = {
  timestamp: 1000,
  rightHand: new Array(21).fill(null).map((_, i) => ({ x: 0.1 * i, y: 0.2 * i, z: 0 })),
  leftHand: null,
};
const validSeq = {
  sign: 'HELLO',
  sessionId: 'test_session_1',
  sampleIndex: 1,
  durationMs: 1000,
  frames: new Array(30).fill(dummyFrame),
};
const resValid = validateSequence(validSeq);
assert(resValid.valid && resValid.errors.length === 0, 'Accepts valid 30-frame sequence');
assert(resValid.frameCount === 30, 'Correctly reports 30 frames');
assert(resValid.handsPresentFraction === 1.0, 'Reports 100% hand presence fraction');

// Test D: Non-finite landmark coordinates
const corruptSeq = {
  sign: 'WATER',
  sessionId: 'test_session_1',
  frames: [
    {
      timestamp: 1000,
      rightHand: [{ x: NaN, y: 0, z: 0 }, ...new Array(20).fill({ x: 0, y: 0, z: 0 })],
      leftHand: null,
    },
  ],
};
const resCorrupt = validateSequence(corruptSeq);
assert(!resCorrupt.valid && resCorrupt.errors.some(e => e.includes('non-finite')), 'Rejects NaN landmarks');

// ---------------------------------------------------------------------------
// 2. Session Split & Leakage Prevention Tests
// ---------------------------------------------------------------------------

interface MockSample {
  id: string;
  sign: string;
  sessionId: string;
}

const mockDataset: MockSample[] = [
  { id: '1', sign: 'HELLO', sessionId: 'session_A' },
  { id: '2', sign: 'HELLO', sessionId: 'session_A' },
  { id: '3', sign: 'YES', sessionId: 'session_A' },
  { id: '4', sign: 'HELLO', sessionId: 'session_B' },
  { id: '5', sign: 'YES', sessionId: 'session_B' },
  { id: '6', sign: 'WATER', sessionId: 'session_C' },
  { id: '7', sign: 'WATER', sessionId: 'session_C' },
  { id: '8', sign: 'HELP', sessionId: 'session_D' },
];

const split = partitionBySession(mockDataset, { trainRatio: 0.75 });
const trainSessions = new Set(split.sessions.trainSessions);
const testSessions = new Set(split.sessions.testSessions);

// Verify zero intersection between train and test sessions
let sessionOverlap = false;
for (const s of split.train) {
  if (testSessions.has(s.sessionId)) sessionOverlap = true;
}
for (const s of split.test) {
  if (trainSessions.has(s.sessionId)) sessionOverlap = true;
}

assert(!sessionOverlap, 'Zero data leakage between session-partitioned splits');
assert(split.train.length > 0 && split.test.length > 0, 'Partitions both train and test samples');

// ---------------------------------------------------------------------------
// 3. Evaluation Metrics & Confusion Matrix Tests
// ---------------------------------------------------------------------------

const labels = ['HELLO', 'YES', 'WATER', 'NONE_RESTING'];
const actual    = ['HELLO', 'HELLO', 'YES', 'YES', 'WATER', 'NONE_RESTING'];
const predicted = ['HELLO', 'YES',   'YES', 'YES', 'WATER', 'NONE_RESTING'];

const metrics = calculateEvaluationMetrics(labels, actual, predicted);

assert(metrics.accuracy === 5 / 6 || Math.abs(metrics.accuracy - 0.833) < 0.01, 'Calculates correct accuracy (5/6 = 83.3%)');
assert(metrics.confusionMatrix['HELLO']['HELLO'] === 1, 'Confusion matrix records true positive for HELLO');
assert(metrics.confusionMatrix['HELLO']['YES'] === 1, 'Confusion matrix records misclassification HELLO -> YES');
assert(metrics.confusionMatrix['YES']['YES'] === 2, 'Confusion matrix records 2 true positives for YES');

const yesMetric = metrics.classMetrics.find(m => m.sign === 'YES');
assert(yesMetric !== undefined && yesMetric.recall === 1.0, 'Calculates 100% recall for YES (all actual YES predicted as YES)');

console.log('\n🌟 ALL DATASET VALIDATOR & METRICS TESTS PASSED! 🌟\n');
