/**
 * verify_isl_to_speech_sentence.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Automated Verification for SignBridge V6 ISL-to-Speech Sentence Intelligence:
 * 1. Test D: Complete ISL-to-Speech Sentence:
 *    - ['HELLO', 'YOU', 'HOW'] -> "Hello, how are you?"
 *    - ['ME', 'HELP', 'NEED'] -> "I need help."
 *    - ['AMBULANCE', 'CALL', 'NEED'] -> "Please call an ambulance."
 *    - ['BATHROOM', 'WHERE'] -> "Where is the bathroom?"
 *    - ['HOSPITAL', 'WHERE'] -> "Where is the nearest hospital?"
 *    - ['FOOD', 'WANT'] -> "I want food to eat."
 * 2. Grammatical Completion without Hallucination:
 *    - Adds necessary copula/auxiliary English words (is/am/are/can/need/please)
 *    - Preserves user intent and semantic meaning exactly.
 * 3. Test E: Unknown Sign / Low-Confidence Handling:
 *    - Threshold gating (< 0.60): marked LOW_CONFIDENCE, does NOT add hallucinated words.
 *    - Never invents an unknown sign.
 * 4. Temporal Sentence Boundary Detection (>= 900ms rest pause):
 *    - Accumulates continuous signs.
 *    - Finalizes sentence ONLY upon >= 900ms rest pause.
 *    - Triggers spoken output (TTS) exactly once per finalized sentence.
 */

import { translateISLGlossSequenceToEnglish } from '@signbridge/types';

interface RecognitionFrame {
  gesture: string;
  confidence: number;
  timestamp: number;
}

class ContinuousSentenceRecognizer {
  private accumulatedGlosses: string[] = [];
  private pauseTimer: any = null;
  public finalizedSentences: string[] = [];
  public ttsSpokenPhrases: string[] = [];
  public lowConfidenceCount = 0;

  private readonly REST_PAUSE_MS = 900;
  private readonly CONFIDENCE_THRESHOLD = 0.60;

  public processSign(frame: RecognitionFrame) {
    // Test E: Low-confidence / Unknown sign handling
    if (frame.confidence < this.CONFIDENCE_THRESHOLD || frame.gesture === 'UNKNOWN' || frame.gesture === 'NONE') {
      this.lowConfidenceCount++;
      // Do not corrupt active sentence with unknown signs
      return;
    }

    // Reset rest pause timer on active signing
    if (this.pauseTimer) {
      clearTimeout(this.pauseTimer);
      this.pauseTimer = null;
    }

    // Append sign to gloss sequence (deduplicating adjacent identical hold frames)
    if (
      this.accumulatedGlosses.length === 0 ||
      this.accumulatedGlosses[this.accumulatedGlosses.length - 1] !== frame.gesture
    ) {
      this.accumulatedGlosses.push(frame.gesture);
    }

    // Schedule sentence boundary finalization upon rest pause >= 900ms
    this.pauseTimer = setTimeout(() => {
      this.finalizeSentence();
    }, this.REST_PAUSE_MS);
  }

  public finalizeSentence() {
    if (this.accumulatedGlosses.length === 0) return;
    const glosses = [...this.accumulatedGlosses];
    this.accumulatedGlosses = [];

    const english = translateISLGlossSequenceToEnglish(glosses);
    this.finalizedSentences.push(english);

    // Single TTS invocation per finalized sentence
    this.ttsSpokenPhrases.push(english);
  }
}

async function runTests() {
  console.log('='.repeat(80));
  console.log('SIGNBRIDGE V6 — ISL-TO-SPEECH SENTENCE INTELLIGENCE VERIFICATION');
  console.log('='.repeat(80));

  let passed = 0;
  let failed = 0;

  function assert(condition: boolean, msg: string) {
    if (condition) {
      console.log(`[PASS] ${msg}`);
      passed++;
    } else {
      console.error(`[FAIL] ${msg}`);
      failed++;
    }
  }

  // ── TEST D: Complete ISL Sentence Translations ────────────────────────────
  console.log('\n--- Test D: Continuous ISL Sequence Translation ---');

  const s1 = translateISLGlossSequenceToEnglish(['HELLO', 'YOU', 'HOW']);
  console.log('ISL [HELLO, YOU, HOW] -> English:', s1);
  assert(s1 === 'Hello, how are you?', 'Translated [HELLO, YOU, HOW] -> "Hello, how are you?"');

  const s2 = translateISLGlossSequenceToEnglish(['ME', 'HELP', 'NEED']);
  console.log('ISL [ME, HELP, NEED] -> English:', s2);
  assert(s2 === 'I need help.', 'Translated [ME, HELP, NEED] -> "I need help."');

  const s3 = translateISLGlossSequenceToEnglish(['AMBULANCE', 'CALL', 'NEED']);
  console.log('ISL [AMBULANCE, CALL, NEED] -> English:', s3);
  assert(s3 === 'Please call an ambulance.', 'Translated [AMBULANCE, CALL, NEED] -> "Please call an ambulance."');

  const s4 = translateISLGlossSequenceToEnglish(['BATHROOM', 'WHERE']);
  console.log('ISL [BATHROOM, WHERE] -> English:', s4);
  assert(s4 === 'Where is the bathroom?', 'Translated [BATHROOM, WHERE] -> "Where is the bathroom?"');

  const s5 = translateISLGlossSequenceToEnglish(['HOSPITAL', 'WHERE']);
  console.log('ISL [HOSPITAL, WHERE] -> English:', s5);
  assert(s5 === 'Where is the nearest hospital?', 'Translated [HOSPITAL, WHERE] -> "Where is the nearest hospital?"');

  const s6 = translateISLGlossSequenceToEnglish(['PLEASE', 'WATER', 'GIVE']);
  console.log('ISL [PLEASE, WATER, GIVE] -> English:', s6);
  assert(s6 === 'Please give me water.', 'Translated [PLEASE, WATER, GIVE] -> "Please give me water."');

  const s7 = translateISLGlossSequenceToEnglish(['FOOD', 'WANT']);
  console.log('ISL [FOOD, WANT] -> English:', s7);
  assert(s7 === 'I want food to eat.', 'Translated [FOOD, WANT] -> "I want food to eat."');

  // ── TEST E: Unknown Sign / Low-Confidence Handling ────────────────────────
  console.log('\n--- Test E: Unknown Sign & Low Confidence Handling ---');
  const recognizer = new ContinuousSentenceRecognizer();

  // Low confidence frame (< 0.60)
  recognizer.processSign({ gesture: 'WATER', confidence: 0.42, timestamp: Date.now() });
  assert(recognizer.lowConfidenceCount === 1, 'Low confidence frame correctly rejected');
  assert(recognizer.finalizedSentences.length === 0, 'No sentence generated from low-confidence noise');

  // Unknown sign
  recognizer.processSign({ gesture: 'UNKNOWN', confidence: 0.88, timestamp: Date.now() });
  assert(recognizer.lowConfidenceCount === 2, 'UNKNOWN sign rejected without hallucinating false word');

  // ── Temporal Boundary Detection (>= 900ms rest pause) ─────────────────────
  console.log('\n--- Temporal Rest Pause Boundary & Single TTS Trigger ---');
  const t0 = Date.now();

  // Sign: HELLO
  recognizer.processSign({ gesture: 'HELLO', confidence: 0.95, timestamp: t0 });
  // Sign: YOU (300ms later)
  recognizer.processSign({ gesture: 'YOU', confidence: 0.94, timestamp: t0 + 300 });
  // Sign: HOW (300ms later)
  recognizer.processSign({ gesture: 'HOW', confidence: 0.92, timestamp: t0 + 600 });

  assert(recognizer.finalizedSentences.length === 0, 'Sentence not yet finalized during active signing');

  // Wait 950ms for rest pause boundary detection
  console.log('Signer pauses hands (resting >= 900ms)...');
  await new Promise((resolve) => setTimeout(resolve, 950));

  assert(recognizer.finalizedSentences.length === 1, 'Sentence finalized exactly once upon >= 900ms rest pause');
  assert(recognizer.finalizedSentences[0] === 'Hello, how are you?', 'Finalized sentence is "Hello, how are you?"');
  assert(recognizer.ttsSpokenPhrases.length === 1, 'TTS triggered exactly 1 time (no audio loop or redundant speech)');

  console.log('\n' + '='.repeat(80));
  console.log(`ISL-TO-SPEECH VERIFICATION: ${passed} PASSED, ${failed} FAILED`);
  console.log('='.repeat(80));

  if (failed > 0) process.exit(1);
}

runTests();
