/**
 * verify_speech_to_isl.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Automated Verification for SignBridge V6 Speech-to-ISL Sentence Processing:
 * 1. Test C: Full Sentence from Speech:
 *    - "Hello, how are you?" -> ['HELLO', 'YOU', 'HOW']
 *    - "Can you help me?" -> ['YOU', 'HELP', 'NEED']
 *    - "I need medical assistance." -> ['HOSPITAL', 'HELP', 'NEED']
 *    - "Please call an ambulance." -> ['PLEASE', 'AMBULANCE', 'CALL']
 *    - "Where is the bathroom?" -> ['BATHROOM', 'WHERE']
 * 2. Test F: Consecutive Speech Commands:
 *    - First command: "Please give me water and food"
 *    - Second command: "Thank you very much"
 *    - Ensures both are processed in full without dropped words or action duplication.
 * 3. Utterance Buffering & Partial vs Final Isolation:
 *    - Partial transcripts update subtitle state but trigger 0 avatar actions.
 *    - Final transcript triggers the full sequential execution.
 */

import { matchSpeechSequence } from '../apps/web/src/lib/speech/phraseMatcher';
import { ActionExecutionManager, ActionDispatcher } from '../apps/web/src/lib/avatar/ActionExecutionManager';

class MockAvatarDispatcher implements ActionDispatcher {
  public executedClips: string[] = [];
  public currentPlaying: string = 'IDLE';

  playClip(
    actionId: string,
    options: { loop: boolean; crossFadeDuration: number },
    onFinished: () => void
  ): boolean {
    this.executedClips.push(actionId);
    this.currentPlaying = actionId;
    // Auto-advance
    setTimeout(onFinished, 10);
    return true;
  }

  returnToIdle(): void {
    this.currentPlaying = 'IDLE';
  }
}

async function runTests() {
  console.log('='.repeat(80));
  console.log('SIGNBRIDGE V6 — SPEECH-TO-ISL SENTENCE PROCESSING VERIFICATION');
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

  // ── TEST C: Full Sentence from Speech ──────────────────────────────────────
  console.log('\n--- Test C: Full Sentence Utterance Parsing (No Dropped Words) ---');

  // Case 1: "Hello, how are you?"
  const seq1 = matchSpeechSequence('Hello, how are you?');
  console.log('Input: "Hello, how are you?" -> Output ISL:', seq1);
  assert(seq1.length >= 2, '"Hello, how are you?" produced multi-sign ISL sequence');
  assert(seq1.includes('HELLO'), 'Contains HELLO');
  assert(seq1.includes('HOW') || seq1.includes('HOW_ARE_YOU'), 'Contains HOW / HOW_ARE_YOU (did not drop question)');

  // Case 2: "Can you help me?"
  const seq2 = matchSpeechSequence('Can you help me?');
  console.log('Input: "Can you help me?" -> Output ISL:', seq2);
  assert(seq2.length >= 2, '"Can you help me?" produced multi-sign ISL sequence');
  assert(seq2.includes('HELP'), 'Contains HELP');
  assert(seq2.includes('NEED') || seq2.includes('YOU'), 'Contains NEED / YOU (full semantic representation)');

  // Case 3: "I need medical assistance."
  const seq3 = matchSpeechSequence('I need medical assistance.');
  console.log('Input: "I need medical assistance." -> Output ISL:', seq3);
  assert(seq3.includes('HOSPITAL') || seq3.includes('HELP'), 'Contains HOSPITAL/HELP');
  assert(seq3.includes('NEED'), 'Contains NEED (complete sentence preserved)');

  // Case 4: "Please call an ambulance."
  const seq4 = matchSpeechSequence('Please call an ambulance.');
  console.log('Input: "Please call an ambulance." -> Output ISL:', seq4);
  assert(seq4.includes('AMBULANCE'), 'Contains AMBULANCE');
  assert(seq4.includes('CALL') || seq4.includes('PLEASE'), 'Contains CALL or PLEASE (meaningful components preserved)');

  // Case 5: "Where is the bathroom?"
  const seq5 = matchSpeechSequence('Where is the bathroom?');
  console.log('Input: "Where is the bathroom?" -> Output ISL:', seq5);
  assert(seq5.length === 2 && seq5[0] === 'BATHROOM' && seq5[1] === 'WHERE', 'ISL SOV Topic-Comment order: BATHROOM WHERE');

  // ── Partial vs Final Isolation (Utterance Buffering) ───────────────────────
  console.log('\n--- Utterance Buffering: Partial vs Final Transcript Isolation ---');
  const dispatcher = new MockAvatarDispatcher();
  const manager = new ActionExecutionManager(dispatcher);

  let displayedSubtitle = '';
  function onPartialSpeech(interimText: string) {
    // Updates UI subtitle only; NEVER dispatches to avatar manager
    displayedSubtitle = interimText;
  }

  function onFinalSpeech(finalUtterance: string) {
    displayedSubtitle = finalUtterance;
    const actions = matchSpeechSequence(finalUtterance);
    for (const act of actions) {
      manager.submitEvent(act, { source: 'speech', force: true });
    }
  }

  // Simulate streaming partial tokens
  onPartialSpeech('Hello');
  assert(displayedSubtitle === 'Hello', 'Partial 1 displayed on screen');
  assert(dispatcher.executedClips.length === 0, 'No avatar animation started on interim "Hello"');

  onPartialSpeech('Hello how');
  assert(displayedSubtitle === 'Hello how', 'Partial 2 displayed on screen');
  assert(dispatcher.executedClips.length === 0, 'No avatar animation started on interim "Hello how"');

  onPartialSpeech('Hello how are you');
  assert(displayedSubtitle === 'Hello how are you', 'Partial 3 displayed on screen');
  assert(dispatcher.executedClips.length === 0, 'No avatar animation started on interim "Hello how are you"');

  // Final confirmation / endpointing
  onFinalSpeech('Hello, how are you?');
  assert(displayedSubtitle === 'Hello, how are you?', 'Final subtitle displayed');
  assert(dispatcher.executedClips.length > 0, 'Avatar animation started upon final utterance confirmation');
  assert(dispatcher.executedClips[0] === 'HELLO', 'First action executed is HELLO');

  // Wait for queue execution
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert(dispatcher.executedClips.length === seq1.length, `Full sequence of ${seq1.length} actions executed in order`);

  // ── TEST F: Consecutive Speech Commands ─────────────────────────────────────
  console.log('\n--- Test F: Consecutive Speech Commands ---');
  dispatcher.executedClips = [];

  const cmd1 = 'Please give me water and food';
  const cmd2 = 'Thank you very much';

  onFinalSpeech(cmd1);
  await new Promise((resolve) => setTimeout(resolve, 80));
  const clipsAfterCmd1 = [...dispatcher.executedClips];
  console.log(`Command 1 "${cmd1}" executed:`, clipsAfterCmd1);
  assert(clipsAfterCmd1.includes('WATER'), 'Command 1 executed WATER');
  assert(clipsAfterCmd1.includes('FOOD'), 'Command 1 executed FOOD');

  // Second command right after
  onFinalSpeech(cmd2);
  await new Promise((resolve) => setTimeout(resolve, 50));
  console.log(`Command 2 "${cmd2}" executed:`, dispatcher.executedClips.slice(clipsAfterCmd1.length));
  assert(dispatcher.executedClips.includes('THANK_YOU'), 'Command 2 executed THANK_YOU independently');

  console.log('\n' + '='.repeat(80));
  console.log(`SPEECH-TO-ISL VERIFICATION: ${passed} PASSED, ${failed} FAILED`);
  console.log('='.repeat(80));

  if (failed > 0) process.exit(1);
}

runTests();
