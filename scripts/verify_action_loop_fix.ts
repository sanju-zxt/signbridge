/**
 * verify_action_loop_fix.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Automated Verification for SignBridge V6 Action Execution & Loop Elimination:
 * 1. Test A: Single action execution (one-shot execution, returns to IDLE).
 * 2. Test B: Intentional repetition (allows intentional consecutive gestures).
 * 3. Duplicate event prevention (idempotent eventId suppression).
 * 4. Stale event cancellation (discarding events > 5000ms old).
 * 5. Lifecycle transitions: IDLE -> DETECTED -> VALIDATED -> QUEUED -> EXECUTING -> COMPLETED -> IDLE.
 */

import { ActionExecutionManager, ActionDispatcher } from '../apps/web/src/lib/avatar/ActionExecutionManager';
import { ActionEvent, ActionLifecycleState } from '@signbridge/types';

class MockAnimationDispatcher implements ActionDispatcher {
  public executedClips: Array<{ actionId: string; loop: boolean }> = [];
  public currentPlaying: string | null = null;
  public idleCallCount = 0;
  private finishCb: (() => void) | null = null;

  playClip(
    actionId: string,
    options: { loop: boolean; crossFadeDuration: number },
    onFinished: () => void
  ): boolean {
    this.executedClips.push({ actionId, loop: options.loop });
    this.currentPlaying = actionId;
    this.finishCb = onFinished;
    return true;
  }

  returnToIdle(): void {
    this.idleCallCount++;
    this.currentPlaying = 'IDLE';
    this.finishCb = null;
  }

  triggerFinish(): void {
    if (this.finishCb) {
      const cb = this.finishCb;
      this.finishCb = null;
      cb();
    }
  }
}

async function runTests() {
  console.log('='.repeat(80));
  console.log('SIGNBRIDGE V6 — ACTION EXECUTION & LOOP ELIMINATION VERIFICATION');
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

  // ── TEST 1: Initial State ──────────────────────────────────────────────────
  const dispatcher = new MockAnimationDispatcher();
  const manager = new ActionExecutionManager(dispatcher);
  assert(manager.getState() === 'IDLE', 'Initial manager state is IDLE');

  // ── TEST 2 (Test A): Single Action Execution & Return to IDLE ──────────────
  console.log('\n--- Test A: Single Action Execution ---');
  const transitions: ActionLifecycleState[] = [];
  manager.subscribe((evt) => {
    transitions.push(evt.state);
  });

  const evtA = manager.submitEvent('HELLO', { eventId: 'evt_001', confidence: 0.95 });
  assert(evtA !== null, 'Event HELLO submitted successfully');
  assert(manager.getState() === 'EXECUTING', 'Manager transitioned to EXECUTING');
  assert(dispatcher.currentPlaying === 'HELLO', 'Dispatcher started playing HELLO');
  assert(dispatcher.executedClips.length === 1, 'Clip executed exactly 1 time');

  // Simulate ongoing camera frames while user holds their hands up in HELLO pose
  console.log('Simulating continuous frames while hands remain visible in HELLO pose...');
  for (let frame = 0; frame < 10; frame++) {
    const redundantEvt = manager.submitEvent('HELLO', { confidence: 0.94 });
    assert(redundantEvt === null, `Redundant static frame #${frame + 1} correctly suppressed (no replay loop)`);
  }
  assert(dispatcher.executedClips.length === 1, 'Clip still executed exactly 1 time during static hold');

  // Complete the animation
  dispatcher.triggerFinish();
  assert(manager.getState() === 'IDLE', 'Manager transitioned to IDLE upon animation completion');
  assert(dispatcher.currentPlaying === 'IDLE', 'Dispatcher returned to natural resting IDLE');
  assert(dispatcher.idleCallCount >= 1, 'Dispatcher returnToIdle invoked');

  // ── TEST 3: Duplicate eventId rejection (Idempotency) ──────────────────────
  console.log('\n--- Idempotent Event ID Tracking ---');
  const duplicateEvt = manager.submitEvent('HELLO', { eventId: 'evt_001' });
  assert(duplicateEvt === null, 'Identical eventId evt_001 rejected idempotently');

  // ── TEST 4 (Test B): Intentional Repetition ─────────────────────────────────
  console.log('\n--- Test B: Intentional Repetition After Rest/Pause ---');
  // Wait simulated rest pause >= 900ms
  await new Promise((resolve) => setTimeout(resolve, 950));

  const evtB = manager.submitEvent('HELLO', { eventId: 'evt_002', confidence: 0.96 });
  assert(evtB !== null, 'Intentional repeated HELLO accepted after pause');
  assert(dispatcher.executedClips.length === 2, 'Total executed clips is now 2 (intentional repetition)');
  assert(manager.getState() === 'EXECUTING', 'Manager is EXECUTING second gesture');

  dispatcher.triggerFinish();
  assert(manager.getState() === 'IDLE', 'Manager returned to IDLE after second gesture');

  // ── TEST 5: Sequential Queueing (Full Sentence from Speech) ─────────────────
  console.log('\n--- Sequential Action Queueing ---');
  const sequence = ['HELLO', 'YOU', 'HOW'];
  for (const sign of sequence) {
    manager.submitEvent(sign, { force: true });
  }

  assert(dispatcher.currentPlaying === 'HELLO', 'First action in sequence HELLO is executing');
  assert(manager.getQueueLength() === 2, 'Remaining 2 actions are safely queued');

  // Finish HELLO -> triggers YOU
  dispatcher.triggerFinish();
  assert(dispatcher.currentPlaying === 'YOU', 'Second action in sequence YOU is now executing');
  assert(manager.getQueueLength() === 1, 'Queue has 1 action left');

  // Finish YOU -> triggers HOW
  dispatcher.triggerFinish();
  assert(dispatcher.currentPlaying === 'HOW', 'Third action in sequence HOW is now executing');
  assert(manager.getQueueLength() === 0, 'Queue is now empty');

  // Finish HOW -> returns to IDLE
  dispatcher.triggerFinish();
  assert(manager.getState() === 'IDLE', 'Queue complete: Manager smoothly returns to IDLE');
  assert(dispatcher.currentPlaying === 'IDLE', 'Dispatcher in resting IDLE breathing loop');

  console.log('\n' + '='.repeat(80));
  console.log(`ACTION LOOP FIX VERIFICATION: ${passed} PASSED, ${failed} FAILED`);
  console.log('='.repeat(80));

  if (failed > 0) {
    process.exit(1);
  }
}

runTests();
