/**
 * verify_action_loop_exhaustive.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Exhaustive Real-World Validation of Avatar Action Loop State Machine.
 *
 * Verifies all Section 3 Requirements:
 * 1. One sign held for several seconds (static hold suppression -> exactly 1 execution, 0 resets).
 * 2. The same sign performed twice intentionally (>=900ms rest pause -> 2 executions).
 * 3. Two different signs performed consecutively (HELLO -> THANK_YOU -> sequential execution).
 * 4. Long animation followed by another animation (sequential queue completion).
 * 5. Animation interruption / emergency queue cancellation (resetQueue -> IDLE).
 * 6. Recognition failure / unsupported clip (graceful FAILED transition -> IDLE).
 */

import { ActionExecutionManager, ActionDispatcher } from '../apps/web/src/lib/avatar/ActionExecutionManager';
import { ActionEvent } from '@signbridge/types';

class MockAvatarDispatcher implements ActionDispatcher {
  public currentClip: string = 'IDLE';
  public isIdle: boolean = true;
  public playCount: Record<string, number> = {};
  public clipDurations: Record<string, number> = {
    HELLO: 800,
    THANK_YOU: 900,
    HOSPITAL: 3000,
    PLEASE: 700,
  };
  private finishTimer: any = null;

  playClip(
    actionId: string,
    options: { loop: boolean; crossFadeDuration: number },
    onFinished: () => void
  ): boolean {
    if (actionId === 'NON_EXISTENT_CLIP') {
      return false;
    }

    this.currentClip = actionId;
    this.isIdle = false;
    this.playCount[actionId] = (this.playCount[actionId] || 0) + 1;

    const duration = this.clipDurations[actionId] || 800;
    this.finishTimer = setTimeout(() => {
      onFinished();
    }, duration);

    return true;
  }

  returnToIdle(crossFadeDuration: number = 0.25): void {
    if (this.finishTimer) {
      clearTimeout(this.finishTimer);
      this.finishTimer = null;
    }
    this.currentClip = 'IDLE';
    this.isIdle = true;
  }
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runExhaustiveActionLoopTests() {
  console.log('='.repeat(80));
  console.log('EXHAUSTIVE AVATAR ACTION LOOP & LIFECYCLE VALIDATION');
  console.log('='.repeat(80));

  let totalTests = 0;
  let passedTests = 0;

  function assert(condition: boolean, msg: string) {
    totalTests++;
    if (condition) {
      passedTests++;
      console.log(`  [PASS] Test ${totalTests}: ${msg}`);
    } else {
      console.error(`  [FAIL] Test ${totalTests}: ${msg}`);
      throw new Error(`Assertion failed: ${msg}`);
    }
  }

  // -------------------------------------------------------------------------
  // SCENARIO 1: ONE SIGN HELD FOR SEVERAL SECONDS (STATIC HOLD SUPPRESSION)
  // -------------------------------------------------------------------------
  console.log('\n--- Scenario 1: One Sign Held for Several Seconds ---');
  const dispatcher1 = new MockAvatarDispatcher();
  const manager1 = new ActionExecutionManager(dispatcher1);

  // Initial trigger
  const firstEvt = manager1.submitEvent('HELLO', { source: 'vision' });
  assert(firstEvt !== null, 'Initial HELLO submission must be accepted');
  assert(firstEvt?.state === 'EXECUTING', 'Initial HELLO transitions to EXECUTING immediately');

  // Simulate 30 fps stream of HELLO frames for 3.0 seconds (90 frames)
  let rejectedCount = 0;
  for (let frame = 0; frame < 90; frame++) {
    const res = manager1.submitEvent('HELLO', { source: 'vision' });
    if (res === null) rejectedCount++;
  }
  assert(rejectedCount === 90, `All 90 redundant static hold frames were suppressed (rejected: ${rejectedCount})`);
  assert(dispatcher1.playCount['HELLO'] === 1, `Dispatcher play count for HELLO is exactly 1 (got: ${dispatcher1.playCount['HELLO']})`);

  // Wait for animation to finish
  await sleep(900);
  assert(manager1.getState() === 'IDLE', `Manager returns to IDLE after clip completion (state: ${manager1.getState()})`);
  assert(dispatcher1.isIdle === true, 'Dispatcher is in IDLE mode');

  // Immediately submit HELLO without cooldown (<900ms after completion)
  const immediateRetrigger = manager1.submitEvent('HELLO', { source: 'vision' });
  assert(immediateRetrigger === null, 'Immediate retrigger without rest pause (<900ms) is suppressed');
  assert(dispatcher1.playCount['HELLO'] === 1, 'Dispatcher play count remains exactly 1');

  // -------------------------------------------------------------------------
  // SCENARIO 2: SAME SIGN PERFORMED TWICE INTENTIONALLY (>=900ms REST PAUSE)
  // -------------------------------------------------------------------------
  console.log('\n--- Scenario 2: Same Sign Performed Twice Intentionally ---');
  // Wait past repetition cooldown (1000ms > 900ms)
  await sleep(1000);
  const secondEvt = manager1.submitEvent('HELLO', { source: 'vision' });
  assert(secondEvt !== null, 'Intentional second HELLO after rest pause is accepted');
  assert(secondEvt?.state === 'EXECUTING', 'Second HELLO transitions to EXECUTING');
  assert(firstEvt?.eventId !== secondEvt?.eventId, 'Second HELLO has a distinct unique eventId');
  assert(dispatcher1.playCount['HELLO'] === 2, `Dispatcher play count incremented to exactly 2 (got: ${dispatcher1.playCount['HELLO']})`);

  await sleep(900);
  assert(manager1.getState() === 'IDLE', 'Manager returns to IDLE after second HELLO');

  // -------------------------------------------------------------------------
  // SCENARIO 3: TWO DIFFERENT SIGNS PERFORMED CONSECUTIVELY
  // -------------------------------------------------------------------------
  console.log('\n--- Scenario 3: Two Different Signs Performed Consecutively ---');
  const dispatcher3 = new MockAvatarDispatcher();
  const manager3 = new ActionExecutionManager(dispatcher3);

  const evtA = manager3.submitEvent('HELLO', { source: 'vision' });
  const evtB = manager3.submitEvent('THANK_YOU', { source: 'vision' });

  assert(evtA !== null && evtA.state === 'EXECUTING', 'HELLO starts EXECUTING immediately');
  assert(evtB !== null && evtB.state === 'QUEUED', 'THANK_YOU is successfully QUEUED behind HELLO');
  assert(manager3.getQueueLength() === 1, 'Queue length is exactly 1');

  // Wait for HELLO to complete (800ms)
  await sleep(850);
  assert(evtA?.state === 'COMPLETED', 'HELLO transitions to COMPLETED');
  assert(evtB?.state === 'EXECUTING', 'THANK_YOU transitions to EXECUTING immediately');
  assert(dispatcher3.currentClip === 'THANK_YOU', 'Dispatcher is now playing THANK_YOU');

  // Wait for THANK_YOU to complete (900ms)
  await sleep(950);
  assert(evtB?.state === 'COMPLETED', 'THANK_YOU transitions to COMPLETED');
  assert(manager3.getState() === 'IDLE', 'Manager returns to IDLE after sequence completion');

  // -------------------------------------------------------------------------
  // SCENARIO 4: LONG ANIMATION FOLLOWED BY ANOTHER ANIMATION
  // -------------------------------------------------------------------------
  console.log('\n--- Scenario 4: Long Animation Followed by Another Animation ---');
  const dispatcher4 = new MockAvatarDispatcher();
  const manager4 = new ActionExecutionManager(dispatcher4);

  const longEvt = manager4.submitEvent('HOSPITAL', { source: 'vision' }); // 3000ms duration
  assert(longEvt !== null && longEvt.state === 'EXECUTING', 'HOSPITAL (3000ms) starts EXECUTING');

  // At t=1000ms, queue PLEASE
  await sleep(1000);
  const nextEvt = manager4.submitEvent('PLEASE', { source: 'vision' });
  assert(nextEvt !== null && nextEvt.state === 'QUEUED', 'PLEASE is queued while HOSPITAL is in progress');
  assert(manager4.getState() === 'EXECUTING', 'Manager is still EXECUTING HOSPITAL');

  // Wait for HOSPITAL to finish (2100ms more)
  await sleep(2100);
  assert(longEvt?.state === 'COMPLETED', 'HOSPITAL transitions to COMPLETED');
  assert(nextEvt?.state === 'EXECUTING', 'PLEASE starts EXECUTING immediately after HOSPITAL');

  await sleep(800);
  assert(nextEvt?.state === 'COMPLETED', 'PLEASE transitions to COMPLETED');
  assert(manager4.getState() === 'IDLE', 'Manager returns to IDLE');

  // -------------------------------------------------------------------------
  // SCENARIO 5: ANIMATION INTERRUPTION / EMERGENCY RESET
  // -------------------------------------------------------------------------
  console.log('\n--- Scenario 5: Animation Interruption & Emergency Reset ---');
  const dispatcher5 = new MockAvatarDispatcher();
  const manager5 = new ActionExecutionManager(dispatcher5);

  const activeEvt5 = manager5.submitEvent('HOSPITAL');
  const queuedEvt5 = manager5.submitEvent('PLEASE');
  assert(activeEvt5?.state === 'EXECUTING', 'HOSPITAL is executing');
  assert(queuedEvt5?.state === 'QUEUED', 'PLEASE is queued');

  // Trigger emergency interruption
  manager5.resetQueue();
  assert(activeEvt5?.state === 'CANCELLED', 'Active event is marked CANCELLED');
  assert(queuedEvt5?.state === 'CANCELLED', 'Queued event is marked CANCELLED');
  assert(manager5.getState() === 'IDLE', 'Manager state returns immediately to IDLE');
  assert(dispatcher5.isIdle === true, 'Dispatcher returns to IDLE');
  assert(manager5.getQueueLength() === 0, 'Queue is completely cleared');

  // -------------------------------------------------------------------------
  // SCENARIO 6: RECOGNITION FAILURE / UNSUPPORTED CLIP
  // -------------------------------------------------------------------------
  console.log('\n--- Scenario 6: Recognition Failure & Unsupported Clip Recovery ---');
  const dispatcher6 = new MockAvatarDispatcher();
  const manager6 = new ActionExecutionManager(dispatcher6);

  // Submitting invalid / noise signs
  assert(manager6.submitEvent('UNKNOWN') === null, 'UNKNOWN gesture is rejected immediately');
  assert(manager6.submitEvent('NONE') === null, 'NONE gesture is rejected immediately');
  assert(manager6.submitEvent('IDLE') === null, 'IDLE submission is rejected');

  // Submitting an action unsupported by 3D dispatcher
  const failEvt = manager6.submitEvent('NON_EXISTENT_CLIP');
  assert(failEvt !== null, 'Event initially passes validation');
  assert(failEvt?.state === 'FAILED', 'Unsupported clip transitions immediately to FAILED');
  assert(manager6.getState() === 'IDLE', 'Manager gracefully returns to IDLE without hanging');
  assert(dispatcher6.isIdle === true, 'Dispatcher remains in IDLE');

  console.log('\n' + '='.repeat(80));
  console.log(`ALL EXHAUSTIVE ACTION LOOP TESTS PASSED: ${passedTests}/${totalTests}`);
  console.log('='.repeat(80));
}

runExhaustiveActionLoopTests().catch((err) => {
  console.error('Test failed with error:', err);
  process.exit(1);
});
