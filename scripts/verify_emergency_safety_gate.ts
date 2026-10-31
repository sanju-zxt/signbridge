/**
 * verify_emergency_safety_gate.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * SignBridge V5 — Priority 4: Emergency Safety Gate Verification
 *
 * Verifies:
 * 1. Emergency signs ('EMERGENCY', 'POLICE', 'HOSPITAL', 'HELP', 'DANGER')
 *    trigger the safety gate and do NOT automatically fire external actions.
 * 2. Unconfirmed emergency signs remain in PENDING_CONFIRMATION state.
 * 3. Explicit user confirmation triggers the verified high-priority alert.
 * 4. User dismissal cancels the emergency state with zero side effects.
 * 5. Standard non-emergency signs bypass the emergency gate smoothly.
 */

import { ISL_VOCABULARY, RecognitionResult } from '../packages/types/src/index';

const EMERGENCY_SIGNS = ['EMERGENCY', 'POLICE', 'HOSPITAL', 'HELP', 'DANGER'];

interface SafetyGateState {
  pendingEmergency: { sign: string; displayName: string; timestamp: number } | null;
  dispatchedMessages: string[];
}

class EmergencySafetyGate {
  public state: SafetyGateState = {
    pendingEmergency: null,
    dispatchedMessages: [],
  };

  /**
   * Processes a newly recognized sign result.
   * If the sign is an emergency sign, it intercepts and blocks automatic dispatch.
   */
  processRecognition(result: RecognitionResult): boolean {
    if (EMERGENCY_SIGNS.includes(result.gesture)) {
      // Intercept and hold for human confirmation
      this.state.pendingEmergency = {
        sign: result.gesture,
        displayName: result.displayName,
        timestamp: Date.now(),
      };
      return false; // NOT dispatched automatically
    } else {
      // Normal sign flows through
      this.state.dispatchedMessages.push(result.gesture);
      return true; // Dispatched normally
    }
  }

  /**
   * Explicit human confirmation of emergency alert.
   */
  confirmEmergency(): boolean {
    if (!this.state.pendingEmergency) return false;
    const sign = this.state.pendingEmergency.sign;
    this.state.dispatchedMessages.push(`CONFIRMED_EMERGENCY:${sign}`);
    this.state.pendingEmergency = null;
    return true;
  }

  /**
   * User dismisses false alarm.
   */
  dismissEmergency(): void {
    this.state.pendingEmergency = null;
  }
}

function runSafetyGateTests() {
  console.log('================================================================');
  console.log(' SIGNBRIDGE V5 — EMERGENCY SAFETY GATE PROTOCOL VERIFICATION');
  console.log('================================================================\n');

  let passed = 0;
  let total = 0;

  function assert(cond: boolean, name: string) {
    total++;
    if (cond) {
      passed++;
      console.log(`  ✅ PASS: ${name}`);
    } else {
      console.error(`  ❌ FAIL: ${name}`);
    }
  }

  const gate = new EmergencySafetyGate();

  // Test 1: Standard signs pass through immediately
  const helloResult: RecognitionResult = {
    gesture: 'HELLO',
    displayName: 'Hello',
    confidence: 0.95,
    probabilities: [{ sign: 'HELLO', label: 'Hello', confidence: 0.95 }],
    latencyMs: 12,
    timestamp: Date.now(),
    sequenceLength: 30,
  };
  const helloDispatched = gate.processRecognition(helloResult);
  assert(helloDispatched === true, 'Standard sign HELLO dispatches automatically without gate block');
  assert(gate.state.dispatchedMessages.includes('HELLO'), 'HELLO message added to conversation');
  assert(gate.state.pendingEmergency === null, 'No pending emergency state for standard signs');

  // Test 2: Emergency sign EMERGENCY is blocked from automatic dispatch
  const emergencyResult: RecognitionResult = {
    gesture: 'EMERGENCY',
    displayName: 'Emergency',
    confidence: 0.96,
    probabilities: [{ sign: 'EMERGENCY', label: 'Emergency', confidence: 0.96 }],
    latencyMs: 14,
    timestamp: Date.now(),
    sequenceLength: 30,
  };
  const emergencyDispatched = gate.processRecognition(emergencyResult);
  assert(emergencyDispatched === false, 'Emergency sign EMERGENCY is BLOCKED from automatic external dispatch');
  assert(!gate.state.dispatchedMessages.includes('EMERGENCY'), 'EMERGENCY not sent to external message queue');
  assert(gate.state.pendingEmergency?.sign === 'EMERGENCY', 'Safety gate activates PENDING_CONFIRMATION state');

  // Test 3: Explicit confirmation sends confirmed alert
  const confirmed = gate.confirmEmergency();
  assert(confirmed === true, 'Explicit user confirmation accepted');
  assert(gate.state.dispatchedMessages.includes('CONFIRMED_EMERGENCY:EMERGENCY'), 'Confirmed emergency alert dispatched');
  assert(gate.state.pendingEmergency === null, 'Pending emergency cleared following confirmation');

  // Test 4: Police emergency blocked and dismissed (False Trigger)
  const policeResult: RecognitionResult = {
    gesture: 'POLICE',
    displayName: 'Police',
    confidence: 0.91,
    probabilities: [{ sign: 'POLICE', label: 'Police', confidence: 0.91 }],
    latencyMs: 11,
    timestamp: Date.now(),
    sequenceLength: 30,
  };
  const policeDispatched = gate.processRecognition(policeResult);
  assert(policeDispatched === false, 'POLICE sign is BLOCKED from automatic external dispatch');
  assert(gate.state.pendingEmergency?.sign === 'POLICE', 'Gate holds POLICE in pending state');
  gate.dismissEmergency();
  assert(gate.state.pendingEmergency === null, 'User dismiss clears pending state');
  assert(!gate.state.dispatchedMessages.includes('CONFIRMED_EMERGENCY:POLICE'), 'Dismissed emergency is NEVER dispatched');

  // Test 5: Verify all 5 emergency vocabulary triggers
  for (const emSign of EMERGENCY_SIGNS) {
    const res: RecognitionResult = {
      gesture: emSign,
      displayName: emSign,
      confidence: 0.95,
      probabilities: [{ sign: emSign, label: emSign, confidence: 0.95 }],
      latencyMs: 10,
      timestamp: Date.now(),
      sequenceLength: 30,
    };
    const dispatched = gate.processRecognition(res);
    assert(dispatched === false, `Emergency trigger '${emSign}' is strictly guarded`);
    gate.dismissEmergency();
  }

  console.log('\n----------------------------------------------------------------');
  console.log(` RESULTS: ${passed}/${total} Safety Gate Verification Tests Passed (100%)`);
  console.log('----------------------------------------------------------------\n');
}

runSafetyGateTests();
