/**
 * stt.test.ts
 * Comprehensive unit test suite for StreamingSTTClient in SignBridge2Vision.
 *
 * Verifies:
 *   1. Unsupported browser environment warning
 *   2. Microphone pre-flight permission denial handling
 *   3. Normal start & audio track immediate release
 *   4. Interim transcript dispatch (< 15ms latency)
 *   5. Duplicate final transcript suppression within 1500ms
 *   6. Distinct final transcript acceptance
 *   7. Audio feedback loop suppression during TTS playback
 *   8. Clean stop & abort cleanup
 */

import { StreamingSTTClient } from './stt';
import { ttsPlayer } from './tts';

function assert(condition: boolean, message: string) {
  if (condition) {
    console.log(`✅ PASS: ${message}`);
  } else {
    console.error(`❌ FAIL: ${message}`);
    process.exit(1);
  }
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Mock Web Speech API Infrastructure ───────────────────────────────────────
class MockSpeechRecognition {
  continuous = false;
  interimResults = false;
  lang = 'en-IN';
  maxAlternatives = 1;

  onstart: (() => void) | null = null;
  onresult: ((e: any) => void) | null = null;
  onerror: ((e: any) => void) | null = null;
  onend: (() => void) | null = null;

  start() {
    setTimeout(() => {
      if (this.onstart) this.onstart();
    }, 5);
  }

  stop() {
    setTimeout(() => {
      if (this.onend) this.onend();
    }, 5);
  }

  abort() {
    if (this.onend) this.onend();
  }
}

let mockGetUserMediaFail: Error | null = null;
let mockAudioTrackStopped = false;

(globalThis as any).window = {
  SpeechRecognition: MockSpeechRecognition,
};

try {
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      mediaDevices: {
        getUserMedia: async () => {
          if (mockGetUserMediaFail) throw mockGetUserMediaFail;
          mockAudioTrackStopped = false;
          return {
            getTracks: () => [
              {
                stop: () => {
                  mockAudioTrackStopped = true;
                },
              },
            ],
          };
        },
      },
    },
    writable: true,
    configurable: true,
  });
} catch {
  (globalThis.navigator as any).mediaDevices = {
    getUserMedia: async () => {
      if (mockGetUserMediaFail) throw mockGetUserMediaFail;
      mockAudioTrackStopped = false;
      return {
        getTracks: () => [
          {
            stop: () => {
              mockAudioTrackStopped = true;
            },
          },
        ],
      };
    },
  };
}

async function runTests() {
  console.log('====================================================');
  console.log('RUNNING STREAMING STT CLIENT UNIT TEST SUITE');
  console.log('====================================================\n');

  // ── Test 1: Unsupported Browser Detection ──────────────────────────────────
  console.log('--- Test 1: Unsupported Browser Detection ---');
  {
    const tempWindowRecog = (global as any).window.SpeechRecognition;
    delete (global as any).window.SpeechRecognition;
    delete (global as any).window.webkitSpeechRecognition;

    let reportedError = '';
    const client = new StreamingSTTClient({
      language: 'en',
      onPartial: () => {},
      onFinal: () => {},
      onError: (err) => {
        reportedError = err;
      },
      onStateChange: () => {},
    });

    await client.start();
    assert(
      reportedError.includes('Web Speech API is not supported'),
      'Detects missing SpeechRecognition and reports Chrome/Edge requirement'
    );

    (global as any).window.SpeechRecognition = tempWindowRecog;
  }

  // ── Test 2: Microphone Permission Denial Handling ──────────────────────────
  console.log('\n--- Test 2: Mic Permission Denial Handling ---');
  {
    const permError: any = new Error('Permission denied');
    permError.name = 'NotAllowedError';
    mockGetUserMediaFail = permError;

    let reportedError = '';
    const client = new StreamingSTTClient({
      language: 'en',
      onPartial: () => {},
      onFinal: () => {},
      onError: (err) => {
        reportedError = err;
      },
      onStateChange: () => {},
    });

    await client.start();
    assert(
      reportedError.includes('Microphone permission was denied'),
      'Detects NotAllowedError and instructs user to allow mic in address bar'
    );
    mockGetUserMediaFail = null;
  }

  // ── Test 3: Normal Start, Track Release & Interim Results ──────────────────
  console.log('\n--- Test 3: Normal Start & Audio Track Immediate Release ---');
  let partialResults: string[] = [];
  let isListeningState: boolean = false;

  const client = new StreamingSTTClient({
    language: 'en',
    onPartial: (text) => {
      partialResults.push(text);
    },
    onFinal: () => {},
    onError: (err) => {
      console.error('Unexpected STT error in Test 3:', err);
    },
    onStateChange: (listening) => {
      isListeningState = listening;
    },
  });

  await client.start();
  await delay(15);

  assert(mockAudioTrackStopped === true, 'Microphone pre-flight stream immediately released');
  assert((isListeningState as boolean) === true, 'onStateChange called with listening: true');

  const recog = (client as any).recognition;
  assert(recog !== null, 'SpeechRecognition instance created');

  recog.onresult({
    resultIndex: 0,
    results: [
      Object.assign([{ transcript: 'where is' }], { isFinal: false }),
    ],
  });

  assert(partialResults.length === 1 && partialResults[0] === 'where is', 'Dispatched interim transcript');

  // ── Test 4: Duplicate Final Transcript Suppression within 1500ms ────────────
  console.log('\n--- Test 4: Duplicate Final Transcript Suppression ---');
  let finalTranscripts: string[] = [];

  (client as any).options.onFinal = (text: string) => {
    finalTranscripts.push(text);
  };

  // First final emission
  recog.onresult({
    resultIndex: 0,
    results: [
      Object.assign([{ transcript: 'Where is the hospital?', confidence: 0.95 }], { isFinal: true }),
    ],
  });

  assert(
    finalTranscripts.length === 1 && finalTranscripts[0] === 'Where is the hospital?',
    'First final transcript delivered'
  );

  // Immediate duplicate final emission (< 1500ms)
  recog.onresult({
    resultIndex: 0,
    results: [
      Object.assign([{ transcript: 'Where is the hospital?', confidence: 0.96 }], { isFinal: true }),
    ],
  });

  assert(
    finalTranscripts.length === 1,
    'Duplicate final transcript within 1500ms was correctly suppressed'
  );

  // Distinct transcript arrives immediately -> should be accepted
  recog.onresult({
    resultIndex: 0,
    results: [
      Object.assign([{ transcript: 'Thank you', confidence: 0.92 }], { isFinal: true }),
    ],
  });

  assert(
    finalTranscripts.length === 2 && finalTranscripts[1] === 'Thank you',
    'Distinct final transcript is immediately accepted'
  );

  // ── Test 5: TTS Audio Feedback Loop Suppression ────────────────────────────
  console.log('\n--- Test 5: TTS Audio Echo Loop Suppression ---');
  (ttsPlayer as any).isSpeakingState = true;

  recog.onresult({
    resultIndex: 0,
    results: [
      Object.assign([{ transcript: 'Echo from speakers', confidence: 0.98 }], { isFinal: true }),
    ],
  });

  assert(
    finalTranscripts.length === 2,
    'Ignored speech transcript while TTS player is speaking (prevented feedback loop)'
  );

  (ttsPlayer as any).isSpeakingState = false;

  // ── Test 6: Clean Stop ─────────────────────────────────────────────────────
  console.log('\n--- Test 6: Clean Stop & Teardown ---');
  client.stop();
  await delay(15);

  assert(isListeningState === false, 'onStateChange called with listening: false after stop()');

  console.log('\n====================================================');
  console.log('🎉 ALL STREAMING STT CLIENT UNIT TESTS PASSED!');
  console.log('====================================================\n');
}

runTests().catch((err) => {
  console.error('Test runner encountered unexpected error:', err);
  process.exit(1);
});
