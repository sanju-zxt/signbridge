const fs = require('fs');
const path = require('path');

function save(relPath, content) {
  const full = path.resolve(__dirname, '..', relPath);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content, 'utf8');
  console.log('Saved:', relPath);
}

// 1. apps/web/src/lib/vision/normalizer.ts
save('apps/web/src/lib/vision/normalizer.ts', `import { Landmark3D } from '@signbridge/types';

export function normalizeHandLandmarks(landmarks: Landmark3D[]): Landmark3D[] {
  if (!landmarks || landmarks.length !== 21) return [];
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale = Math.sqrt(
    Math.pow(middleMcp.x - wrist.x, 2) +
    Math.pow(middleMcp.y - wrist.y, 2) +
    Math.pow(middleMcp.z - wrist.z, 2)
  ) || 1.0;

  return landmarks.map(lm => ({
    x: (lm.x - wrist.x) / scale,
    y: (lm.y - wrist.y) / scale,
    z: (lm.z - wrist.z) / scale,
    visibility: lm.visibility ?? 1.0
  }));
}

export function euclideanDist(a: Landmark3D, b: Landmark3D): number {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) +
    Math.pow(a.y - b.y, 2) +
    Math.pow(a.z - b.z, 2)
  );
}

export function computeFingerExtensions(normLm: Landmark3D[]): [number, number, number, number, number] {
  if (normLm.length < 21) return [0, 0, 0, 0, 0];

  const thumbTipDist = euclideanDist(normLm[4], normLm[2]);
  const indexTipDist = euclideanDist(normLm[8], normLm[5]);
  const middleTipDist = euclideanDist(normLm[12], normLm[9]);
  const ringTipDist = euclideanDist(normLm[16], normLm[13]);
  const pinkyTipDist = euclideanDist(normLm[20], normLm[17]);

  const ext = (tipToMcp: number, baseThreshold: number = 0.8) =>
    Math.min(1, Math.max(0, (tipToMcp - 0.3) / baseThreshold));

  return [
    ext(thumbTipDist, 0.7),
    ext(indexTipDist, 1.0),
    ext(middleTipDist, 1.0),
    ext(ringTipDist, 0.9),
    ext(pinkyTipDist, 0.8)
  ];
}
`);

// 2. apps/web/src/lib/vision/temporalMatcher.ts
save('apps/web/src/lib/vision/temporalMatcher.ts', `import { NormalizedFrame, AlternativeCandidate } from '@signbridge/types';
import { computeFingerExtensions } from './normalizer';

export interface TemporalPattern {
  signId: string;
  name: string;
  fingerStates: [number, number, number, number, number];
  palmFacing: 'camera' | 'user' | 'up' | 'down' | 'left' | 'right';
  motionVector: [number, number, number];
  relativeLocation: 'head' | 'chin' | 'chest' | 'neutral_space';
  isTwoHanded: boolean;
}

export const SIGN_FEATURE_TEMPLATES: Record<string, TemporalPattern> = {
  HELLO: { signId: 'HELLO', name: 'Hello / Namaste', fingerStates: [0.9, 1.0, 1.0, 1.0, 1.0], palmFacing: 'camera', motionVector: [0.3, -0.2, 0.1], relativeLocation: 'head', isTwoHanded: false },
  THANK_YOU: { signId: 'THANK_YOU', name: 'Thank You', fingerStates: [0.8, 1.0, 1.0, 1.0, 1.0], palmFacing: 'user', motionVector: [0.0, 0.2, 0.6], relativeLocation: 'chin', isTwoHanded: false },
  PLEASE: { signId: 'PLEASE', name: 'Please', fingerStates: [0.7, 1.0, 1.0, 1.0, 1.0], palmFacing: 'user', motionVector: [0.2, 0.2, 0.0], relativeLocation: 'chest', isTwoHanded: false },
  GOODBYE: { signId: 'GOODBYE', name: 'Goodbye', fingerStates: [0.8, 1.0, 1.0, 1.0, 1.0], palmFacing: 'camera', motionVector: [0.5, 0.0, 0.0], relativeLocation: 'neutral_space', isTwoHanded: false },
  HELP: { signId: 'HELP', name: 'Help / I Need Help', fingerStates: [0.9, 0.1, 0.1, 0.1, 0.1], palmFacing: 'camera', motionVector: [0.0, -0.4, 0.0], relativeLocation: 'chest', isTwoHanded: true },
  WATER: { signId: 'WATER', name: 'Water / Paani', fingerStates: [0.1, 1.0, 1.0, 1.0, 0.1], palmFacing: 'camera', motionVector: [0.0, 0.1, -0.1], relativeLocation: 'chin', isTwoHanded: false },
  FOOD: { signId: 'FOOD', name: 'Food / Khana', fingerStates: [0.4, 0.4, 0.4, 0.4, 0.4], palmFacing: 'user', motionVector: [0.0, 0.1, -0.2], relativeLocation: 'chin', isTwoHanded: false },
  HOSPITAL: { signId: 'HOSPITAL', name: 'Hospital / Doctor', fingerStates: [0.1, 1.0, 1.0, 0.1, 0.1], palmFacing: 'camera', motionVector: [0.1, 0.2, 0.0], relativeLocation: 'chest', isTwoHanded: true },
  BATHROOM: { signId: 'BATHROOM', name: 'Bathroom / Washroom', fingerStates: [0.6, 0.2, 0.1, 0.1, 0.1], palmFacing: 'camera', motionVector: [0.3, 0.0, 0.0], relativeLocation: 'neutral_space', isTwoHanded: false },
  WHERE: { signId: 'WHERE', name: 'Where / Kahan', fingerStates: [0.8, 1.0, 1.0, 1.0, 1.0], palmFacing: 'up', motionVector: [0.2, 0.0, 0.1], relativeLocation: 'neutral_space', isTwoHanded: true },
  HERE: { signId: 'HERE', name: 'Here / Yahan', fingerStates: [0.2, 1.0, 0.1, 0.1, 0.1], palmFacing: 'down', motionVector: [0.0, 0.2, 0.0], relativeLocation: 'neutral_space', isTwoHanded: true },
  LEFT: { signId: 'LEFT', name: 'Left / Baayein', fingerStates: [0.2, 1.0, 0.1, 0.1, 0.1], palmFacing: 'camera', motionVector: [-0.4, 0.0, 0.0], relativeLocation: 'neutral_space', isTwoHanded: false },
  RIGHT: { signId: 'RIGHT', name: 'Right / Daayein', fingerStates: [0.2, 1.0, 0.1, 0.1, 0.1], palmFacing: 'camera', motionVector: [0.4, 0.0, 0.0], relativeLocation: 'neutral_space', isTwoHanded: false },
  YES: { signId: 'YES', name: 'Yes / Haan', fingerStates: [0.6, 0.1, 0.1, 0.1, 0.1], palmFacing: 'camera', motionVector: [0.0, 0.3, 0.0], relativeLocation: 'neutral_space', isTwoHanded: false },
  NO: { signId: 'NO', name: 'No / Nahi', fingerStates: [0.8, 0.9, 0.9, 0.1, 0.1], palmFacing: 'camera', motionVector: [0.1, 0.2, 0.0], relativeLocation: 'neutral_space', isTwoHanded: false },
  DONT_UNDERSTAND: { signId: 'DONT_UNDERSTAND', name: "Don't Understand", fingerStates: [0.2, 1.0, 0.1, 0.1, 0.1], palmFacing: 'user', motionVector: [0.3, -0.2, 0.2], relativeLocation: 'head', isTwoHanded: false },
  REPEAT: { signId: 'REPEAT', name: 'Repeat / Phir Se', fingerStates: [0.5, 0.8, 0.8, 0.8, 0.8], palmFacing: 'camera', motionVector: [0.0, 0.2, 0.3], relativeLocation: 'neutral_space', isTwoHanded: true }
};

export function matchTemporalSequence(frames: NormalizedFrame[]) {
  const startTime = performance.now();
  if (!frames || frames.length < 5) {
    return { sign: 'IDLE', label: 'Position hands in view', confidence: 0, alternatives: [], latencyMs: 0 };
  }

  const latestFrame = frames[frames.length - 1];
  const firstFrame = frames[0];
  const activeHand = latestFrame.rightHand || latestFrame.leftHand;
  if (!activeHand) {
    return { sign: 'IDLE', label: 'No hands detected', confidence: 0, alternatives: [], latencyMs: 0 };
  }

  const fingerExt = computeFingerExtensions(activeHand);
  const isTwoHanded = Boolean(latestFrame.leftHand && latestFrame.rightHand);

  const firstWrist = (firstFrame.rightHand || firstFrame.leftHand)?.[0] || { x: 0, y: 0, z: 0 };
  const lastWrist = (latestFrame.rightHand || latestFrame.leftHand)?.[0] || { x: 0, y: 0, z: 0 };
  const motion = [
    lastWrist.x - firstWrist.x,
    lastWrist.y - firstWrist.y,
    lastWrist.z - firstWrist.z
  ];

  const scores: Array<{ signId: string; name: string; score: number }> = [];

  for (const [signId, template] of Object.entries(SIGN_FEATURE_TEMPLATES)) {
    let fingerScore = 0;
    for (let i = 0; i < 5; i++) {
      const diff = Math.abs(fingerExt[i] - template.fingerStates[i]);
      fingerScore += 1 - diff;
    }
    fingerScore /= 5;

    const handMatch = (template.isTwoHanded === isTwoHanded) ? 1.0 : 0.65;

    const tm = template.motionVector;
    const motionMag = Math.sqrt(motion[0] * motion[0] + motion[1] * motion[1] + motion[2] * motion[2]) || 0.001;
    const tmMag = Math.sqrt(tm[0] * tm[0] + tm[1] * tm[1] + tm[2] * tm[2]) || 0.001;
    const dot = motion[0] * tm[0] + motion[1] * tm[1] + motion[2] * tm[2];
    const motionCos = Math.max(0, dot / (motionMag * tmMag));
    const motionScore = 0.5 + 0.5 * motionCos;

    const totalScore = (fingerScore * 0.55) + (handMatch * 0.20) + (motionScore * 0.25);
    scores.push({
      signId,
      name: template.name,
      score: Math.min(0.98, Math.max(0.05, totalScore))
    });
  }

  scores.sort((a, b) => b.score - a.score);
  const topMatch = scores[0];
  const confidence = Number(topMatch.score.toFixed(3));
  const alternatives: AlternativeCandidate[] = scores.slice(1, 4).map(s => ({
    sign: s.signId,
    confidence: Number(s.score.toFixed(3)),
    label: s.name
  }));

  const latencyMs = Number((performance.now() - startTime).toFixed(1));

  return {
    sign: topMatch.signId,
    label: topMatch.name,
    confidence,
    alternatives,
    latencyMs
  };
}
`);

// 3. apps/web/src/lib/speech/stt.ts
save('apps/web/src/lib/speech/stt.ts', `import { SupportedLanguage } from '@signbridge/types';

export interface STTOptions {
  language: SupportedLanguage;
  onPartial: (text: string) => void;
  onFinal: (text: string, confidence: number) => void;
  onError: (err: string) => void;
  onStateChange: (listening: boolean) => void;
}

export class StreamingSTTClient {
  private recognition: any = null;
  private isListening: boolean = false;
  private options: STTOptions;

  constructor(options: STTOptions) {
    this.options = options;
  }

  public start() {
    if (typeof window === 'undefined') return;

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      this.options.onError('Web Speech API is not supported in this browser. You can use manual speech buttons or text input.');
      return;
    }

    try {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.maxAlternatives = 1;

      // Set language locale
      const langMap: Record<SupportedLanguage, string> = {
        en: 'en-IN',
        hi: 'hi-IN',
        kn: 'kn-IN',
        te: 'te-IN'
      };
      this.recognition.lang = langMap[this.options.language] || 'en-IN';

      this.recognition.onstart = () => {
        this.isListening = true;
        this.options.onStateChange(true);
      };

      this.recognition.onresult = (event: any) => {
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const item = event.results[i];
          if (item.isFinal) {
            const transcript = item[0].transcript.trim();
            const confidence = item[0].confidence || 0.94;
            if (transcript.length > 0) {
              this.options.onFinal(transcript, confidence);
            }
          } else {
            interimTranscript += item[0].transcript;
          }
        }
        if (interimTranscript.trim()) {
          this.options.onPartial(interimTranscript.trim());
        }
      };

      this.recognition.onerror = (event: any) => {
        if (event.error !== 'no-speech') {
          this.options.onError(\`Speech recognition: \${event.error}\`);
        }
      };

      this.recognition.onend = () => {
        if (this.isListening) {
          // Auto-restart for continuous communication room
          try {
            this.recognition.start();
          } catch {
            this.isListening = false;
            this.options.onStateChange(false);
          }
        } else {
          this.options.onStateChange(false);
        }
      };

      this.recognition.start();
    } catch (err: any) {
      this.options.onError(err.message || 'Could not start microphone');
    }
  }

  public stop() {
    this.isListening = false;
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch {}
    }
    this.options.onStateChange(false);
  }

  public setLanguage(lang: SupportedLanguage) {
    this.options.language = lang;
    if (this.isListening && this.recognition) {
      this.stop();
      this.start();
    }
  }
}
`);

// 4. apps/web/src/lib/speech/tts.ts
save('apps/web/src/lib/speech/tts.ts', `import { SupportedLanguage } from '@signbridge/types';

export class TTSPlayer {
  private synth: SpeechSynthesis | null = null;
  private isSpeakingState: boolean = false;

  constructor() {
    if (typeof window !== 'undefined') {
      this.synth = window.speechSynthesis;
    }
  }

  public speak(text: string, language: SupportedLanguage = 'en', onEnd?: () => void) {
    if (!this.synth) return;

    // Interrupt any ongoing utterance to keep perceived latency low
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    const langMap: Record<SupportedLanguage, string> = {
      en: 'en-IN',
      hi: 'hi-IN',
      kn: 'kn-IN',
      te: 'te-IN'
    };
    utterance.lang = langMap[language] || 'en-IN';
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      this.isSpeakingState = true;
    };

    utterance.onend = () => {
      this.isSpeakingState = false;
      if (onEnd) onEnd();
    };

    utterance.onerror = () => {
      this.isSpeakingState = false;
      if (onEnd) onEnd();
    };

    this.synth.speak(utterance);
  }

  public stop() {
    if (this.synth) {
      this.synth.cancel();
      this.isSpeakingState = false;
    }
  }

  public isSpeaking(): boolean {
    return this.isSpeakingState;
  }
}

export const ttsPlayer = new TTSPlayer();
`);
