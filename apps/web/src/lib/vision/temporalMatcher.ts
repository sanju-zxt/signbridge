import { NormalizedFrame, AlternativeCandidate } from '@signbridge/types';
import { computeFingerExtensions } from './normalizer';

export interface TemporalPattern {
  signId: string;
  name: string;
  fingerStates: [number, number, number, number, number];
  palmFacing: 'camera' | 'user' | 'up' | 'down' | 'left' | 'right';
  motionVector: [number, number, number];
  location: 'head' | 'chin' | 'chest' | 'neutral_space';
  isTwoHanded: boolean;
}

export const SIGN_FEATURE_TEMPLATES: Record<string, TemporalPattern> = {
  HELLO: { signId: 'HELLO', name: 'Hello / Namaste', fingerStates: [0.9, 1.0, 1.0, 1.0, 1.0], palmFacing: 'camera', motionVector: [0.4, -0.3, 0.1], location: 'head', isTwoHanded: false },
  THANK_YOU: { signId: 'THANK_YOU', name: 'Thank You', fingerStates: [0.85, 1.0, 1.0, 1.0, 1.0], palmFacing: 'user', motionVector: [0.0, 0.3, 0.7], location: 'chin', isTwoHanded: false },
  PLEASE: { signId: 'PLEASE', name: 'Please', fingerStates: [0.75, 1.0, 1.0, 1.0, 1.0], palmFacing: 'user', motionVector: [0.3, 0.3, 0.0], location: 'chest', isTwoHanded: false },
  GOODBYE: { signId: 'GOODBYE', name: 'Goodbye', fingerStates: [0.8, 1.0, 1.0, 1.0, 1.0], palmFacing: 'camera', motionVector: [0.8, 0.0, 0.0], location: 'neutral_space', isTwoHanded: false },
  HELP: { signId: 'HELP', name: 'Help / I Need Help', fingerStates: [0.9, 0.05, 0.05, 0.05, 0.05], palmFacing: 'camera', motionVector: [0.0, -0.5, 0.0], location: 'chest', isTwoHanded: true },
  WATER: { signId: 'WATER', name: 'Water / Paani', fingerStates: [0.05, 1.0, 1.0, 1.0, 0.05], palmFacing: 'camera', motionVector: [0.0, 0.15, -0.2], location: 'chin', isTwoHanded: false },
  FOOD: { signId: 'FOOD', name: 'Food / Khana', fingerStates: [0.35, 0.35, 0.35, 0.35, 0.35], palmFacing: 'user', motionVector: [0.0, 0.1, -0.4], location: 'chin', isTwoHanded: false },
  HOSPITAL: { signId: 'HOSPITAL', name: 'Hospital / Doctor', fingerStates: [0.05, 1.0, 1.0, 0.05, 0.05], palmFacing: 'camera', motionVector: [0.2, 0.3, 0.0], location: 'chest', isTwoHanded: true },
  BATHROOM: { signId: 'BATHROOM', name: 'Bathroom / Washroom', fingerStates: [0.65, 0.1, 0.05, 0.05, 0.05], palmFacing: 'camera', motionVector: [0.4, 0.0, 0.0], location: 'neutral_space', isTwoHanded: false },
  WHERE: { signId: 'WHERE', name: 'Where / Kahan', fingerStates: [0.95, 1.0, 1.0, 1.0, 1.0], palmFacing: 'up', motionVector: [0.4, 0.0, 0.0], location: 'neutral_space', isTwoHanded: true },
  HERE: { signId: 'HERE', name: 'Here / Yahan', fingerStates: [0.1, 1.0, 0.05, 0.05, 0.05], palmFacing: 'down', motionVector: [0.0, 0.4, 0.0], location: 'neutral_space', isTwoHanded: true },
  LEFT: { signId: 'LEFT', name: 'Left / Baayein', fingerStates: [0.1, 1.0, 0.05, 0.05, 0.05], palmFacing: 'camera', motionVector: [-0.6, 0.0, 0.0], location: 'neutral_space', isTwoHanded: false },
  RIGHT: { signId: 'RIGHT', name: 'Right / Daayein', fingerStates: [0.1, 1.0, 0.05, 0.05, 0.05], palmFacing: 'camera', motionVector: [0.6, 0.0, 0.0], location: 'neutral_space', isTwoHanded: false },
  YES: { signId: 'YES', name: 'Yes / Haan', fingerStates: [0.6, 0.0, 0.0, 0.0, 0.0], palmFacing: 'camera', motionVector: [0.0, 0.6, 0.0], location: 'neutral_space', isTwoHanded: false },
  NO: { signId: 'NO', name: 'No / Nahi', fingerStates: [0.75, 0.9, 0.9, 0.0, 0.0], palmFacing: 'camera', motionVector: [0.3, 0.0, 0.0], location: 'neutral_space', isTwoHanded: false },
  DONT_UNDERSTAND: { signId: 'DONT_UNDERSTAND', name: "Don't Understand", fingerStates: [0.1, 1.0, 0.05, 0.05, 0.05], palmFacing: 'user', motionVector: [0.4, -0.2, 0.3], location: 'head', isTwoHanded: false },
  REPEAT: { signId: 'REPEAT', name: 'Repeat / Phir Se', fingerStates: [0.55, 0.85, 0.85, 0.85, 0.85], palmFacing: 'camera', motionVector: [0.0, 0.3, 0.5], location: 'neutral_space', isTwoHanded: true }
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
      fingerScore += Math.max(0, 1 - diff * 1.6);
    }
    fingerScore /= 5;

    const handMatch = (template.isTwoHanded === isTwoHanded) ? 1.0 : 0.3;

    const tm = template.motionVector;
    const motionMag = Math.sqrt(motion[0] * motion[0] + motion[1] * motion[1] + motion[2] * motion[2]) || 0.001;
    const tmMag = Math.sqrt(tm[0] * tm[0] + tm[1] * tm[1] + tm[2] * tm[2]) || 0.001;
    const dot = motion[0] * tm[0] + motion[1] * tm[1] + motion[2] * tm[2];
    const motionCos = Math.max(-1, Math.min(1, dot / (motionMag * tmMag)));
    const motionScore = Math.max(0, 0.5 + 0.5 * motionCos);

    const magRatio = Math.min(motionMag, tmMag) / Math.max(motionMag, tmMag);

    const totalScore = (fingerScore * 0.45) + (handMatch * 0.25) + (motionScore * 0.20) + (magRatio * 0.10);
    scores.push({
      signId,
      name: template.name,
      score: Math.min(0.99, Math.max(0.05, totalScore))
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
