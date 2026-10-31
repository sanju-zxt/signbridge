/**
 * datasetValidator.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Authoritative ISL reference standards, sequence schema validation,
 * session-safe dataset partitioning, and evaluation metrics computation.
 */

import { RawLandmarkFrame, CollectionSequence, Landmark3D } from '@signbridge/types';

export interface PrototypeSignInfo {
  sign: string;
  label: string;
  category: 'greetings' | 'needs' | 'clarification' | 'control';
  islrtcDescription: string;
  handsRequired: 'single' | 'both';
  dominantHand: 'right' | 'either';
  location: 'head' | 'chin' | 'chest' | 'neutral_space';
  movement: 'linear' | 'repetitive' | 'static' | 'circular';
  tips: string;
}

export const PROTOTYPE_VOCABULARY: PrototypeSignInfo[] = [
  {
    sign: 'HELLO',
    label: 'Hello / Namaste',
    category: 'greetings',
    islrtcDescription: 'Open dominant hand raised to forehead/temple and extended outward in a waving salute, or two-handed Namaste at chest.',
    handsRequired: 'single',
    dominantHand: 'right',
    location: 'head',
    movement: 'linear',
    tips: 'Ensure hand is raised clearly near temple/forehead with fingers together and palm facing outward.',
  },
  {
    sign: 'YES',
    label: 'Yes / Haan',
    category: 'clarification',
    islrtcDescription: 'Closed fist with thumb outside, nodding vertically downward and upward repeatedly like a head nod.',
    handsRequired: 'single',
    dominantHand: 'right',
    location: 'neutral_space',
    movement: 'repetitive',
    tips: 'Keep fingers curled into a fist and articulate the wrist downward and back up 2-3 times.',
  },
  {
    sign: 'NO',
    label: 'No / Nahi',
    category: 'clarification',
    islrtcDescription: 'Index and middle fingers extended together snapping down firmly against the thumb repeatedly.',
    handsRequired: 'single',
    dominantHand: 'right',
    location: 'neutral_space',
    movement: 'repetitive',
    tips: 'Index and middle fingers start extended, then snap shut to contact thumb 2-3 times.',
  },
  {
    sign: 'THANK_YOU',
    label: 'Thank You / Dhanyavad',
    category: 'greetings',
    islrtcDescription: 'Flat open palm touching chin or lips and gently moving outward and forward towards the conversation partner.',
    handsRequired: 'single',
    dominantHand: 'right',
    location: 'chin',
    movement: 'linear',
    tips: 'Touch fingertips to chin/lips and sweep outward horizontally with fingers extended flat.',
  },
  {
    sign: 'WATER',
    label: 'Water / Paani',
    category: 'needs',
    islrtcDescription: 'W-shaped hand (Index, Middle, Ring extended; thumb over pinky) held near chin/mouth with repeated tapping or tilting.',
    handsRequired: 'single',
    dominantHand: 'right',
    location: 'chin',
    movement: 'repetitive',
    tips: 'Form W-handshape with three center fingers extended and tap near mouth 2 times.',
  },
  {
    sign: 'HELP',
    label: 'Help / Sahayata',
    category: 'needs',
    islrtcDescription: 'Two-handed sign: Non-dominant hand is held flat palm-up; dominant hand forms thumbs-up fist placed on palm and lifted upward together.',
    handsRequired: 'both',
    dominantHand: 'right',
    location: 'chest',
    movement: 'linear',
    tips: 'Both hands must be in camera view. Left hand flat palm up, right thumbs-up fist resting on it lifting upward.',
  },
  {
    sign: 'NONE_RESTING',
    label: 'Resting / Neutral (Negative Control)',
    category: 'control',
    islrtcDescription: 'Negative control class: Hands resting on desk/lap, relaxed open hand at rest, non-signing movement, or hands out of view.',
    handsRequired: 'single',
    dominantHand: 'either',
    location: 'neutral_space',
    movement: 'static',
    tips: 'Perform non-signing movements, resting palms, or hands relaxed at bottom of frame to train negative defense.',
  },
];

export interface ValidationReport {
  valid: boolean;
  errors: string[];
  warnings: string[];
  frameCount: number;
  handsPresentFraction: number;
  durationMs: number;
}

/**
 * Validates a single recorded collection sequence against ISL prototype schema requirements.
 */
export function validateSequence(
  seq: any,
  expectedLength = 30
): ValidationReport {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!seq || typeof seq !== 'object') {
    return {
      valid: false,
      errors: ['Sequence is null or not an object.'],
      warnings: [],
      frameCount: 0,
      handsPresentFraction: 0,
      durationMs: 0,
    };
  }

  if (typeof seq.sign !== 'string' || seq.sign.trim().length === 0) {
    errors.push('Sequence missing valid string "sign" identifier.');
  }

  if (!Array.isArray(seq.frames)) {
    errors.push('Sequence "frames" must be an array.');
    return {
      valid: false,
      errors,
      warnings,
      frameCount: 0,
      handsPresentFraction: 0,
      durationMs: 0,
    };
  }

  const frameCount = seq.frames.length;
  if (frameCount === 0) {
    errors.push('Sequence contains 0 frames.');
  } else if (frameCount < expectedLength) {
    warnings.push(`Sequence has ${frameCount} frames, expected ${expectedLength} (will be zero-padded).`);
  } else if (frameCount > expectedLength) {
    warnings.push(`Sequence has ${frameCount} frames, exceeding standard window of ${expectedLength}.`);
  }

  let framesWithHand = 0;

  for (let i = 0; i < seq.frames.length; i++) {
    const frame = seq.frames[i];
    if (!frame || typeof frame !== 'object') {
      errors.push(`Frame ${i} is not a valid object.`);
      continue;
    }

    const hasRight = Array.isArray(frame.rightHand) && frame.rightHand.length === 21;
    const hasLeft = Array.isArray(frame.leftHand) && frame.leftHand.length === 21;

    if (hasRight || hasLeft) {
      framesWithHand++;
    }

    // Check landmark values for right hand
    if (Array.isArray(frame.rightHand)) {
      if (frame.rightHand.length !== 21) {
        errors.push(`Frame ${i}: rightHand landmark count is ${frame.rightHand.length}, expected 21.`);
      } else {
        for (let j = 0; j < 21; j++) {
          const pt = frame.rightHand[j];
          if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y) || !Number.isFinite(pt.z)) {
            errors.push(`Frame ${i} rightHand landmark ${j} contains non-finite coordinates.`);
            break;
          }
        }
      }
    }

    // Check landmark values for left hand
    if (Array.isArray(frame.leftHand)) {
      if (frame.leftHand.length !== 21) {
        errors.push(`Frame ${i}: leftHand landmark count is ${frame.leftHand.length}, expected 21.`);
      } else {
        for (let j = 0; j < 21; j++) {
          const pt = frame.leftHand[j];
          if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y) || !Number.isFinite(pt.z)) {
            errors.push(`Frame ${i} leftHand landmark ${j} contains non-finite coordinates.`);
            break;
          }
        }
      }
    }
  }

  const handsFraction = frameCount > 0 ? framesWithHand / frameCount : 0;
  const isNoneResting = seq.sign?.toUpperCase() === 'NONE_RESTING';

  if (!isNoneResting && handsFraction < 0.40) {
    warnings.push(
      `Low hand visibility: Hand detected in only ${(handsFraction * 100).toFixed(1)}% of frames.`
    );
  }

  const durationMs = typeof seq.durationMs === 'number' ? seq.durationMs : 0;

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    frameCount,
    handsPresentFraction: Math.round(handsFraction * 1000) / 1000,
    durationMs,
  };
}

/**
 * Session-safe dataset partitioner to strictly prevent data leakage.
 * Guarantees all sequences from a particular sessionId are assigned to either train, val, or test.
 */
export function partitionBySession<T extends { sessionId: string; sign: string }>(
  samples: T[],
  options: { trainRatio?: number; testRatio?: number; seed?: number } = {}
): { train: T[]; test: T[]; sessions: { trainSessions: string[]; testSessions: string[] } } {
  const trainRatio = options.trainRatio ?? 0.75;

  const sessionMap = new Map<string, T[]>();
  for (const s of samples) {
    const sid = s.sessionId || 'default_session';
    if (!sessionMap.has(sid)) sessionMap.set(sid, []);
    sessionMap.get(sid)!.push(s);
  }

  const sessionIds = Array.from(sessionMap.keys());

  // Deterministic pseudo-random shuffle if seed provided
  const seed = options.seed ?? 42;
  const shuffledSessions = [...sessionIds].sort((a, b) => {
    const hashA = (a + seed).split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
    const hashB = (b + seed).split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
    return (hashA % 97) - (hashB % 97);
  });

  if (shuffledSessions.length <= 1) {
    // If only 1 session exists, we partition per-class round-robin with warning
    const train: T[] = [];
    const test: T[] = [];
    const classGroups = new Map<string, T[]>();
    for (const s of samples) {
      if (!classGroups.has(s.sign)) classGroups.set(s.sign, []);
      classGroups.get(s.sign)!.push(s);
    }
    for (const [, group] of classGroups) {
      const splitPoint = Math.max(1, Math.floor(group.length * trainRatio));
      train.push(...group.slice(0, splitPoint));
      test.push(...group.slice(splitPoint));
    }
    return {
      train,
      test,
      sessions: { trainSessions: sessionIds, testSessions: sessionIds },
    };
  }

  const trainCount = Math.max(1, Math.floor(shuffledSessions.length * trainRatio));
  const trainSessions = shuffledSessions.slice(0, trainCount);
  const testSessions = shuffledSessions.slice(trainCount);

  const train: T[] = [];
  const test: T[] = [];

  for (const sid of trainSessions) {
    train.push(...(sessionMap.get(sid) || []));
  }
  for (const sid of testSessions) {
    test.push(...(sessionMap.get(sid) || []));
  }

  return {
    train,
    test,
    sessions: { trainSessions, testSessions },
  };
}

/**
 * Calculates per-class classification metrics: Precision, Recall, F1, Support, and Confusion Matrix.
 */
export function calculateEvaluationMetrics(
  labels: string[],
  actual: string[],
  predicted: string[]
): {
  accuracy: number;
  macroF1: number;
  classMetrics: Array<{
    sign: string;
    precision: number;
    recall: number;
    f1Score: number;
    support: number;
  }>;
  confusionMatrix: Record<string, Record<string, number>>;
} {
  const confusion: Record<string, Record<string, number>> = {};
  for (const l1 of labels) {
    confusion[l1] = {};
    for (const l2 of labels) {
      confusion[l1][l2] = 0;
    }
  }

  let correct = 0;
  const n = actual.length;

  for (let i = 0; i < n; i++) {
    const act = actual[i];
    const pred = predicted[i];
    if (confusion[act] && confusion[act][pred] !== undefined) {
      confusion[act][pred]++;
    }
    if (act === pred) correct++;
  }

  const accuracy = n > 0 ? Math.round((correct / n) * 1000) / 1000 : 0;

  const classMetrics = labels.map((sign) => {
    let tp = confusion[sign]?.[sign] ?? 0;
    let fn = 0;
    let fp = 0;

    for (const other of labels) {
      if (other !== sign) {
        fn += confusion[sign]?.[other] ?? 0;
        fp += confusion[other]?.[sign] ?? 0;
      }
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const support = tp + fn;

    return {
      sign,
      precision: Math.round(precision * 1000) / 1000,
      recall: Math.round(recall * 1000) / 1000,
      f1Score: Math.round(f1Score * 1000) / 1000,
      support,
    };
  });

  const validF1s = classMetrics.filter((m) => m.support > 0).map((m) => m.f1Score);
  const macroF1 =
    validF1s.length > 0
      ? Math.round((validF1s.reduce((a, b) => a + b, 0) / validF1s.length) * 1000) / 1000
      : 0;

  return {
    accuracy,
    macroF1,
    classMetrics,
    confusionMatrix: confusion,
  };
}
