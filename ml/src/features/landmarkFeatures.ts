/**
 * landmarkFeatures.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Pure landmark feature extractor shared between Node training pipeline
 * and browser inference layer.
 *
 * Output per frame: Float32Array of length FEATURE_DIM = 126
 *   [0..62]   right-hand 21 landmarks, each (x,y,z) normalised:
 *               - translated so wrist (LM 0) is origin
 *               - scale-normalised by palm diagonal (LM 0 → LM 9)
 *   [63..125] left-hand 21 landmarks, same encoding (zeros if absent)
 *
 * The GRU receives a sequence of these frames and learns temporal patterns.
 * We deliberately do NOT pre-compute velocities as explicit features here —
 * the recurrent network learns velocity implicitly, and that generalises better.
 */

import { Landmark3D } from '@signbridge2vision/types';

export const FEATURE_DIM = 126; // 21 landmarks × 3 coords × 2 hands
export const NUM_LANDMARKS = 21;

/**
 * Normalise one hand's 21 landmarks.
 * Returns a flat Float32Array of length 63 (21 × 3).
 * If landmarks is null or too short, returns zeros.
 */
export function normaliseSingleHand(
  landmarks: Landmark3D[] | null
): Float32Array {
  const out = new Float32Array(NUM_LANDMARKS * 3); // zeros by default
  if (!landmarks || landmarks.length < NUM_LANDMARKS) return out;

  // Translate: wrist (LM 0) becomes origin
  const wx = landmarks[0].x;
  const wy = landmarks[0].y;
  const wz = landmarks[0].z;

  // Scale: distance from wrist (LM 0) to middle-finger MCP (LM 9)
  const dx = landmarks[9].x - wx;
  const dy = landmarks[9].y - wy;
  const dz = landmarks[9].z - wz;
  const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;

  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[i * 3 + 0] = (landmarks[i].x - wx) / scale;
    out[i * 3 + 1] = (landmarks[i].y - wy) / scale;
    out[i * 3 + 2] = (landmarks[i].z - wz) / scale;
  }

  return out;
}

/**
 * Build the full 126-element feature vector for one frame.
 * Right hand occupies [0..62], left hand [63..125].
 */
export function extractFrameFeatures(
  rightHand: Landmark3D[] | null,
  leftHand: Landmark3D[] | null
): Float32Array {
  const out = new Float32Array(FEATURE_DIM);
  const right = normaliseSingleHand(rightHand);
  const left  = normaliseSingleHand(leftHand);
  out.set(right, 0);
  out.set(left, 63);
  return out;
}

/**
 * Build a padded feature matrix for a full temporal window.
 * Returns Float32Array of shape [windowSize, FEATURE_DIM] stored row-major.
 *
 * If fewer frames than windowSize are provided the leading frames are zero-padded.
 */
export function buildFeatureMatrix(
  frames: Array<{ rightHand: Landmark3D[] | null; leftHand: Landmark3D[] | null }>,
  windowSize: number
): Float32Array {
  const matrix = new Float32Array(windowSize * FEATURE_DIM); // zero-padded
  const offset = Math.max(0, windowSize - frames.length);

  for (let i = 0; i < frames.length && i < windowSize; i++) {
    const f = extractFrameFeatures(frames[i].rightHand, frames[i].leftHand);
    matrix.set(f, (offset + i) * FEATURE_DIM);
  }

  return matrix;
}

// ─── Utility helpers used by training ────────────────────────────────────────

/**
 * Augmentation: add small Gaussian noise to landmark coordinates.
 * σ should be small (e.g., 0.02) relative to the normalised scale.
 */
export function jitterLandmarks(
  features: Float32Array,
  sigma: number = 0.02
): Float32Array {
  const out = new Float32Array(features.length);
  for (let i = 0; i < features.length; i++) {
    // Box-Muller for Gaussian noise
    const u1 = Math.random() || 1e-9;
    const u2 = Math.random();
    const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * sigma;
    out[i] = features[i] + noise;
  }
  return out;
}

/**
 * Augmentation: time-warp a sequence by randomly stretching or compressing it
 * to fill the target windowSize.
 */
export function timeWarp(
  matrix: Float32Array,
  srcWindowSize: number,
  targetWindowSize: number
): Float32Array {
  const out = new Float32Array(targetWindowSize * FEATURE_DIM);
  for (let t = 0; t < targetWindowSize; t++) {
    const srcT = (t / (targetWindowSize - 1)) * (srcWindowSize - 1);
    const lo = Math.floor(srcT);
    const hi = Math.min(Math.ceil(srcT), srcWindowSize - 1);
    const alpha = srcT - lo;

    for (let f = 0; f < FEATURE_DIM; f++) {
      out[t * FEATURE_DIM + f] =
        (1 - alpha) * matrix[lo * FEATURE_DIM + f] +
        alpha       * matrix[hi * FEATURE_DIM + f];
    }
  }
  return out;
}

/**
 * Mirror a single-hand feature vector along the x-axis.
 * Used for linguistically symmetric signs (LEFT ↔ RIGHT excluded).
 */
export function mirrorFeatures(features: Float32Array): Float32Array {
  const out = new Float32Array(features.length);
  // Right hand → left hand slot, negating x
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[i * 3 + 0] = -features[i * 3 + 0]; // negate x for right hand
    out[i * 3 + 1] =  features[i * 3 + 1];
    out[i * 3 + 2] =  features[i * 3 + 2];
    // Swap right (0..62) and left (63..125) hand slots
    out[(NUM_LANDMARKS + i) * 3 + 0] = -features[i * 3 + 0];
    out[(NUM_LANDMARKS + i) * 3 + 1] =  features[i * 3 + 1];
    out[(NUM_LANDMARKS + i) * 3 + 2] =  features[i * 3 + 2];
  }
  return out;
}
