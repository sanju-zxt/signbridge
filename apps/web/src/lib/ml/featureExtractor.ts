/**
 * featureExtractor.ts
 * Browser-side landmark feature extraction — identical algorithm to ml/src/features/landmarkFeatures.ts
 */
import type { Landmark3D } from '@signbridge2vision/types';

export const FEATURE_DIM  = 126;  // 63 per hand × 2 hands
const NUM_LANDMARKS       = 21;
const FEATURES_PER_HAND   = NUM_LANDMARKS * 3; // 63

function normaliseHand(lms: Landmark3D[] | null): Float32Array {
  const out = new Float32Array(FEATURES_PER_HAND);
  if (!lms || lms.length < NUM_LANDMARKS) return out;
  const wx = lms[0].x, wy = lms[0].y, wz = lms[0].z;
  const dx = lms[9].x - wx, dy = lms[9].y - wy, dz = lms[9].z - wz;
  const scale = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1.0;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[i*3+0] = (lms[i].x - wx) / scale;
    out[i*3+1] = (lms[i].y - wy) / scale;
    out[i*3+2] = (lms[i].z - wz) / scale;
  }
  return out;
}

export function extractFrameFeatures(
  rightHand: Landmark3D[] | null,
  leftHand:  Landmark3D[] | null
): Float32Array {
  const out = new Float32Array(FEATURE_DIM);
  out.set(normaliseHand(rightHand), 0);
  out.set(normaliseHand(leftHand), FEATURES_PER_HAND);
  return out;
}

export function buildFeatureMatrix(frames: Float32Array[]): Float32Array {
  const matrix = new Float32Array(frames.length * FEATURE_DIM);
  for (let t = 0; t < frames.length; t++) matrix.set(frames[t], t * FEATURE_DIM);
  return matrix;
}