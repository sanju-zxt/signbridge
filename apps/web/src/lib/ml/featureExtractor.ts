/**
 * featureExtractor.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Browser-side landmark feature extraction matching ml/src/features/landmarkFeatures.ts
 *
 * Implements:
 * 1. Backward-compatible 126-dimensional hand feature extraction (FEATURE_DIM = 126)
 * 2. Non-manual facial and gaze feature extraction (NON_MANUAL_DIM = 24)
 * 3. Multimodal combined feature extraction (MULTIMODAL_DIM = 150)
 */

import type {
  Landmark3D,
  GazeDirection,
  HeadOrientation,
  EyebrowState,
  MouthState,
  NonManualFeatures,
  MultimodalFeatureFrame,
} from '@signbridge/types';

export const FEATURE_DIM = 126; // 63 per hand x 2 hands
export const NUM_LANDMARKS = 21;
export const FEATURES_PER_HAND = NUM_LANDMARKS * 3; // 63
export const NON_MANUAL_DIM = 24;
export const MULTIMODAL_DIM = 150;

export function normaliseSingleHand(lms: Landmark3D[] | null): Float32Array {
  const out = new Float32Array(FEATURES_PER_HAND);
  if (!lms || lms.length < NUM_LANDMARKS) return out;
  const wx = lms[0].x, wy = lms[0].y, wz = lms[0].z;
  const dx = lms[9].x - wx, dy = lms[9].y - wy, dz = lms[9].z - wz;
  const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[i * 3 + 0] = Math.max(-5.0, Math.min(5.0, (lms[i].x - wx) / scale));
    out[i * 3 + 1] = Math.max(-5.0, Math.min(5.0, (lms[i].y - wy) / scale));
    out[i * 3 + 2] = Math.max(-5.0, Math.min(5.0, (lms[i].z - wz) / scale));
  }
  return out;
}

export function extractFrameFeatures(
  rightHand: Landmark3D[] | null,
  leftHand: Landmark3D[] | null
): Float32Array {
  const out = new Float32Array(FEATURE_DIM);
  out.set(normaliseSingleHand(rightHand), 0);
  out.set(normaliseSingleHand(leftHand), FEATURES_PER_HAND);
  return out;
}

export function extractNonManualFeatures(
  faceLandmarks: Landmark3D[] | null,
  prevFeatures?: NonManualFeatures | null
): NonManualFeatures {
  const vec = new Float32Array(NON_MANUAL_DIM);

  if (!faceLandmarks || faceLandmarks.length < 168) {
    return {
      vector: vec,
      gaze: { horizontal: 0, vertical: 0, target: 'center', confidence: 0 },
      headPose: { yaw: 0, pitch: 0, roll: 0 },
      eyebrows: { leftHeight: 0, rightHeight: 0, furrow: 0, raised: false, lowered: false },
      mouth: { aperture: 0, width: 0, shape: 'neutral' },
      hasFace: false,
    };
  }

  const lm = (idx: number): Landmark3D => faceLandmarks[idx] || { x: 0, y: 0, z: 0 };

  const noseTip = lm(1);
  const noseBridge = lm(168);
  const chin = lm(152);
  const leftTemple = lm(234);
  const rightTemple = lm(454);

  const faceWidth = Math.abs(rightTemple.x - leftTemple.x) || 1.0;
  const faceHeight = Math.abs(chin.y - noseBridge.y) || 1.0;
  const midTempleX = (leftTemple.x + rightTemple.x) / 2;
  const midTempleY = (leftTemple.y + rightTemple.y) / 2;

  const yaw = Math.max(-1.0, Math.min(1.0, (noseTip.x - midTempleX) / (faceWidth * 0.4)));
  const pitch = Math.max(-1.0, Math.min(1.0, (noseTip.y - midTempleY) / (faceHeight * 0.4)));
  const roll = Math.max(-1.0, Math.min(1.0, (rightTemple.y - leftTemple.y) / faceWidth));

  const leftEyeInner = lm(133);
  const leftEyeOuter = lm(33);
  const rightEyeInner = lm(362);
  const rightEyeOuter = lm(263);

  const leftPupil = faceLandmarks.length > 468 ? lm(468) : {
    x: (leftEyeInner.x + leftEyeOuter.x) / 2,
    y: (leftEyeInner.y + leftEyeOuter.y) / 2,
    z: (leftEyeInner.z + leftEyeOuter.z) / 2,
  };
  const rightPupil = faceLandmarks.length > 473 ? lm(473) : {
    x: (rightEyeInner.x + rightEyeOuter.x) / 2,
    y: (rightEyeInner.y + rightEyeOuter.y) / 2,
    z: (rightEyeInner.z + rightEyeOuter.z) / 2,
  };

  const leftEyeWidth = Math.abs(leftEyeOuter.x - leftEyeInner.x) || 0.05;
  const rightEyeWidth = Math.abs(rightEyeOuter.x - rightEyeInner.x) || 0.05;

  const leftGazeH = ((leftPupil.x - (leftEyeInner.x + leftEyeOuter.x) / 2) / leftEyeWidth) * 2.0;
  const rightGazeH = ((rightPupil.x - (rightEyeInner.x + rightEyeOuter.x) / 2) / rightEyeWidth) * 2.0;
  const avgGazeH = Math.max(-1.0, Math.min(1.0, (leftGazeH + rightGazeH) / 2));
  const avgGazeV = Math.max(-1.0, Math.min(1.0, -((leftPupil.y + rightPupil.y) / 2 - midTempleY) / faceHeight));

  let gazeTarget: 'center' | 'left' | 'right' | 'up' | 'down' = 'center';
  if (avgGazeH > 0.25) gazeTarget = 'right';
  else if (avgGazeH < -0.25) gazeTarget = 'left';
  else if (avgGazeV > 0.25) gazeTarget = 'up';
  else if (avgGazeV < -0.25) gazeTarget = 'down';

  const leftBrowInner = lm(107);
  const rightBrowInner = lm(336);
  const browMidY = (leftBrowInner.y + rightBrowInner.y) / 2;
  const eyeMidY = (leftEyeInner.y + rightEyeInner.y) / 2;
  const browElevation = Math.max(-1.0, Math.min(1.0, (eyeMidY - browMidY) / (faceHeight * 0.2) - 1.0));
  const browFurrow = Math.max(0.0, Math.min(1.0, 1.0 - Math.abs(rightBrowInner.x - leftBrowInner.x) / (faceWidth * 0.25)));

  const isRaised = browElevation > 0.20;
  const isLowered = browElevation < -0.15 || browFurrow > 0.65;

  const upperLip = lm(13);
  const lowerLip = lm(14);
  const mouthLeft = lm(61);
  const mouthRight = lm(291);

  const mouthAperture = Math.max(0.0, Math.min(1.0, (lowerLip.y - upperLip.y) / (faceHeight * 0.25)));
  const mouthWidth = Math.max(0.0, Math.min(1.0, (mouthRight.x - mouthLeft.x) / (faceWidth * 0.5)));

  let mouthShape: 'neutral' | 'open' | 'flat' | 'puckered' = 'neutral';
  if (mouthAperture > 0.35) mouthShape = 'open';
  else if (mouthWidth > 0.75) mouthShape = 'flat';
  else if (mouthWidth < 0.40 && mouthAperture < 0.25) mouthShape = 'puckered';

  const prevYaw = prevFeatures?.headPose?.yaw ?? yaw;
  const prevPitch = prevFeatures?.headPose?.pitch ?? pitch;
  const dYaw = Math.max(-1.0, Math.min(1.0, (yaw - prevYaw) * 10));
  const dPitch = Math.max(-1.0, Math.min(1.0, (pitch - prevPitch) * 10));

  vec[0] = avgGazeH;
  vec[1] = avgGazeV;
  vec[2] = leftGazeH;
  vec[3] = rightGazeH;
  vec[4] = yaw;
  vec[5] = pitch;
  vec[6] = roll;
  vec[7] = dYaw;
  vec[8] = dPitch;
  vec[9] = Math.abs(dYaw);
  vec[10] = browElevation;
  vec[11] = browFurrow;
  vec[12] = isRaised ? 1.0 : 0.0;
  vec[13] = isLowered ? 1.0 : 0.0;
  vec[14] = mouthAperture;
  vec[15] = mouthWidth;
  vec[16] = mouthShape === 'open' ? 1.0 : 0.0;
  vec[17] = mouthShape === 'puckered' ? 1.0 : 0.0;
  vec[18] = yaw > 0.20 ? 1.0 : (yaw < -0.20 ? -1.0 : 0.0);
  vec[19] = avgGazeH > 0.20 ? 1.0 : (avgGazeH < -0.20 ? -1.0 : 0.0);
  vec[20] = pitch > 0.15 ? 1.0 : 0.0;
  vec[21] = Math.abs(dPitch);
  vec[22] = 1.0;
  vec[23] = 1.0;

  return {
    vector: vec,
    gaze: { horizontal: avgGazeH, vertical: avgGazeV, target: gazeTarget, confidence: 0.95 },
    headPose: { yaw: yaw * 45, pitch: pitch * 30, roll: roll * 30 },
    eyebrows: { leftHeight: browElevation, rightHeight: browElevation, furrow: browFurrow, raised: isRaised, lowered: isLowered },
    mouth: { aperture: mouthAperture, width: mouthWidth, shape: mouthShape },
    hasFace: true,
  };
}

export function extractMultimodalFrame(
  rightHand: Landmark3D[] | null,
  leftHand: Landmark3D[] | null,
  faceLandmarks: Landmark3D[] | null,
  prevNonManual?: NonManualFeatures | null,
  timestampMs = Date.now()
): MultimodalFeatureFrame {
  const handFeatures = extractFrameFeatures(rightHand, leftHand);
  const nonManual = extractNonManualFeatures(faceLandmarks, prevNonManual);

  const combined = new Float32Array(MULTIMODAL_DIM);
  combined.set(handFeatures, 0);
  combined.set(nonManual.vector, FEATURE_DIM);

  return {
    timestamp: timestampMs,
    handFeatures,
    nonManualFeatures: nonManual.vector,
    combinedFeatures: combined,
    hasHand: (rightHand !== null && rightHand.length >= 21) || (leftHand !== null && leftHand.length >= 21),
    hasFace: nonManual.hasFace,
    rightHandPresent: rightHand !== null && rightHand.length >= 21,
    leftHandPresent: leftHand !== null && leftHand.length >= 21,
  };
}

export function buildFeatureMatrix(frames: Float32Array[]): Float32Array {
  const matrix = new Float32Array(frames.length * FEATURE_DIM);
  for (let t = 0; t < frames.length; t++) matrix.set(frames[t], t * FEATURE_DIM);
  return matrix;
}
