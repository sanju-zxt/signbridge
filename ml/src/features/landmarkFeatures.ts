/**
 * landmarkFeatures.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Landmark feature extractor shared between Node training pipeline
 * and browser inference layer.
 *
 * Implements:
 * 1. Backward-compatible 126-dimensional hand feature extraction (FEATURE_DIM = 126)
 * 2. Non-manual facial and gaze feature extraction (NON_MANUAL_DIM = 24)
 * 3. Multimodal combined feature extraction (MULTIMODAL_DIM = 150)
 */

import {
  Landmark3D,
  GazeDirection,
  HeadOrientation,
  EyebrowState,
  MouthState,
  NonManualFeatures,
  MultimodalFeatureFrame,
} from '@signbridge/types';

export const FEATURE_DIM = 126; // 21 landmarks x 3 coords x 2 hands
export const NUM_LANDMARKS = 21;
export const NON_MANUAL_DIM = 24; // 24 non-manual grammatical signals
export const MULTIMODAL_DIM = 150; // 126 hand + 24 non-manual

/**
 * Normalise one hand's 21 landmarks.
 * Returns a flat Float32Array of length 63 (21 x 3).
 * If landmarks is null or too short, returns zeros.
 */
export function normaliseSingleHand(
  landmarks: Landmark3D[] | null
): Float32Array {
  const out = new Float32Array(NUM_LANDMARKS * 3);
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
    out[i * 3 + 0] = Math.max(-5.0, Math.min(5.0, (landmarks[i].x - wx) / scale));
    out[i * 3 + 1] = Math.max(-5.0, Math.min(5.0, (landmarks[i].y - wy) / scale));
    out[i * 3 + 2] = Math.max(-5.0, Math.min(5.0, (landmarks[i].z - wz) / scale));
  }

  return out;
}

/**
 * Build the backward-compatible 126-element hand feature vector for one frame.
 * Right hand occupies [0..62], left hand [63..125].
 */
export function extractFrameFeatures(
  rightHand: Landmark3D[] | null,
  leftHand: Landmark3D[] | null
): Float32Array {
  const out = new Float32Array(FEATURE_DIM);
  const right = normaliseSingleHand(rightHand);
  const left = normaliseSingleHand(leftHand);
  out.set(right, 0);
  out.set(left, 63);
  return out;
}

/**
 * Extracts 24-dimensional normalized non-manual facial and gaze features.
 * Robust to occlusion, lighting shifts, and missing face detections.
 */
export function extractNonManualFeatures(
  faceLandmarks: Landmark3D[] | null,
  prevFeatures?: NonManualFeatures | null
): NonManualFeatures {
  const vec = new Float32Array(NON_MANUAL_DIM);

  if (!faceLandmarks || faceLandmarks.length < 168) {
    // Neutral fallback defaults when face is occluded or not detected
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

  // Face spatial reference anchors
  const noseTip = lm(1);
  const noseBridge = lm(168);
  const chin = lm(152);
  const leftTemple = lm(234);
  const rightTemple = lm(454);

  const faceWidth = Math.abs(rightTemple.x - leftTemple.x) || 1.0;
  const faceHeight = Math.abs(chin.y - noseBridge.y) || 1.0;
  const midTempleX = (leftTemple.x + rightTemple.x) / 2;
  const midTempleY = (leftTemple.y + rightTemple.y) / 2;

  // 1. Head Pose (Yaw, Pitch, Roll)
  const yaw = Math.max(-1.0, Math.min(1.0, (noseTip.x - midTempleX) / (faceWidth * 0.4)));
  const pitch = Math.max(-1.0, Math.min(1.0, (noseTip.y - midTempleY) / (faceHeight * 0.4)));
  const roll = Math.max(-1.0, Math.min(1.0, (rightTemple.y - leftTemple.y) / faceWidth));

  // 2. Eye Gaze Direction
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

  // 3. Eyebrows (Grammatical Wh vs Yes/No inflection)
  const leftBrowInner = lm(107);
  const rightBrowInner = lm(336);
  const browMidY = (leftBrowInner.y + rightBrowInner.y) / 2;
  const eyeMidY = (leftEyeInner.y + rightEyeInner.y) / 2;
  const browElevation = Math.max(-1.0, Math.min(1.0, (eyeMidY - browMidY) / (faceHeight * 0.2) - 1.0));
  const browFurrow = Math.max(0.0, Math.min(1.0, 1.0 - Math.abs(rightBrowInner.x - leftBrowInner.x) / (faceWidth * 0.25)));

  const isRaised = browElevation > 0.20;
  const isLowered = browElevation < -0.15 || browFurrow > 0.65;

  // 4. Mouth Shape (Adverbial / descriptive non-manual markers)
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

  // 5. Head Motion Derivatives
  const prevYaw = prevFeatures?.headPose?.yaw ?? yaw;
  const prevPitch = prevFeatures?.headPose?.pitch ?? pitch;
  const dYaw = Math.max(-1.0, Math.min(1.0, (yaw - prevYaw) * 10)); // Head shake derivative
  const dPitch = Math.max(-1.0, Math.min(1.0, (pitch - prevPitch) * 10)); // Head nod derivative

  // Assemble 24-dimensional feature vector
  vec[0] = avgGazeH;
  vec[1] = avgGazeV;
  vec[2] = leftGazeH;
  vec[3] = rightGazeH;
  vec[4] = yaw;
  vec[5] = pitch;
  vec[6] = roll;
  vec[7] = dYaw;
  vec[8] = dPitch;
  vec[9] = Math.abs(dYaw); // Shake magnitude (Negation)
  vec[10] = browElevation;
  vec[11] = browFurrow;
  vec[12] = isRaised ? 1.0 : 0.0;
  vec[13] = isLowered ? 1.0 : 0.0;
  vec[14] = mouthAperture;
  vec[15] = mouthWidth;
  vec[16] = mouthShape === 'open' ? 1.0 : 0.0;
  vec[17] = mouthShape === 'puckered' ? 1.0 : 0.0;
  vec[18] = yaw > 0.20 ? 1.0 : (yaw < -0.20 ? -1.0 : 0.0); // Lateral pointing target quadrant
  vec[19] = avgGazeH > 0.20 ? 1.0 : (avgGazeH < -0.20 ? -1.0 : 0.0);
  vec[20] = pitch > 0.15 ? 1.0 : 0.0; // Looking down / local 'HERE' reference
  vec[21] = Math.abs(dPitch); // Nod magnitude (Affirmation)
  vec[22] = 1.0; // Face tracking confidence
  vec[23] = 1.0; // Valid detection flag

  return {
    vector: vec,
    gaze: { horizontal: avgGazeH, vertical: avgGazeV, target: gazeTarget, confidence: 0.95 },
    headPose: { yaw: yaw * 45, pitch: pitch * 30, roll: roll * 30 },
    eyebrows: { leftHeight: browElevation, rightHeight: browElevation, furrow: browFurrow, raised: isRaised, lowered: isLowered },
    mouth: { aperture: mouthAperture, width: mouthWidth, shape: mouthShape },
    hasFace: true,
  };
}

/**
 * Builds backward-compatible hand vector, non-manual vector, and multimodal combined vector.
 */
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
  combined.set(handFeatures, 0); // [0..125]
  combined.set(nonManual.vector, FEATURE_DIM); // [126..149]

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

/**
 * Build a padded feature matrix for a full temporal window.
 * Returns Float32Array of shape [windowSize, FEATURE_DIM] stored row-major.
 */
export function buildFeatureMatrix(
  frames: Array<{ rightHand: Landmark3D[] | null; leftHand: Landmark3D[] | null }>,
  windowSize: number
): Float32Array {
  const matrix = new Float32Array(windowSize * FEATURE_DIM);
  const startIdx = Math.max(0, windowSize - frames.length);
  const framesToUse = frames.slice(-windowSize);

  for (let i = 0; i < framesToUse.length; i++) {
    const row = extractFrameFeatures(framesToUse[i].rightHand, framesToUse[i].leftHand);
    matrix.set(row, (startIdx + i) * FEATURE_DIM);
  }

  return matrix;
}

/**
 * Adds Gaussian coordinate jitter augmentation for training robustness.
 */
export function jitterLandmarks(matrix: Float32Array, sigma = 0.02): Float32Array {
  const out = new Float32Array(matrix.length);
  for (let i = 0; i < matrix.length; i++) {
    out[i] = matrix[i] + (Math.random() - 0.5) * 2 * sigma;
  }
  return out;
}

/**
 * Resamples / warps the temporal execution speed of a feature matrix.
 */
export function timeWarp(
  matrix: Float32Array,
  factor = 0.2,
  windowSize = 30,
  featureDim = FEATURE_DIM
): Float32Array {
  const out = new Float32Array(matrix.length);
  const warp = 1.0 + (Math.random() - 0.5) * 2 * factor;
  for (let t = 0; t < windowSize; t++) {
    const srcT = Math.min(windowSize - 1, Math.max(0, Math.round(t * warp)));
    const srcOffset = srcT * featureDim;
    const dstOffset = t * featureDim;
    for (let d = 0; d < featureDim; d++) {
      out[dstOffset + d] = matrix[srcOffset + d];
    }
  }
  return out;
}
