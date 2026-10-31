import { Landmark3D } from '@signbridge/types';

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
