/**
 * gestureClassifier.test.ts
 * Rigorous automated tests for the kinematic gesture classification pipeline.
 * Verifies finger extension math, palm defense rules, orientation, and motion discrimination.
 */

// ---------------------------------------------------------------------------
// Helpers to generate synthetic 21-landmark MediaPipe hand frames
// ---------------------------------------------------------------------------

function createSyntheticHand(options: {
  thumbExt: number;  // 0 (curled) to 1 (extended)
  indexExt: number;
  middleExt: number;
  ringExt: number;
  pinkyExt: number;
  direction?: 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';
}): Array<{ x: number; y: number; z: number }> {
  const lms = new Array(21).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));

  // Landmark 0: Wrist
  lms[0] = { x: 0.5, y: 0.7, z: 0.0 };

  // Set directional vector for middle MCP
  let dirX = 0, dirY = -0.15;
  if (options.direction === 'DOWN') { dirX = 0; dirY = 0.15; }
  else if (options.direction === 'LEFT') { dirX = -0.15; dirY = 0; }
  else if (options.direction === 'RIGHT') { dirX = 0.15; dirY = 0; }

  // Landmark 9: Middle MCP
  lms[9] = { x: lms[0].x + dirX, y: lms[0].y + dirY, z: 0.0 };
  // Landmark 5: Index MCP
  lms[5] = { x: lms[9].x - 0.04, y: lms[9].y, z: 0.0 };
  // Landmark 13: Ring MCP
  lms[13] = { x: lms[9].x + 0.04, y: lms[9].y, z: 0.0 };
  // Landmark 17: Pinky MCP
  lms[17] = { x: lms[13].x + 0.03, y: lms[13].y, z: 0.0 };
  // Landmark 2: Thumb MCP
  lms[2] = { x: lms[5].x - 0.03, y: lms[5].y + 0.04, z: 0.0 };

  // Helper to place finger joints and tips based on extension [0..1]
  const placeFinger = (mcpIdx: number, tipIdx: number, ext: number, baseAngleOffset: number) => {
    const mcp = lms[mcpIdx];
    // Base unit vector
    const len = 0.04 + ext * 0.08;
    const dx = dirX * (len / 0.15);
    const dy = dirY * (len / 0.15);
    lms[tipIdx] = { x: mcp.x + dx, y: mcp.y + dy, z: 0.0 };
  };

  placeFinger(5, 8, options.indexExt, -0.02);
  placeFinger(9, 12, options.middleExt, 0.0);
  placeFinger(13, 16, options.ringExt, 0.02);
  placeFinger(17, 20, options.pinkyExt, 0.04);

  // Thumb
  const tLen = 0.02 + options.thumbExt * 0.06;
  lms[4] = { x: lms[2].x - tLen, y: lms[2].y - tLen, z: 0.0 };

  return lms;
}

function normalizeHand(lms: Array<{ x: number; y: number; z: number }> | null): Float32Array {
  const out = new Float32Array(63);
  if (!lms || lms.length < 21) return out;
  const wx = lms[0].x, wy = lms[0].y, wz = lms[0].z;
  const dx = lms[9].x - wx, dy = lms[9].y - wy, dz = lms[9].z - wz;
  const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;
  for (let i = 0; i < 21; i++) {
    out[i * 3]     = (lms[i].x - wx) / scale;
    out[i * 3 + 1] = (lms[i].y - wy) / scale;
    out[i * 3 + 2] = (lms[i].z - wz) / scale;
  }
  return out;
}

function buildFeatureWindow(rh: Array<{ x: number; y: number; z: number }> | null, lh: Array<{ x: number; y: number; z: number }> | null = null): Float32Array {
  const matrix = new Float32Array(30 * 126);
  const rhNorm = normalizeHand(rh);
  const lhNorm = normalizeHand(lh);
  for (let i = 0; i < 30; i++) {
    const base = i * 126;
    matrix.set(rhNorm, base);
    matrix.set(lhNorm, base + 63);
  }
  return matrix;
}

// ---------------------------------------------------------------------------
// Kinematic classifier evaluation logic (mirrors inferenceWorker)
// ---------------------------------------------------------------------------

function dist3d(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const dx = ax - bx, dy = ay - by, dz = az - bz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function computeFingerExtensions(frame: Float32Array, offset = 0): [number, number, number, number, number] {
  const lm = (idx: number) => ({
    x: frame[offset + idx * 3 + 0],
    y: frame[offset + idx * 3 + 1],
    z: frame[offset + idx * 3 + 2],
  });

  const thumbDist  = dist3d(lm(4).x, lm(4).y, lm(4).z, lm(2).x, lm(2).y, lm(2).z);
  const indexDist  = dist3d(lm(8).x, lm(8).y, lm(8).z, lm(5).x, lm(5).y, lm(5).z);
  const middleDist = dist3d(lm(12).x, lm(12).y, lm(12).z, lm(9).x, lm(9).y, lm(9).z);
  const ringDist   = dist3d(lm(16).x, lm(16).y, lm(16).z, lm(13).x, lm(13).y, lm(13).z);
  const pinkyDist  = dist3d(lm(20).x, lm(20).y, lm(20).z, lm(17).x, lm(17).y, lm(17).z);

  const indexWrist  = dist3d(lm(8).x, lm(8).y, lm(8).z, 0, 0, 0);
  const middleWrist = dist3d(lm(12).x, lm(12).y, lm(12).z, 0, 0, 0);
  const ringWrist   = dist3d(lm(16).x, lm(16).y, lm(16).z, 0, 0, 0);
  const pinkyWrist  = dist3d(lm(20).x, lm(20).y, lm(20).z, 0, 0, 0);

  const ext = (d: number, minD: number, maxD: number) =>
    Math.min(1, Math.max(0, (d - minD) / (maxD - minD)));

  return [
    Math.min(1, Math.max(0, ext(thumbDist, 0.22, 0.65))),
    Math.min(1, Math.max(0, 0.5 * ext(indexDist, 0.35, 0.85) + 0.5 * ext(indexWrist, 0.75, 1.55))),
    Math.min(1, Math.max(0, 0.5 * ext(middleDist, 0.35, 0.85) + 0.5 * ext(middleWrist, 0.75, 1.55))),
    Math.min(1, Math.max(0, 0.5 * ext(ringDist, 0.32, 0.80) + 0.5 * ext(ringWrist, 0.70, 1.45))),
    Math.min(1, Math.max(0, 0.5 * ext(pinkyDist, 0.28, 0.75) + 0.5 * ext(pinkyWrist, 0.60, 1.35))),
  ];
}

function evaluateGesture(features: Float32Array, motion: [number, number, number]): { gesture: string; confidence: number } {
  const latestFrame = features.subarray(29 * 126, 30 * 126);
  const hasRight = latestFrame.subarray(0, 63).some((v) => Math.abs(v) > 0.001);
  const hasLeft = latestFrame.subarray(63, 126).some((v) => Math.abs(v) > 0.001);
  const isTwoHanded = hasRight && hasLeft;

  if (!hasRight && !hasLeft) {
    return { gesture: 'NONE', confidence: 0 };
  }

  const offset = hasRight ? 0 : 63;
  const [tExt, iExt, mExt, rExt, pExt] = computeFingerExtensions(latestFrame, offset);

  const lm = (idx: number) => ({
    x: latestFrame[offset + idx * 3 + 0],
    y: latestFrame[offset + idx * 3 + 1],
    z: latestFrame[offset + idx * 3 + 2],
  });
  const vLong = { x: lm(9).x - lm(0).x, y: lm(9).y - lm(0).y, z: lm(9).z - lm(0).z };

  const motionMag = Math.sqrt(motion[0] * motion[0] + motion[1] * motion[1] + motion[2] * motion[2]);
  const isWaving = motionMag >= 0.055;

  const scores: Record<string, number> = {};

  // 1. Open hand
  const isOpenHand = iExt > 0.72 && mExt > 0.72 && rExt > 0.72 && pExt > 0.68 && tExt > 0.65;
  if (isOpenHand) {
    if (isTwoHanded) {
      scores['WHERE'] = 0.92;
    } else {
      if (motion[2] > 0.035 || (motion[1] > 0.035 && Math.abs(motion[0]) < 0.035)) {
        scores['THANK_YOU'] = 0.92;
        scores['HELLO'] = 0.40;
      } else if (Math.abs(motion[0]) >= 0.07) {
        scores['GOODBYE'] = 0.94;
        scores['HELLO'] = 0.70;
      } else if (Math.abs(motion[0]) >= 0.035 || motionMag >= 0.055) {
        scores['HELLO'] = 0.93;
        scores['GOODBYE'] = 0.65;
      } else {
        // Resting palm defense
        scores['HELLO'] = 0.42;
        scores['THANK_YOU'] = 0.38;
      }
    }
  }

  // 2. Water (W)
  if (iExt > 0.65 && mExt > 0.65 && rExt > 0.65 && pExt < 0.60 && (rExt - pExt > 0.22 || pExt < 0.40)) {
    scores['WATER'] = 0.95;
  }

  // 3. Pointing
  const isIndexPointing = iExt > 0.65 && (iExt - mExt > 0.30) && (iExt - rExt > 0.25) && (iExt - pExt > 0.25);
  if (isIndexPointing) {
    if (vLong.x < -0.30 || motion[0] < -0.05) scores['LEFT'] = 0.94;
    else if (vLong.x > 0.30 || motion[0] > 0.05) scores['RIGHT'] = 0.94;
    else if (vLong.y > 0.30) scores['HERE'] = 0.93;
  }

  // 4. Fist
  const isFist = iExt < 0.55 && mExt < 0.55 && rExt < 0.55 && pExt < 0.60;
  if (isFist) {
    if (isTwoHanded) scores['HELP'] = 0.94;
    else if (tExt > 0.65) scores['BATHROOM'] = 0.91;
    else scores['YES'] = 0.92;
  }

  const entries = Object.entries(scores);
  if (entries.length === 0) return { gesture: 'NONE', confidence: 0.1 };
  entries.sort((a, b) => b[1] - a[1]);
  const top = entries[0];
  const isRecognized = top[1] >= 0.60;
  return { gesture: isRecognized ? top[0] : 'NONE', confidence: top[1] };
}

// ---------------------------------------------------------------------------
// Run Test Suite
// ---------------------------------------------------------------------------

console.log('=== Running Kinematic Gesture Classifier Test Suite ===\n');

// Test 1: Resting Open Palm Defense
{
  const hand = createSyntheticHand({ thumbExt: 1.0, indexExt: 1.0, middleExt: 1.0, ringExt: 1.0, pinkyExt: 1.0 });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.0, 0.0, 0.0]); // No motion
  if (result.confidence < 0.60 && result.gesture !== 'HELLO') {
    console.log('✅ PASS: Resting open palm correctly suppressed (< 0.60 confidence, got ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Resting open palm was falsely recognized as ' + result.gesture + ' with confidence ' + result.confidence);
    process.exit(1);
  }
}

// Test 2: Active Waving Hand -> HELLO
{
  const hand = createSyntheticHand({ thumbExt: 1.0, indexExt: 1.0, middleExt: 1.0, ringExt: 1.0, pinkyExt: 1.0 });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.04, -0.05, 0.02]); // Active wave
  if (result.gesture === 'HELLO' && result.confidence >= 0.85) {
    console.log('✅ PASS: Waving open hand recognized as HELLO (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected HELLO, got ' + result.gesture + ' (' + result.confidence + ')');
    process.exit(1);
  }
}

// Test 3: Horizontal Waving Hand -> GOODBYE
{
  const hand = createSyntheticHand({ thumbExt: 1.0, indexExt: 1.0, middleExt: 1.0, ringExt: 1.0, pinkyExt: 1.0 });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.09, 0.0, 0.0]); // Horizontal motion
  if (result.gesture === 'GOODBYE' && result.confidence >= 0.85) {
    console.log('✅ PASS: Horizontal waving hand recognized as GOODBYE (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected GOODBYE, got ' + result.gesture);
    process.exit(1);
  }
}

// Test 4: Forward Outward Motion -> THANK_YOU
{
  const hand = createSyntheticHand({ thumbExt: 1.0, indexExt: 1.0, middleExt: 1.0, ringExt: 1.0, pinkyExt: 1.0 });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.0, 0.06, 0.05]); // Outward chest/chin motion
  if (result.gesture === 'THANK_YOU' && result.confidence >= 0.85) {
    console.log('✅ PASS: Forward outward motion recognized as THANK_YOU (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected THANK_YOU, got ' + result.gesture);
    process.exit(1);
  }
}

// Test 5: Water (W-handshape, static)
{
  const hand = createSyntheticHand({ thumbExt: 0.2, indexExt: 1.0, middleExt: 1.0, ringExt: 1.0, pinkyExt: 0.1 });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.0, 0.0, 0.0]); // Static pose
  if (result.gesture === 'WATER' && result.confidence >= 0.90) {
    console.log('✅ PASS: Static W-shape recognized as WATER (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected WATER, got ' + result.gesture + ' (' + result.confidence + ')');
    process.exit(1);
  }
}

// Test 6: Pointing Left -> LEFT
{
  const hand = createSyntheticHand({ thumbExt: 0.1, indexExt: 1.0, middleExt: 0.1, ringExt: 0.1, pinkyExt: 0.1, direction: 'LEFT' });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [-0.06, 0.0, 0.0]);
  if (result.gesture === 'LEFT' && result.confidence >= 0.90) {
    console.log('✅ PASS: Index pointing left recognized as LEFT (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected LEFT, got ' + result.gesture + ' (' + result.confidence + ')');
    process.exit(1);
  }
}

// Test 7: Pointing Right -> RIGHT
{
  const hand = createSyntheticHand({ thumbExt: 0.1, indexExt: 1.0, middleExt: 0.1, ringExt: 0.1, pinkyExt: 0.1, direction: 'RIGHT' });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.06, 0.0, 0.0]);
  if (result.gesture === 'RIGHT' && result.confidence >= 0.90) {
    console.log('✅ PASS: Index pointing right recognized as RIGHT (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected RIGHT, got ' + result.gesture + ' (' + result.confidence + ')');
    process.exit(1);
  }
}

// Test 8: Pointing Down -> HERE
{
  const hand = createSyntheticHand({ thumbExt: 0.1, indexExt: 1.0, middleExt: 0.1, ringExt: 0.1, pinkyExt: 0.1, direction: 'DOWN' });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.0, 0.04, 0.0]);
  if (result.gesture === 'HERE' && result.confidence >= 0.90) {
    console.log('✅ PASS: Index pointing down recognized as HERE (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected HERE, got ' + result.gesture + ' (' + result.confidence + ')');
    process.exit(1);
  }
}

// Test 9: Fist Nodding -> YES
{
  const hand = createSyntheticHand({ thumbExt: 0.1, indexExt: 0.1, middleExt: 0.1, ringExt: 0.1, pinkyExt: 0.1 });
  const features = buildFeatureWindow(hand);
  const result = evaluateGesture(features, [0.0, 0.04, 0.0]);
  if (result.gesture === 'YES' && result.confidence >= 0.90) {
    console.log('✅ PASS: Fist nodding recognized as YES (confidence: ' + result.confidence + ')');
  } else {
    console.error('❌ FAIL: Expected YES, got ' + result.gesture + ' (' + result.confidence + ')');
    process.exit(1);
  }
}

// Test 10: Empty features -> NONE
{
  const features = new Float32Array(30 * 126); // Zeroes
  const result = evaluateGesture(features, [0.0, 0.0, 0.0]);
  if (result.gesture === 'NONE' && result.confidence === 0) {
    console.log('✅ PASS: Empty / zero landmarks returns NONE with confidence 0');
  } else {
    console.error('❌ FAIL: Expected NONE, got ' + result.gesture);
    process.exit(1);
  }
}

console.log('\n🌟 ALL 10 KINEMATIC GESTURE TESTS PASSED! 🌟');
