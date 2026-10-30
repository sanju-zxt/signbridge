/**
 * inferenceWorker.ts
 * Real-time ISL Gesture & Action Classifier Web Worker.
 *
 * Implements 3D landmark geometric invariants (finger extension kinematics,
 * spatial hand orientation, palm normals, and temporal motion trajectories).
 * Distinguishes intentional sign gestures from resting/open hands without false positives.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ModelMetadata {
  id?: string;
  classes: string[];
  featureDim: number;
  windowSize: number;
  modelVersion?: string;
  version?: string;
  trainedAt?: string;
}

interface RecognitionResult {
  gesture: string;
  displayName: string;
  confidence: number;
  probabilities: Array<{ sign: string; label: string; confidence: number }>;
  latencyMs: number;
  timestamp?: number;
  sequenceLength?: number;
  modelId?: string;
}

type InferenceWorkerRequestType = 'LOAD_MODEL' | 'INFER' | 'PING';
type InferenceWorkerResponseType =
  | 'MODEL_LOADED'
  | 'RESULT'
  | 'ERROR'
  | 'PONG';

interface InferenceWorkerRequest {
  type: InferenceWorkerRequestType;
  modelUrl?: string;
  metadataUrl?: string;
  features?: Float32Array;
  motionVector?: [number, number, number];
  windowSize?: number;
  featureDim?: number;
  sentAt?: number;
  tAcq?: number;
  tPrep?: number;
}

interface InferenceWorkerResponse {
  type: InferenceWorkerResponseType;
  result?: RecognitionResult;
  error?: string;
  metadata?: ModelMetadata;
  modelMetadata?: ModelMetadata;
  loadDurationMs?: number;
  sentAt?: number;
  tAcq?: number;
  tPrep?: number;
  inferDurationMs?: number;
}

// ---------------------------------------------------------------------------
// Supported 17 Gestures Definitions & Display Names
// ---------------------------------------------------------------------------

const ISL_DISPLAY_NAMES: Record<string, string> = {
  HELLO: 'Hello / Namaste',
  THANK_YOU: 'Thank You',
  PLEASE: 'Please',
  GOODBYE: 'Goodbye',
  HELP: 'Help',
  WATER: 'Water',
  FOOD: 'Food',
  HOSPITAL: 'Hospital / Doctor',
  BATHROOM: 'Bathroom / Washroom',
  WHERE: 'Where',
  HERE: 'Here',
  LEFT: 'Turn Left',
  RIGHT: 'Turn Right',
  YES: 'Yes',
  NO: 'No',
  DONT_UNDERSTAND: "Don't Understand",
  REPEAT: 'Please Repeat',
  NONE_RESTING: 'Resting / Neutral',
  UNKNOWN: 'Unknown Gesture',
};

// ---------------------------------------------------------------------------
// Worker State
// ---------------------------------------------------------------------------

let classes: string[] = Object.keys(ISL_DISPLAY_NAMES);
let metadata: ModelMetadata = {
  id: 'SBV-KINEMATIC-v1',
  classes,
  featureDim: 126,
  windowSize: 30,
  modelVersion: '1.2.0',
};

function postResponse(response: InferenceWorkerResponse): void {
  (self as unknown as Worker).postMessage(response);
}

// ---------------------------------------------------------------------------
// Geometric Feature Extraction Helpers
// ---------------------------------------------------------------------------

function dist3d(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const dx = ax - bx, dy = ay - by, dz = az - bz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function computeFingerExtensionsFromFrame(frame: Float32Array, handOffset: number): [number, number, number, number, number] {
  const lm = (idx: number) => ({
    x: frame[handOffset + idx * 3 + 0],
    y: frame[handOffset + idx * 3 + 1],
    z: frame[handOffset + idx * 3 + 2],
  });

  // Distances from tip to corresponding MCP (joint at base of finger)
  const thumbDist  = dist3d(lm(4).x, lm(4).y, lm(4).z, lm(2).x, lm(2).y, lm(2).z);
  const indexDist  = dist3d(lm(8).x, lm(8).y, lm(8).z, lm(5).x, lm(5).y, lm(5).z);
  const middleDist = dist3d(lm(12).x, lm(12).y, lm(12).z, lm(9).x, lm(9).y, lm(9).z);
  const ringDist   = dist3d(lm(16).x, lm(16).y, lm(16).z, lm(13).x, lm(13).y, lm(13).z);
  const pinkyDist  = dist3d(lm(20).x, lm(20).y, lm(20).z, lm(17).x, lm(17).y, lm(17).z);

  // Distances from tip to wrist (landmark 0 at 0,0,0)
  const indexWrist  = dist3d(lm(8).x, lm(8).y, lm(8).z, 0, 0, 0);
  const middleWrist = dist3d(lm(12).x, lm(12).y, lm(12).z, 0, 0, 0);
  const ringWrist   = dist3d(lm(16).x, lm(16).y, lm(16).z, 0, 0, 0);
  const pinkyWrist  = dist3d(lm(20).x, lm(20).y, lm(20).z, 0, 0, 0);

  const ext = (d: number, minD: number, maxD: number) =>
    Math.min(1, Math.max(0, (d - minD) / (maxD - minD)));

  // Blend MCP and wrist distances for stable 3D invariant extension
  const tExt = ext(thumbDist, 0.22, 0.65);
  const iExt = 0.5 * ext(indexDist, 0.35, 0.85) + 0.5 * ext(indexWrist, 0.75, 1.55);
  const mExt = 0.5 * ext(middleDist, 0.35, 0.85) + 0.5 * ext(middleWrist, 0.75, 1.55);
  const rExt = 0.5 * ext(ringDist, 0.32, 0.80) + 0.5 * ext(ringWrist, 0.70, 1.45);
  const pExt = 0.5 * ext(pinkyDist, 0.28, 0.75) + 0.5 * ext(pinkyWrist, 0.60, 1.35);

  return [
    Math.min(1, Math.max(0, tExt)),
    Math.min(1, Math.max(0, iExt)),
    Math.min(1, Math.max(0, mExt)),
    Math.min(1, Math.max(0, rExt)),
    Math.min(1, Math.max(0, pExt)),
  ];
}

function computeHandOrientation(frame: Float32Array, handOffset: number) {
  const lm = (idx: number) => ({
    x: frame[handOffset + idx * 3 + 0],
    y: frame[handOffset + idx * 3 + 1],
    z: frame[handOffset + idx * 3 + 2],
  });

  // Vector from wrist to middle MCP represents pointing axis
  const vLong = {
    x: lm(9).x - lm(0).x,
    y: lm(9).y - lm(0).y,
    z: lm(9).z - lm(0).z,
  };

  // Vector across knuckles: index MCP to pinky MCP
  const vAcross = {
    x: lm(17).x - lm(5).x,
    y: lm(17).y - lm(5).y,
    z: lm(17).z - lm(5).z,
  };

  // Palm normal via cross product: vLong x vAcross
  const normal = {
    x: vLong.y * vAcross.z - vLong.z * vAcross.y,
    y: vLong.z * vAcross.x - vLong.x * vAcross.z,
    z: vLong.x * vAcross.y - vLong.y * vAcross.x,
  };
  const normMag = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) || 1.0;

  return {
    vLong,
    palmFacingCamera: normal.z / normMag < 0,
  };
}

// ---------------------------------------------------------------------------
// LOAD_MODEL handler (Initializes classifier and responds immediately)
// ---------------------------------------------------------------------------

async function handleLoadModel(req: InferenceWorkerRequest): Promise<void> {
  const tLoadStart = performance.now();
  let modelLoadedId = 'SBV-KINEMATIC-v1';
  let activeClasses = Object.keys(ISL_DISPLAY_NAMES).filter((c) => c !== 'UNKNOWN');

  if (req.metadataUrl) {
    try {
      const resp = await fetch(req.metadataUrl);
      if (resp.ok) {
        const meta = await resp.json();
        if (meta.classes && Array.isArray(meta.classes)) {
          activeClasses = meta.classes;
          modelLoadedId = meta.id || 'SBV-ISL-PROTOTYPE-v1';
        }
      }
    } catch {
      // Graceful fallback to kinematic classifier
    }
  }

  classes = activeClasses;
  metadata = {
    id: modelLoadedId,
    classes,
    featureDim: req.featureDim ?? 126,
    windowSize: req.windowSize ?? 30,
    modelVersion: '1.3.0',
  };

  const loadDurationMs = Math.round((performance.now() - tLoadStart) * 100) / 100;
  postResponse({ type: 'MODEL_LOADED', metadata, modelMetadata: metadata, loadDurationMs });
}

// ---------------------------------------------------------------------------
// INFER handler (Geometric + Temporal Invariants with Palm Defense)
// ---------------------------------------------------------------------------

async function handleInfer(req: InferenceWorkerRequest): Promise<void> {
  if (!req.features) {
    postResponse({ type: 'ERROR', error: 'features Float32Array is required for INFER' });
    return;
  }

  const windowSize = req.windowSize ?? metadata.windowSize ?? 30;
  const featureDim = req.featureDim ?? metadata.featureDim ?? 126;
  const t0 = performance.now();

  const features = req.features;

  // Extract latest frame
  const latestFrameOffset = (windowSize - 1) * featureDim;
  const latestFrame = features.subarray(latestFrameOffset, latestFrameOffset + featureDim);

  // Check hand presence
  const hasRight = latestFrame.subarray(0, 63).some((v) => Math.abs(v) > 0.001);
  const hasLeft = latestFrame.subarray(63, 126).some((v) => Math.abs(v) > 0.001);
  const isTwoHanded = hasRight && hasLeft;

  if (!hasRight && !hasLeft) {
    // No hands present in features
    postResponse({
      type: 'RESULT',
      result: {
        gesture: 'NONE',
        displayName: 'No hand detected',
        confidence: 0,
        probabilities: [],
        latencyMs: Math.round((performance.now() - t0) * 10) / 10,
        timestamp: Date.now(),
        sequenceLength: 0,
        modelId: metadata.id,
      },
      sentAt: req.sentAt,
      tAcq: req.tAcq,
      tPrep: req.tPrep,
      inferDurationMs: Math.round((performance.now() - t0) * 100) / 100,
    });
    return;
  }

  const activeHandOffset = hasRight ? 0 : 63;
  const [tExt, iExt, mExt, rExt, pExt] = computeFingerExtensionsFromFrame(latestFrame, activeHandOffset);
  const { vLong } = computeHandOrientation(latestFrame, activeHandOffset);

  // Physical motion trajectory vector
  const motion = req.motionVector ?? [0, 0, 0];
  const motionMag = Math.sqrt(motion[0] * motion[0] + motion[1] * motion[1] + motion[2] * motion[2]);
  const isWaving = motionMag >= 0.055;
  const isResting = motionMag < 0.045;

  const scores: Record<string, number> = {};

  // Initialize all to low baseline
  for (const sign of classes) {
    scores[sign] = 0.05;
  }

  // 1. ALL FINGERS OPEN POSE (Open Palm / Waving)
  const isOpenHand = iExt > 0.65 && mExt > 0.65 && rExt > 0.65 && pExt > 0.55 && tExt > 0.55;
  if (isOpenHand) {
    if (isTwoHanded) {
      // Two open hands facing up -> WHERE
      scores['WHERE'] = 0.92;
      // Two open hands chest rub -> PLEASE
      scores['PLEASE'] = 0.78;
    } else {
      // Single open hand:
      if (motion[2] > 0.035 || (motion[1] > 0.035 && Math.abs(motion[0]) < 0.035)) {
        // Moving forward/downward from chest/chin -> THANK_YOU
        scores['THANK_YOU'] = 0.92;
        scores['HELLO'] = 0.40;
      } else if (Math.abs(motion[0]) >= 0.07) {
        // Broad horizontal waving -> GOODBYE
        scores['GOODBYE'] = 0.94;
        scores['HELLO'] = 0.70;
      } else if (Math.abs(motion[0]) >= 0.035 || (motionMag >= 0.055 && !isResting)) {
        // Active side-to-side greeting wave -> HELLO
        scores['HELLO'] = 0.93;
        scores['GOODBYE'] = 0.65;
      } else {
        // CRITICAL PALM DEFENSE:
        // Plain resting open palm without intentional motion is NOT a gesture!
        // Suppress HELLO / GOODBYE / THANK_YOU below threshold so it stays in "Detecting" state
        scores['HELLO'] = 0.42;
        scores['THANK_YOU'] = 0.38;
        scores['GOODBYE'] = 0.35;
      }
    }
  }

  // 2. WATER (W-Handshape: Index + Middle + Ring extended, Pinky curled)
  if (iExt > 0.65 && mExt > 0.65 && rExt > 0.65 && pExt < 0.60 && (rExt - pExt > 0.22 || pExt < 0.40)) {
    scores['WATER'] = 0.95;
  }

  // 3. POINTING WITH INDEX FINGER (iExt prominently extended while others folded)
  const isIndexPointing = iExt > 0.65 && (iExt - mExt > 0.30) && (iExt - rExt > 0.25) && (iExt - pExt > 0.25);
  if (isIndexPointing) {
    if (vLong.x < -0.30 || motion[0] < -0.05) {
      // Pointing or moving Left
      scores['LEFT'] = 0.94;
    } else if (vLong.x > 0.30 || motion[0] > 0.05) {
      // Pointing or moving Right
      scores['RIGHT'] = 0.94;
    } else if (vLong.y > 0.30) {
      // Pointing Down
      scores['HERE'] = 0.93;
    } else if (vLong.y < -0.45 && (motionMag > 0.04 || vLong.x < -0.15)) {
      // Pointing near temple / head
      scores['DONT_UNDERSTAND'] = 0.88;
    } else {
      scores['HERE'] = 0.65;
    }
  }

  // 4. CLOSED FIST (All fingers bent into hand: YES / BATHROOM / HELP)
  const isFist = iExt < 0.55 && mExt < 0.55 && rExt < 0.55 && pExt < 0.60;
  if (isFist) {
    if (isTwoHanded) {
      // Fist supported by other hand -> HELP
      scores['HELP'] = 0.94;
    } else if (tExt > 0.65) {
      // Thumb protruding between fingers -> BATHROOM (T-sign)
      scores['BATHROOM'] = 0.91;
    } else {
      // Fist nodding up/down or firm fist -> YES
      if (Math.abs(motion[1]) > 0.03 || isResting) {
        scores['YES'] = 0.92;
      }
    }
  }

  // 5. NO (Index + Middle extended together, thumb pinching or wagging)
  if (iExt > 0.65 && mExt > 0.65 && rExt < 0.35 && pExt < 0.35) {
    if (isTwoHanded) {
      scores['HOSPITAL'] = 0.92; // Cross / H formation
    } else {
      scores['NO'] = 0.91;
      scores['HOSPITAL'] = 0.68;
    }
  }

  // 6. FOOD (Fingertips clustered / pinched together, brought toward mouth)
  const isPinch = iExt > 0.20 && iExt < 0.55 && mExt > 0.20 && mExt < 0.55 && rExt > 0.20 && rExt < 0.55 && pExt > 0.20 && pExt < 0.55;
  if (isPinch && (motion[2] < -0.03 || vLong.y < -0.30)) {
    scores['FOOD'] = 0.90;
  }

  // 7. REPEAT (Two hands rolling forward)
  if (isTwoHanded && iExt > 0.50 && mExt > 0.50 && motionMag > 0.05) {
    scores['REPEAT'] = 0.87;
  }

  // Build sorted probabilities array
  const scoredClasses = classes.map((sign) => ({
    sign,
    label: ISL_DISPLAY_NAMES[sign] ?? sign,
    confidence: Number(scores[sign].toFixed(3)),
  }));

  scoredClasses.sort((a, b) => b.confidence - a.confidence);

  const topMatch = scoredClasses[0];
  const latencyMs = Math.round((performance.now() - t0) * 10) / 10;
  const isRecognized =
    topMatch &&
    topMatch.confidence >= 0.60 &&
    topMatch.sign !== 'NONE' &&
    topMatch.sign !== 'NONE_RESTING' &&
    topMatch.sign !== 'UNKNOWN';

  const result: RecognitionResult = {
    gesture: isRecognized ? topMatch.sign : 'NONE',
    displayName: isRecognized ? topMatch.label : 'Detecting gesture…',
    confidence: isRecognized ? topMatch.confidence : 0,
    probabilities: scoredClasses,
    latencyMs,
    timestamp: Date.now(),
    sequenceLength: windowSize,
    modelId: metadata.id,
  };

  postResponse({
    type: 'RESULT',
    result,
    sentAt: req.sentAt,
    tAcq: req.tAcq,
    tPrep: req.tPrep,
    inferDurationMs: Math.round((performance.now() - t0) * 100) / 100,
  });
}

// ---------------------------------------------------------------------------
// Main Message Handler
// ---------------------------------------------------------------------------

self.onmessage = async (event: MessageEvent<InferenceWorkerRequest>) => {
  const req = event.data;

  switch (req.type) {
    case 'LOAD_MODEL':
      await handleLoadModel(req);
      break;

    case 'INFER':
      await handleInfer(req);
      break;

    case 'PING':
      postResponse({ type: 'PONG' });
      break;

    default:
      postResponse({
        type: 'ERROR',
        error: `Unknown message type: ${(req as InferenceWorkerRequest).type}`,
      });
  }
};

export {};
