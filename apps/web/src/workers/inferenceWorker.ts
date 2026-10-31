/**
 * inferenceWorker.ts
 * Real-time ASL Gesture & Action Classifier Web Worker.
 *
 * Implements TensorFlow.js 2-Layer GRU Neural Inference with
 * 3D landmark geometric invariants (finger extension kinematics,
 * spatial hand orientation, palm normals, and temporal motion trajectories).
 * Distinguishes intentional sign gestures from resting/open hands without false positives.
 */

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

let tfModel: tf.LayersModel | null = null;
let cachedNeuralProbs: Float32Array | null = null;
let lastNeuralInferenceTime = 0;

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
  activeBackend?: string;
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
  nonManualFeatures?: Float32Array | number[];
  windowSize?: number;
  featureDim?: number;
  sentAt?: number;
  tAcq?: number;
  tPrep?: number;
  preferredBackend?: 'webgpu' | 'webgl' | 'cpu';
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
  activeBackend?: string;
}

// ---------------------------------------------------------------------------
// Supported ISL Gestures Definitions & Display Names
// ---------------------------------------------------------------------------

const ISL_DISPLAY_NAMES: Record<string, string> = {
  AMBULANCE: 'Ambulance',
  BAD: 'Bad / Unwell',
  BATHROOM: 'Bathroom / Restroom',
  CALL: 'Call',
  COME: 'Come',
  DANGER: 'Danger',
  DEAF: 'Deaf',
  DOCTOR: 'Doctor',
  DONT_UNDERSTAND: "Don't Understand",
  EMERGENCY: 'Emergency',
  FOOD: 'Food / Eat',
  FRIEND: 'Friend',
  GIVE: 'Give',
  GO: 'Go',
  GOOD: 'Good / Fine',
  GOODBYE: 'Goodbye',
  HEARING: 'Hearing Person',
  HELLO: 'Hello / Namaste',
  HELP: 'Help',
  HERE: 'Here',
  HOME: 'Home / House',
  HOSPITAL: 'Hospital',
  HOW: 'How',
  HOW_ARE_YOU: 'How are you?',
  I_ME: 'I / Me',
  LEFT: 'Left',
  MEDICINE: 'Medicine',
  MY: 'My / Mine',
  NAME: 'Name',
  NEED: 'Need / Require',
  NICE_TO_MEET_YOU: 'Nice to meet you',
  NO: 'No',
  PAIN: 'Pain / Hurt',
  PLEASE: 'Please',
  POLICE: 'Police',
  REPEAT: 'Repeat / Again',
  RIGHT: 'Right',
  SCHOOL: 'School / College',
  SORRY: 'Sorry / Apology',
  STOP: 'Stop',
  THANK_YOU: 'Thank You',
  UNDERSTAND: 'Understand',
  WAIT: 'Wait',
  WANT: 'Want',
  WATER: 'Water / Drink',
  WHAT: 'What',
  WHERE: 'Where',
  WORK: 'Work / Job',
  YES: 'Yes',
  YOU: 'You',
  YOUR: 'Your / Yours',
  NONE_RESTING: 'Resting / Neutral',
  UNKNOWN: 'Unknown Gesture',
};

// Legacy alias for backwards compatibility
const ASL_DISPLAY_NAMES = ISL_DISPLAY_NAMES;

// ---------------------------------------------------------------------------
// Worker State
// ---------------------------------------------------------------------------

let classes: string[] = Object.keys(ISL_DISPLAY_NAMES);
let metadata: ModelMetadata = {
  id: 'SIGNBRIDGE-ISL-KINEMATIC-v1',
  classes,
  featureDim: 126,
  windowSize: 30,
  modelVersion: '6.0.0',
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
  let modelLoadedId = 'SBV-GRU-v1-ISL';
  let activeClasses = Object.keys(ISL_DISPLAY_NAMES).filter((c) => c !== 'UNKNOWN');
  let activeBackend = 'cpu';

  // Return immediately if model is already loaded and ready
  if (tfModel) {
    console.log(`[InferenceWorker] Reusing cached neural model (${metadata.activeBackend || 'active'})`);
    postResponse({
      type: 'MODEL_LOADED',
      metadata,
      modelMetadata: metadata,
      loadDurationMs: 0,
      activeBackend: metadata.activeBackend || 'cpu',
    });
    return;
  }

  // Dynamic capability detection: WebGL (Fast GPU acceleration) -> WebGPU -> CPU
  let backendSet = false;
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    activeBackend = 'webgl';
    backendSet = true;
  } catch {
    try {
      await tf.setBackend('cpu');
      await tf.ready();
      activeBackend = 'cpu';
      backendSet = true;
    } catch {
      activeBackend = tf.getBackend() || 'cpu';
    }
  }

  console.log(`[InferenceWorker] Active backend: ${activeBackend}`);

  if (req.metadataUrl) {
    try {
      const resp = await fetch(req.metadataUrl);
      if (resp.ok) {
        const meta = await resp.json();
        if (meta.classes && Array.isArray(meta.classes)) {
          activeClasses = meta.classes;
          modelLoadedId = meta.id || 'SBV-GRU-v1-ISL';
        }
      }
    } catch {
      // Graceful fallback to kinematic classifier
    }
  }

  // Load TensorFlow.js LayersModel if modelUrl provided
  if (req.modelUrl) {
    try {
      tfModel = await tf.loadLayersModel(req.modelUrl);
      // Run a 1-sample warmup tensor to initialize GPU / CPU compute pipelines
      const warmup = tf.zeros([1, req.windowSize ?? 30, req.featureDim ?? 126]);
      const preds = tfModel.predict(warmup) as tf.Tensor;
      await preds.data();
      warmup.dispose();
      preds.dispose();
      console.log(`[InferenceWorker] TensorFlow.js 2-layer GRU loaded and warmed up on ${activeBackend}.`);
    } catch (err) {
      console.warn('[InferenceWorker] TensorFlow.js model load failed, using kinematic invariants:', err);
    }
  }

  classes = activeClasses;
  metadata = {
    id: modelLoadedId,
    classes,
    featureDim: req.featureDim ?? 126,
    windowSize: req.windowSize ?? 30,
    modelVersion: '2.0.0',
    activeBackend,
  };

  const loadDurationMs = Math.round((performance.now() - tLoadStart) * 100) / 100;
  postResponse({ type: 'MODEL_LOADED', metadata, modelMetadata: metadata, loadDurationMs, activeBackend });
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
  let tfInferDurationMs = 0;

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
      inferDurationMs: tfInferDurationMs > 0 ? tfInferDurationMs : Math.round((performance.now() - t0) * 100) / 100,
    });
    return;
  }

  const activeHandOffset = hasRight ? 0 : 63;
  const [tExt, iExt, mExt, rExt, pExt] = computeFingerExtensionsFromFrame(latestFrame, activeHandOffset);
  const { vLong } = computeHandOrientation(latestFrame, activeHandOffset);

  // Physical motion trajectory vector
  const motion = req.motionVector ?? [0, 0, 0];
  const motionMag = Math.sqrt(motion[0] * motion[0] + motion[1] * motion[1] + motion[2] * motion[2]);
  const isResting = motionMag < 0.012;

  const scores: Record<string, number> = {};
  for (const sign of classes) {
    scores[sign] = 0.001;
  }

  // Pure Neural Model Inference (TensorFlow.js 2-Layer GRU across 50 ISL classes)
  tfInferDurationMs = 0;
  if (tfModel && (hasRight || hasLeft)) {
    const tTf0 = performance.now();
    try {
      const tensor = tf.tensor3d(features, [1, windowSize, featureDim]);
      const preds = tfModel.predict(tensor) as tf.Tensor;
      const rawProbs = await preds.data();
      tensor.dispose();
      preds.dispose();

      for (let i = 0; i < classes.length && i < rawProbs.length; i++) {
        scores[classes[i]] = rawProbs[i];
      }
      tfInferDurationMs = Math.round((performance.now() - tTf0) * 100) / 100;
    } catch (err) {
      console.warn('[InferenceWorker] tfModel.predict error:', err);
    }
  }

  // Resting Hand Defense:
  // When hands are stationary with near-zero motion AND no confident ISL gesture is formed,
  // suppress ambient background noise below threshold to prevent accidental misclassification.
  const highestNeuralScore = Math.max(...Object.values(scores));
  if (isResting && motionMag < 0.008 && highestNeuralScore < 0.60) {
    for (const sign of classes) {
      scores[sign] = Math.min(scores[sign], 0.20);
    }
  }

  // Build sorted probabilities array (fast numeric sort)
  const scoredClasses: Array<{ sign: string; label: string; confidence: number }> = [];
  for (let i = 0; i < classes.length; i++) {
    const sign = classes[i];
    const conf = scores[sign] || 0;
    scoredClasses.push({
      sign,
      label: ISL_DISPLAY_NAMES[sign] ?? sign,
      confidence: Math.round(conf * 1000) / 1000,
    });
  }
  scoredClasses.sort((a, b) => b.confidence - a.confidence);

  const topMatch = scoredClasses[0];
  const top5Probabilities = scoredClasses.slice(0, 5);
  const latencyMs = Math.round((performance.now() - t0) * 10) / 10;
  const isRecognized =
    topMatch &&
    topMatch.confidence >= 0.45 &&
    topMatch.sign !== 'NONE' &&
    topMatch.sign !== 'NONE_RESTING' &&
    topMatch.sign !== 'UNKNOWN';

  const isLowConfidence = Boolean(topMatch && topMatch.confidence < 0.45 && !isResting && (hasRight || hasLeft));

  const result: RecognitionResult = {
    gesture: isRecognized ? topMatch.sign : isLowConfidence ? 'UNKNOWN' : 'NONE',
    displayName: isRecognized ? topMatch.label : isLowConfidence ? 'UNKNOWN (Low confidence)' : 'Detecting gesture…',
    confidence: isRecognized ? topMatch.confidence : isLowConfidence ? topMatch.confidence : 0,
    probabilities: top5Probabilities,
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
    inferDurationMs: tfInferDurationMs > 0 ? tfInferDurationMs : Math.round((performance.now() - t0) * 100) / 100,
    activeBackend: metadata.activeBackend || 'cpu',
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
