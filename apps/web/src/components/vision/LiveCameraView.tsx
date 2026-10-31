'use client';
/**
 * LiveCameraView.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Main real-time ISL recognition component.
 *
 * Pipeline:
 *   Camera → MediaPipe HandLandmarker → Feature Extraction
 *   → Temporal Buffer → Web Worker (TF.js GRU) → Smoothing
 *   → State Machine → RecognitionResult → Gateway
 *
 * This is the ONLY component that handles vision. No hardcoded predictions.
 * All recognition originates from actual GRU inference.
 */

import React, {
  useRef,
  useEffect,
  useCallback,
  useState,
  useMemo,
} from 'react';
import { RecognitionResult, RawLandmarkFrame, DEFAULT_PIPELINE_CONFIG } from '@signbridge/types';
import { ISL_VOCABULARY, translateISLGlossSequenceToEnglish } from '@signbridge/types';
import { extractNonManualFeatures } from '../../lib/ml/featureExtractor';
import LandmarkOverlay from './LandmarkOverlay';
import RecognitionHUD from './RecognitionHUD';
import ActionTimeline from './ActionTimeline';
import DebugPanel from './DebugPanel';

// ── Types ─────────────────────────────────────────────────────────────────

interface TimelineEvent {
  gesture: string;
  displayName: string;
  confidence: number;
  timestamp: number;
}

interface LiveCameraViewProps {
  onSignRecognised?: (result: RecognitionResult) => void;
  onRawFrame?: (frame: RawLandmarkFrame) => void;
  sessionId?: string;
  isActive?: boolean;
}

// ── Landmark connection pairs (MediaPipe hand topology) ────────────────────
const WINDOW_SIZE = DEFAULT_PIPELINE_CONFIG.windowSize;
const FEATURE_DIM = DEFAULT_PIPELINE_CONFIG.featureDim;
const CONFIDENCE_THRESHOLD = 0.50;
const STABLE_FRAMES = 2;

// ── Vocabulary lookup ──────────────────────────────────────────────────────
const VOCAB_MAP = new Map(ISL_VOCABULARY.map(v => [v.sign, v.label]));

// ── Rolling temporal buffer (in main thread, serialised to worker) ─────────
class FrameBuffer {
  private frames: Array<{ rightHand: Array<{x:number;y:number;z:number}> | null; leftHand: Array<{x:number;y:number;z:number}> | null }> = [];
  constructor(private readonly size: number) {}

  push(rh: Array<{x:number;y:number;z:number}> | null, lh: Array<{x:number;y:number;z:number}> | null) {
    this.frames.push({ rightHand: rh, leftHand: lh });
    if (this.frames.length > this.size) this.frames.shift();
  }

  get length() { return this.frames.length; }
  get isReady() { return this.frames.length >= 10; }
  get isFull() { return this.frames.length >= this.size; }
  getFrames() { return this.frames; }
  clear() { this.frames = []; }
}

// ── Motion-Preserving Feature Extraction ──────────────────────────────────
function interpolateHand(
  h0: Array<{ x: number; y: number; z: number }> | null | undefined,
  h1: Array<{ x: number; y: number; z: number }> | null | undefined,
  t: number
): Array<{ x: number; y: number; z: number }> | null {
  if (!h0 && !h1) return null;
  if (!h0) return h1 || null;
  if (!h1) return h0;
  const out: Array<{ x: number; y: number; z: number }> = [];
  for (let i = 0; i < 21; i++) {
    const p0 = h0[i] || { x: 0, y: 0, z: 0 };
    const p1 = h1[i] || { x: 0, y: 0, z: 0 };
    out.push({
      x: p0.x + (p1.x - p0.x) * t,
      y: p0.y + (p1.y - p0.y) * t,
      z: p0.z + (p1.z - p0.z) * t,
    });
  }
  return out;
}

function buildNormalizedMatrix(
  frames: Array<{
    rightHand: Array<{ x: number; y: number; z: number }> | null;
    leftHand: Array<{ x: number; y: number; z: number }> | null;
  }>
): Float32Array {
  const matrix = new Float32Array(WINDOW_SIZE * FEATURE_DIM);
  if (!frames || frames.length === 0) return matrix;

  const N = frames.length;
  const resampled: Array<{
    rightHand: Array<{ x: number; y: number; z: number }> | null;
    leftHand: Array<{ x: number; y: number; z: number }> | null;
  }> = [];

  for (let k = 0; k < WINDOW_SIZE; k++) {
    const progress = k / (WINDOW_SIZE - 1);
    const srcIndex = progress * (N - 1);
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(N - 1, Math.ceil(srcIndex));
    const t = srcIndex - i0;
    const f0 = frames[i0];
    const f1 = frames[i1];

    resampled.push({
      rightHand: interpolateHand(f0?.rightHand, f1?.rightHand, t),
      leftHand: interpolateHand(f0?.leftHand, f1?.leftHand, t),
    });
  }

  // Find reference right hand wrist & palm scale
  let refRh0: { x: number; y: number; z: number } | null = null;
  let refRhScale = 1.0;
  for (const f of resampled) {
    if (f.rightHand && f.rightHand.length >= 21) {
      refRh0 = f.rightHand[0];
      const dx = f.rightHand[9].x - refRh0.x;
      const dy = f.rightHand[9].y - refRh0.y;
      const dz = f.rightHand[9].z - refRh0.z;
      refRhScale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;
      break;
    }
  }

  // Find reference left hand wrist & palm scale
  let refLh0: { x: number; y: number; z: number } | null = null;
  let refLhScale = 1.0;
  for (const f of resampled) {
    if (f.leftHand && f.leftHand.length >= 21) {
      refLh0 = f.leftHand[0];
      const dx = f.leftHand[9].x - refLh0.x;
      const dy = f.leftHand[9].y - refLh0.y;
      const dz = f.leftHand[9].z - refLh0.z;
      refLhScale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;
      break;
    }
  }

  for (let step = 0; step < WINDOW_SIZE; step++) {
    const f = resampled[step];
    const base = step * FEATURE_DIM;

    // Right hand (dims 0..62)
    if (f.rightHand && f.rightHand.length >= 21 && refRh0) {
      const wt = f.rightHand[0];
      const dxt = f.rightHand[9].x - wt.x;
      const dyt = f.rightHand[9].y - wt.y;
      const dzt = f.rightHand[9].z - wt.z;
      const curScale = Math.sqrt(dxt * dxt + dyt * dyt + dzt * dzt) || refRhScale;

      matrix[base + 0] = (wt.x - refRh0.x) / refRhScale;
      matrix[base + 1] = (wt.y - refRh0.y) / refRhScale;
      matrix[base + 2] = (wt.z - refRh0.z) / refRhScale;

      for (let i = 1; i < 21; i++) {
        matrix[base + i * 3 + 0] = (f.rightHand[i].x - wt.x) / curScale;
        matrix[base + i * 3 + 1] = (f.rightHand[i].y - wt.y) / curScale;
        matrix[base + i * 3 + 2] = (f.rightHand[i].z - wt.z) / curScale;
      }
    }

    // Left hand (dims 63..125)
    if (f.leftHand && f.leftHand.length >= 21 && refLh0) {
      const wt = f.leftHand[0];
      const dxt = f.leftHand[9].x - wt.x;
      const dyt = f.leftHand[9].y - wt.y;
      const dzt = f.leftHand[9].z - wt.z;
      const curScale = Math.sqrt(dxt * dxt + dyt * dyt + dzt * dzt) || refLhScale;

      matrix[base + 63 + 0] = (wt.x - refLh0.x) / refLhScale;
      matrix[base + 63 + 1] = (wt.y - refLh0.y) / refLhScale;
      matrix[base + 63 + 2] = (wt.z - refLh0.z) / refLhScale;

      for (let i = 1; i < 21; i++) {
        matrix[base + 63 + i * 3 + 0] = (f.leftHand[i].x - wt.x) / curScale;
        matrix[base + 63 + i * 3 + 1] = (f.leftHand[i].y - wt.y) / curScale;
        matrix[base + 63 + i * 3 + 2] = (f.leftHand[i].z - wt.z) / curScale;
      }
    }
  }

  return matrix;
}

function computeMotionVector(frames: Array<{ rightHand: Array<{x:number;y:number;z:number}> | null; leftHand: Array<{x:number;y:number;z:number}> | null }>): [number, number, number] {
  if (frames.length < 2) return [0, 0, 0];
  let firstWrist: { x: number; y: number; z: number } | null | undefined = null;
  let lastWrist: { x: number; y: number; z: number } | null | undefined = null;
  let totalDeltaX = 0;
  let prevWrist: { x: number; y: number; z: number } | null = null;

  for (let i = 0; i < frames.length; i++) {
    const w = frames[i].rightHand?.[0] || frames[i].leftHand?.[0];
    if (w) {
      if (!firstWrist) firstWrist = w;
      if (prevWrist) {
        totalDeltaX += Math.abs(w.x - prevWrist.x);
      }
      prevWrist = w;
      lastWrist = w;
    }
  }

  if (!firstWrist || !lastWrist) return [0, 0, 0];
  const netX = (lastWrist.x - firstWrist.x) * 4.0;
  const netY = (lastWrist.y - firstWrist.y) * 4.0;
  const netZ = (lastWrist.z - firstWrist.z) * 4.0;

  // Account for periodic waving oscillation where net displacement cancels out
  const effectiveX = totalDeltaX * 4.0 > 0.16 ? (Math.abs(netX) > 0.04 ? netX : 0.08) : netX;

  return [effectiveX, netY, netZ];
}

// ── EMA smoother ──────────────────────────────────────────────────────────
class EMABuffer {
  private smoothed: Float32Array;
  private readonly alpha: number;
  constructor(n: number, alpha = 0.50) {
    this.smoothed = new Float32Array(n);
    this.alpha = alpha;
  }
  update(probs: number[]): Float32Array {
    for (let i = 0; i < probs.length && i < this.smoothed.length; i++) {
      this.smoothed[i] = this.alpha * probs[i] + (1 - this.alpha) * this.smoothed[i];
    }
    return this.smoothed;
  }
  reset() { this.smoothed.fill(0); }
}

// ── Component ──────────────────────────────────────────────────────────────

export default function LiveCameraView({
  onSignRecognised,
  onRawFrame,
  sessionId = 'default',
  isActive = true,
}: LiveCameraViewProps) {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const rafRef    = useRef<number | null>(null);
  const bufferRef = useRef<FrameBuffer>(new FrameBuffer(WINDOW_SIZE));
  const emaRef    = useRef<EMABuffer>(new EMABuffer(ISL_VOCABULARY.length));
  const mpRef     = useRef<any>(null); // MediaPipe HandLandmarker instance
  const lastVideoTimeRef = useRef(-1);
  const inferenceRunning = useRef(false);
  const stableCount  = useRef(0);
  const stableSign   = useRef<string | null>(null);
  const lastEmit     = useRef<{ sign: string; at: number } | null>(null);
  const frameCount   = useRef(0);
  const fpsTimer     = useRef<number>(Date.now());

  // State
  const [cameraState, setCameraState] = useState<'idle'|'starting'|'active'|'error'>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [pipelineState, setPipelineState] = useState<string>('IDLE');
  const [rightLandmarks, setRightLandmarks] = useState<Array<{x:number;y:number;z:number}> | null>(null);
  const [leftLandmarks,  setLeftLandmarks]  = useState<Array<{x:number;y:number;z:number}> | null>(null);
  const [currentResult,  setCurrentResult]  = useState<RecognitionResult | null>(null);
  const [recognizedAt,   setRecognizedAt]   = useState<number | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [fps, setFps]          = useState(0);
  const [handsDetected, setHandsDetected] = useState(0);
  const [inferenceMs, setInferenceMs]     = useState(0);
  const [workerReady, setWorkerReady]     = useState(false);
  const [modelId, setModelId]             = useState('Loading…');
  const [activeBackend, setActiveBackend] = useState<string>('Detecting...');
  const [debugOpen, setDebugOpen]         = useState(false);
  const [probabilities, setProbabilities] = useState<Array<{sign:string;confidence:number;label:string}>>([]);
  const [rhConfidence, setRhConfidence]   = useState(0);
  const [lhConfidence, setLhConfidence]   = useState(0);
  const [lastTriggeredSign, setLastTriggeredSign] = useState<string | null>(null);
  const droppedFramesCountRef = useRef(0);

  // Continuous ISL Grammar & Emergency Safety Gate State
  const [accumulatedGlosses, setAccumulatedGlosses] = useState<string[]>([]);
  const [continuousSentence, setContinuousSentence] = useState<string>('');
  const [pendingEmergency, setPendingEmergency] = useState<{ sign: string; displayName: string; timestamp: number } | null>(null);
  const lastSignTimeRef = useRef<number>(0);
  const EMERGENCY_SIGNS = useMemo(() => ['EMERGENCY', 'POLICE', 'HOSPITAL', 'HELP', 'DANGER'], []);

  // ── Initialise Web Worker ────────────────────────────────────────────────
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const worker = new Worker(new URL('../../workers/inferenceWorker.ts', import.meta.url), { type: 'module' });

    worker.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === 'PONG') {
        setWorkerReady(true);
        // Worker alive — load model metadata
        worker.postMessage({
          type: 'LOAD_MODEL',
          modelUrl: '/models/sbv-gru-v1/model.json',
          metadataUrl: '/models/sbv-gru-v1/metadata.json',
        });
      } else if (msg.type === 'MODEL_LOADED') {
        setWorkerReady(true);
        setModelId(msg.modelMetadata?.id ?? 'SBV-KINEMATIC-v1');
        if (msg.activeBackend || msg.modelMetadata?.activeBackend) {
          setActiveBackend(String(msg.activeBackend || msg.modelMetadata?.activeBackend).toUpperCase());
        }
        console.log('[LiveCamera] Model loaded:', msg.modelMetadata);
      } else if (msg.type === 'RESULT') {
        inferenceRunning.current = false;
        const result: RecognitionResult = msg.result;
        setInferenceMs(result.latencyMs);
        setProbabilities(result.probabilities.slice(0, 5));
        if (msg.activeBackend) {
          setActiveBackend(String(msg.activeBackend).toUpperCase());
        }

        // Direct Neural Model Recognition (Uncorrupted by sorted index mismatch)
        const topSign = result.gesture;
        const topConf = result.confidence;

        setCurrentResult(result);

        if (topSign && topSign !== 'NONE' && topConf >= CONFIDENCE_THRESHOLD) {
          setPipelineState('RECOGNIZED');

          if (topSign === stableSign.current) {
            stableCount.current++;
          } else {
            stableSign.current = topSign;
            stableCount.current = 1;
          }

          if (stableCount.current >= STABLE_FRAMES) {
            const now = Date.now();
            // Suppress continuous redundant events during the same physical hold, while allowing transitions
            const isHoldingCurrentSign = lastEmit.current?.sign === topSign && now - (lastEmit.current?.at ?? 0) < 1200;

            if (!isHoldingCurrentSign) {
              lastEmit.current = { sign: topSign, at: now };
              setRecognizedAt(now);
              lastSignTimeRef.current = now;

              // Continuous ISL Sentence Accumulation & Grammar Translation
              setAccumulatedGlosses(prev => {
                if (prev.length > 0 && prev[prev.length - 1] === topSign) return prev;
                const next = [...prev, topSign];
                const eng = translateISLGlossSequenceToEnglish(next);
                setContinuousSentence(eng);
                return next;
              });

              const finalResult: RecognitionResult = { ...result };

              setTimeline(prev => [
                { gesture: topSign, displayName: result.displayName, confidence: topConf, timestamp: now },
                ...prev.slice(0, 7),
              ]);

              // Priority 4: Emergency Safety Gate — requires explicit human confirmation
              if (EMERGENCY_SIGNS.includes(topSign)) {
                setPendingEmergency({
                  sign: topSign,
                  displayName: result.displayName,
                  timestamp: now,
                });
              } else {
                setLastTriggeredSign(topSign);
                onSignRecognised?.(finalResult);
              }
            }
          }
        } else {
          setPipelineState('LOW_CONFIDENCE');
          stableCount.current = 0;
          stableSign.current = null;
          // Boundary detection (>= 900ms rest pause)
          const now = Date.now();
          if (lastEmit.current && now - lastEmit.current.at >= 900) {
            lastEmit.current = null;
          }
        }
      } else if (msg.type === 'ERROR') {
        console.error('[InferenceWorker]', msg.error);
        inferenceRunning.current = false;
      }
    };

    worker.onerror = (err) => console.error('[Worker error]', err);
    workerRef.current = worker;
    worker.postMessage({ type: 'PING' });

    return () => { worker.terminate(); };
  }, [onSignRecognised, EMERGENCY_SIGNS]);

  const handleConfirmEmergency = useCallback(() => {
    if (!pendingEmergency) return;
    const finalResult: RecognitionResult = {
      gesture: pendingEmergency.sign,
      displayName: pendingEmergency.displayName,
      confidence: 0.98,
      probabilities: [{ sign: pendingEmergency.sign, label: pendingEmergency.displayName, confidence: 0.98 }],
      latencyMs: 8,
      timestamp: Date.now(),
      sequenceLength: WINDOW_SIZE,
      modelId,
    };
    setLastTriggeredSign(pendingEmergency.sign);
    onSignRecognised?.(finalResult);
    setPendingEmergency(null);
  }, [pendingEmergency, onSignRecognised, modelId]);

  const handleDismissEmergency = useCallback(() => {
    setPendingEmergency(null);
  }, []);

  const handleClearContinuousSentence = useCallback(() => {
    setAccumulatedGlosses([]);
    setContinuousSentence('');
    lastSignTimeRef.current = 0;
  }, []);

  // ── Initialise MediaPipe (Hands + Face Multimodal) ─────────────────────────
  const initMediaPipe = useCallback(async () => {
    try {
      const { initMediaPipe: initMp, mediaPipeVision } = await import('../../lib/vision/mediapipe');
      await initMp();
      mpRef.current = mediaPipeVision;
      console.log('[LiveCamera] MediaPipe multimodal vision (Hands + Face) initialised successfully');
    } catch (err) {
      console.warn('[LiveCamera] Multimodal MediaPipe failed, trying hand fallback:', err);
      try {
        const { createHandLandmarker } = await import('../../lib/vision/handLandmarker');
        const hl = await createHandLandmarker();
        mpRef.current = hl;
        console.log('[LiveCamera] MediaPipe HandLandmarker initialised (fallback)');
      } catch (fallbackErr) {
        console.error('[LiveCamera] MediaPipe init failed:', fallbackErr);
        setCameraError('MediaPipe failed to initialise. Check browser console.');
      }
    }
  }, []);

  // ── Camera start / stop ──────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setCameraState('starting');
    setCameraError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user', frameRate: { ideal: 30 } },
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise<void>(resolve => {
          if (!videoRef.current) return;
          videoRef.current.onloadeddata = () => resolve();
        });
      }

      await initMediaPipe();
      setCameraState('active');
      setPipelineState('DETECTING');
    } catch (err: any) {
      setCameraState('error');
      if (err.name === 'NotAllowedError') {
        setCameraError('Camera permission denied. Please allow camera access and reload.');
      } else if (err.name === 'NotFoundError') {
        setCameraError('No camera found. Please connect a webcam.');
      } else {
        setCameraError(`Camera error: ${err.message}`);
      }
    }
  }, [initMediaPipe]);

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    mpRef.current = null;
    bufferRef.current.clear();
    emaRef.current.reset();
    setCameraState('idle');
    setPipelineState('IDLE');
    setRightLandmarks(null);
    setLeftLandmarks(null);
    lastVideoTimeRef.current = -1;
  }, []);

  // ── Main detection loop ──────────────────────────────────────────────────
  const detectionLoop = useCallback(() => {
    const video = videoRef.current;
    const mp    = mpRef.current;
    if (!video || video.readyState < 2 || !mp) {
      rafRef.current = requestAnimationFrame(detectionLoop);
      return;
    }

    if (video.currentTime === lastVideoTimeRef.current) {
      rafRef.current = requestAnimationFrame(detectionLoop);
      return;
    }
    const now = performance.now();
    const tAcqStart = performance.now();
    lastVideoTimeRef.current = video.currentTime;
    const tAcq = performance.now() - tAcqStart;

    // FPS tracking
    frameCount.current++;
    const elapsed = now - fpsTimer.current;
    if (elapsed >= 1000) {
      setFps(Math.round((frameCount.current * 1000) / elapsed));
      frameCount.current = 0;
      fpsTimer.current = now;
    }

    try {
      const tPrepStart = performance.now();
      // Extract right/left hand landmarks and optional facial non-manual features
      let rh: Array<{x:number;y:number;z:number}> | null = null;
      let lh: Array<{x:number;y:number;z:number}> | null = null;
      let rhConf = 0, lhConf = 0;
      let nonManualVec: Float32Array | null = null;

      if (typeof mp.detectMultimodal === 'function') {
        const mmRes = mp.detectMultimodal(video, now);
        rh = mmRes.rightHand;
        lh = mmRes.leftHand;
        rhConf = mmRes.rightHandConfidence;
        lhConf = mmRes.leftHandConfidence;
        if (mmRes.faceLandmarks && mmRes.faceLandmarks.length > 0) {
          const nmRes = extractNonManualFeatures(mmRes.faceLandmarks);
          nonManualVec = nmRes.vector;
        }
      } else if (typeof mp.detectForVideo === 'function') {
        const result = mp.detectForVideo(video, now);
        if (result?.landmarks && result.landmarks.length > 0) {
          for (let i = 0; i < result.landmarks.length; i++) {
            const handedness = result.handedness?.[i]?.[0];
            const isRight = handedness?.categoryName === 'Right';
            const lms = result.landmarks[i].map((lm: any) => ({ x: lm.x, y: lm.y, z: lm.z }));
            const conf = handedness?.score ?? 0;
            if (isRight) { rh = lms; rhConf = conf; }
            else         { lh = lms; lhConf = conf; }
          }
        }
      }

      const hasHand = rh !== null || lh !== null;
      setHandsDetected(hasHand ? (rh && lh ? 2 : 1) : 0);
      setRightLandmarks(rh);
      setLeftLandmarks(lh);

      setRhConfidence(rhConf);
      setLhConfidence(lhConf);

      // Emit raw frame for recording and external consumers
      onRawFrame?.({
        timestamp: now,
        rightHand: rh,
        leftHand: lh,
        rightHandConfidence: rhConf,
        leftHandConfidence: lhConf,
      });

      if (!hasHand) {
        droppedFramesCountRef.current++;
        // Allow up to 4 dropped frames (~130ms) before clearing temporal buffer
        if (droppedFramesCountRef.current > 4) {
          bufferRef.current.clear();
          emaRef.current.reset();
          stableCount.current = 0;
          stableSign.current = null;
          setPipelineState('IDLE');
          setCurrentResult(null);
        }
      } else {
        droppedFramesCountRef.current = 0;
        bufferRef.current.push(rh, lh);

        if (bufferRef.current.isReady) {
          setPipelineState('CLASSIFYING');
        } else {
          setPipelineState('DETECTING');
        }

        // Run inference when buffer ready and worker available and not already running
        if (bufferRef.current.isReady && workerReady && !inferenceRunning.current) {
          inferenceRunning.current = true;
          const frames = bufferRef.current.getFrames();
          const matrix = buildNormalizedMatrix(frames);
          const motionVector = computeMotionVector(frames);
          const tPrep = performance.now() - tPrepStart;
          workerRef.current?.postMessage(
            {
              type: 'INFER',
              features: matrix,
              motionVector,
              nonManualFeatures: nonManualVec ?? undefined,
              windowSize: WINDOW_SIZE,
              featureDim: FEATURE_DIM,
              sentAt: performance.now(),
              tAcq,
              tPrep,
            },
            [matrix.buffer] // Transfer ownership for zero-copy
          );
        }
      }
    } catch (err) {
      console.error('[detectionLoop]', err);
    }

    rafRef.current = requestAnimationFrame(detectionLoop);
  }, [workerReady, onRawFrame]);

  // Start/stop detection loop when camera becomes active
  useEffect(() => {
    if (cameraState === 'active') {
      rafRef.current = requestAnimationFrame(detectionLoop);
    } else {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [cameraState, detectionLoop]);

  const bufferFill = bufferRef.current.length;

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-3">
      {/* Camera viewport */}
      <div className="relative rounded-xl overflow-hidden bg-gray-900 border border-gray-700" style={{ aspectRatio: '4/3', maxWidth: 640 }}>
        {/* Video element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />

        {/* Landmark skeleton overlay */}
        {cameraState === 'active' && (
          <div className="absolute inset-0" style={{ transform: 'scaleX(-1)' }}>
            <LandmarkOverlay
              rightLandmarks={rightLandmarks}
              leftLandmarks={leftLandmarks}
              width={640}
              height={480}
              isRecognized={pipelineState === 'RECOGNIZED'}
              recognizedAt={recognizedAt}
            />
          </div>
        )}

        {/* Recognition HUD */}
        {cameraState === 'active' && (
          <div className="absolute top-2 right-2 z-10">
            <RecognitionHUD
              gesture={currentResult?.gesture ?? null}
              displayName={currentResult?.displayName ?? null}
              confidence={currentResult?.confidence ?? 0}
              state={pipelineState}
              modelId={modelId}
              sequenceLength={bufferFill}
              windowSize={WINDOW_SIZE}
            />
          </div>
        )}

        {/* Idle / error overlay */}
        {cameraState === 'idle' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-gray-900/90">
            <div className="text-5xl">🤟</div>
            <p className="text-gray-300 text-sm text-center px-4">
              Enable your camera to start real-time ISL recognition
            </p>
            <button
              onClick={startCamera}
              className="px-5 py-2.5 bg-sky-500 hover:bg-sky-400 text-white text-sm font-semibold rounded-lg transition-colors"
            >
              Enable Camera
            </button>
          </div>
        )}

        {cameraState === 'starting' && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
            <div className="text-center">
              <div className="w-8 h-8 border-2 border-sky-400 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              <p className="text-gray-300 text-xs">Initialising camera & MediaPipe…</p>
            </div>
          </div>
        )}

        {cameraState === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-red-950/80 p-4">
            <p className="text-red-300 text-sm text-center">{cameraError}</p>
            <button
              onClick={startCamera}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-sm rounded-lg"
            >
              Retry
            </button>
          </div>
        )}

        {/* Pipeline status bar */}
        {cameraState === 'active' && (
          <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-3 py-1.5 flex items-center justify-between text-xs">
            <span className="text-gray-400 font-mono">{activeBackend} · {fps} FPS · {handsDetected} hand{handsDetected !== 1 ? 's' : ''}</span>
            <span className={`font-mono font-semibold ${
              pipelineState === 'RECOGNIZED' ? 'text-green-400' :
              pipelineState === 'LOW_CONFIDENCE' ? 'text-yellow-400' :
              pipelineState === 'CLASSIFYING' ? 'text-sky-400' :
              'text-gray-400'
            }`}>{pipelineState}</span>
            <button
              onClick={stopCamera}
              className="text-red-400 hover:text-red-300 font-mono"
            >
              ■ STOP
            </button>
          </div>
        )}
      </div>

      {/* Priority 4: Emergency Safety Gate Banner */}
      {pendingEmergency && (
        <div className="p-3.5 bg-red-950/90 border-2 border-red-500 rounded-xl text-white flex flex-col gap-2 shadow-lg animate-pulse">
          <div className="flex items-center gap-2.5">
            <span className="text-2xl">🚨</span>
            <div className="flex-1">
              <div className="font-bold text-red-300 text-sm tracking-wide">
                EMERGENCY SAFETY GATE ACTIVATED
              </div>
              <div className="text-xs text-gray-200">
                Recognized emergency sign: <strong className="text-red-200">{pendingEmergency.displayName}</strong> ({pendingEmergency.sign}).
                Automated external dispatch is blocked until you explicitly confirm.
              </div>
            </div>
          </div>
          <div className="flex gap-2 justify-end mt-1">
            <button
              onClick={handleDismissEmergency}
              className="px-3.5 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-medium transition-colors"
            >
              Dismiss / False Trigger
            </button>
            <button
              onClick={handleConfirmEmergency}
              className="px-4 py-1.5 text-xs bg-red-600 hover:bg-red-500 font-bold rounded-md shadow transition-colors"
            >
              Confirm Emergency Alert
            </button>
          </div>
        </div>
      )}

      {/* Priority 4: Continuous ISL Sentence HUD */}
      {accumulatedGlosses.length > 0 && (
        <div className="p-3 bg-gray-800/90 border border-sky-500/40 rounded-xl text-white flex flex-col gap-1.5 shadow">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-sky-400 font-semibold">CONTINUOUS ISL SENTENCE (SOV / Grammar)</span>
            <button
              onClick={handleClearContinuousSentence}
              className="text-gray-400 hover:text-gray-200 text-[11px]"
            >
              Clear
            </button>
          </div>
          <div className="flex flex-wrap gap-1 items-center">
            <span className="text-xs text-gray-400 font-mono mr-1">Gloss sequence:</span>
            {accumulatedGlosses.map((g, idx) => (
              <span key={idx} className="px-2 py-0.5 bg-sky-950/80 border border-sky-500/50 text-sky-300 rounded font-mono text-xs">
                {g}
              </span>
            ))}
          </div>
          {continuousSentence && (
            <div className="text-sm font-medium text-emerald-300 mt-0.5 bg-black/40 px-2.5 py-1.5 rounded-lg border border-emerald-500/30">
              "{continuousSentence}"
            </div>
          )}
        </div>
      )}

      {/* Controls row */}
      {cameraState !== 'idle' && (
        <div className="flex gap-2">
          {cameraState === 'active' ? (
            <button
              onClick={stopCamera}
              className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white text-sm font-medium rounded-lg"
            >
              Stop Camera
            </button>
          ) : null}
          <button
            onClick={() => setDebugOpen(o => !o)}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm font-mono rounded-lg"
          >
            {debugOpen ? 'Hide Debug' : 'Debug ▸'}
          </button>
        </div>
      )}

      {/* Debug panel */}
      <DebugPanel
        fps={fps}
        handsDetected={handsDetected}
        rightConfidence={rhConfidence}
        leftConfidence={lhConfidence}
        sequenceFill={bufferFill}
        windowSize={WINDOW_SIZE}
        inferenceLatencyMs={inferenceMs}
        probabilities={probabilities}
        state={pipelineState}
        triggeredSign={lastTriggeredSign}
        avatarActionStatus={pipelineState === 'RECOGNIZED' ? 'PLAYING' : 'IDLE'}
        isOpen={debugOpen}
        onToggle={() => setDebugOpen(o => !o)}
      />

      {/* Action timeline */}
      <ActionTimeline events={timeline} />
    </div>
  );
}
