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
import { RecognitionResult, RawLandmarkFrame, DEFAULT_PIPELINE_CONFIG } from '@signbridge2vision/types';
import { ISL_VOCABULARY } from '@signbridge2vision/types';
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
const CONFIDENCE_THRESHOLD = 0.60;
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
  get isReady() { return this.frames.length >= 5; }
  get isFull() { return this.frames.length >= this.size; }
  getFrames() { return this.frames; }
  clear() { this.frames = []; }
}

// ── Feature extraction (inline, avoids dynamic import lag) ────────────────
function normHand(lms: Array<{x:number;y:number;z:number}> | null): Float32Array {
  const out = new Float32Array(63);
  if (!lms || lms.length < 21) return out;
  const wx = lms[0].x, wy = lms[0].y, wz = lms[0].z;
  const dx = lms[9].x - wx, dy = lms[9].y - wy, dz = lms[9].z - wz;
  const scale = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1.0;
  for (let i = 0; i < 21; i++) {
    out[i*3]   = (lms[i].x - wx) / scale;
    out[i*3+1] = (lms[i].y - wy) / scale;
    out[i*3+2] = (lms[i].z - wz) / scale;
  }
  return out;
}

function buildMatrix(frames: Array<{ rightHand: Array<{x:number;y:number;z:number}> | null; leftHand: Array<{x:number;y:number;z:number}> | null }>): Float32Array {
  const matrix = new Float32Array(WINDOW_SIZE * FEATURE_DIM);
  const offset = Math.max(0, WINDOW_SIZE - frames.length);
  for (let i = 0; i < frames.length && i < WINDOW_SIZE; i++) {
    const rh = normHand(frames[i].rightHand);
    const lh = normHand(frames[i].leftHand);
    const base = (offset + i) * FEATURE_DIM;
    matrix.set(rh, base);
    matrix.set(lh, base + 63);
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
  const [debugOpen, setDebugOpen]         = useState(false);
  const [probabilities, setProbabilities] = useState<Array<{sign:string;confidence:number;label:string}>>([]);

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
        console.log('[LiveCamera] Model loaded:', msg.modelMetadata);
      } else if (msg.type === 'RESULT') {
        inferenceRunning.current = false;
        const result: RecognitionResult = msg.result;
        setInferenceMs(result.latencyMs);
        setProbabilities(result.probabilities.slice(0, 5));

        // EMA smoothing
        const rawProbs = result.probabilities.map(p => p.confidence);
        const smoothed = emaRef.current.update(rawProbs);

        // Find top after smoothing
        let topIdx = 0;
        for (let i = 1; i < smoothed.length; i++) {
          if (smoothed[i] > smoothed[topIdx]) topIdx = i;
        }
        const topSign = ISL_VOCABULARY[topIdx]?.sign ?? result.gesture;
        const topConf = smoothed[topIdx];

        // Update current result with smoothed confidence
        const smoothedResult: RecognitionResult = {
          ...result,
          gesture: topSign,
          displayName: VOCAB_MAP.get(topSign) ?? topSign,
          confidence: topConf,
          probabilities: Array.from(smoothed).map((c, i) => ({
            sign: ISL_VOCABULARY[i]?.sign ?? `CLASS_${i}`,
            confidence: c,
            label: ISL_VOCABULARY[i]?.label ?? `Class ${i}`,
          })).sort((a, b) => b.confidence - a.confidence),
        };

        setCurrentResult(smoothedResult);

        if (topConf >= CONFIDENCE_THRESHOLD) {
          setPipelineState('RECOGNIZED');

          if (topSign === stableSign.current) {
            stableCount.current++;
          } else {
            stableSign.current = topSign;
            stableCount.current = 1;
          }

          if (stableCount.current >= STABLE_FRAMES) {
            const now = Date.now();
            const isDuplicate = lastEmit.current?.sign === topSign && now - (lastEmit.current?.at ?? 0) < 2500;

            if (!isDuplicate) {
              lastEmit.current = { sign: topSign, at: now };
              setRecognizedAt(now);

              const finalResult: RecognitionResult = { ...smoothedResult };

              setTimeline(prev => [
                { gesture: topSign, displayName: smoothedResult.displayName, confidence: topConf, timestamp: now },
                ...prev.slice(0, 7),
              ]);

              onSignRecognised?.(finalResult);
            }
          }
        } else {
          setPipelineState('LOW_CONFIDENCE');
          stableCount.current = 0;
          stableSign.current = null;
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
  }, [onSignRecognised]);

  // ── Initialise MediaPipe ─────────────────────────────────────────────────
  const initMediaPipe = useCallback(async () => {
    try {
      const { createHandLandmarker } = await import('../../lib/vision/handLandmarker');
      const hl = await createHandLandmarker();
      mpRef.current = hl;
      console.log('[LiveCamera] MediaPipe HandLandmarker initialised successfully');
    } catch (err) {
      console.error('[LiveCamera] MediaPipe init failed:', err);
      setCameraError('MediaPipe failed to initialise. Check browser console.');
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
      const result = mp.detectForVideo(video, now);

      // Extract right/left hand landmarks
      let rh: Array<{x:number;y:number;z:number}> | null = null;
      let lh: Array<{x:number;y:number;z:number}> | null = null;
      let rhConf = 0, lhConf = 0;

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

      const hasHand = rh !== null || lh !== null;
      setHandsDetected(hasHand ? (rh && lh ? 2 : 1) : 0);
      setRightLandmarks(rh);
      setLeftLandmarks(lh);

      // Emit raw frame for recording and external consumers
      onRawFrame?.({
        timestamp: now,
        rightHand: rh,
        leftHand: lh,
        rightHandConfidence: rhConf,
        leftHandConfidence: lhConf,
      });

      if (!hasHand) {
        bufferRef.current.clear();
        emaRef.current.reset();
        stableCount.current = 0;
        stableSign.current = null;
        setPipelineState('IDLE');
        setCurrentResult(null);
      } else {
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
          const matrix = buildMatrix(frames);
          const motionVector = computeMotionVector(frames);
          const tPrep = performance.now() - tPrepStart;
          workerRef.current?.postMessage(
            {
              type: 'INFER',
              features: matrix,
              motionVector,
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
            <span className="text-gray-400 font-mono">{fps} FPS · {handsDetected} hand{handsDetected !== 1 ? 's' : ''}</span>
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
        sequenceFill={bufferFill}
        windowSize={WINDOW_SIZE}
        inferenceLatencyMs={inferenceMs}
        probabilities={probabilities}
        state={pipelineState}
        isOpen={debugOpen}
        onToggle={() => setDebugOpen(o => !o)}
      />

      {/* Action timeline */}
      <ActionTimeline events={timeline} />
    </div>
  );
}
