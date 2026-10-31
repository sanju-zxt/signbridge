'use client';
/**
 * OptimizedCameraView.tsx
 * ---------------------------------------------------------------------------
 * High-performance, low-latency Camera & Landmark Perception Viewport for SignBridge.
 *
 * Implements:
 *   1. Zero-React-render RAF loop for landmark extraction & direct 2D canvas rendering.
 *   2. Reused singleton MediaPipe HandLandmarker.
 *   3. Non-blocking Web Worker inference with zero backlog (drops stale frames).
 *   4. Adaptive 5-State Temporal Confirmation Machine:
 *        HAND_DETECTED -> ANALYZING -> CANDIDATE -> CONFIRMED -> UNKNOWN
 *   5. Hold-pose duplicate suppression & neutral-transition re-triggering.
 *   6. Rest-pose suppression preventing false positives while stationary.
 *   7. Developer Performance HUD displaying actual measured timings:
 *        Camera FPS, Landmark Extraction (ms), Model Inference (ms),
 *        Sign Confirmation (ms), Total Latency (ms), Dropped Frames.
 *   8. Fullscreen / Expand controls, landmark toggles, and uncropped object-contain viewport.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  Camera,
  CameraOff,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  Zap,
  Activity,
  Hand,
  CheckCircle2,
  AlertCircle,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import { RecognitionResult } from '@signbridge/types';
import { createHandLandmarker, HandLandmarkerInstance } from '../../lib/vision/handLandmarker';

export type RecognitionState =
  | 'IDLE'
  | 'HAND_DETECTED'
  | 'ANALYZING'
  | 'CANDIDATE'
  | 'CONFIRMED'
  | 'UNKNOWN';

export interface PerformanceMetrics {
  fps: number;
  landmarkMs: number;
  inferMs: number;
  confirmMs: number;
  latencyMs: number;
  droppedFrames: number;
}

interface OptimizedCameraViewProps {
  isCameraOn: boolean;
  onToggleCamera: () => void;
  confidenceThreshold?: number;
  onSignConfirmed: (result: RecognitionResult) => void;
  onCandidateChanged?: (candidate: { sign: string; label: string; confidence: number } | null) => void;
  onRecognitionStateChanged?: (state: RecognitionState) => void;
  onTopProbabilitiesChanged?: (probs: Array<{ sign: string; label: string; confidence: number }>) => void;
  isSigningPaused?: boolean;
  onToggleSigningPaused?: () => void;
  className?: string;
}

// MediaPipe hand connection pairs (21 landmarks)
const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// Linear interpolation for normalized temporal resample
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

// High-performance zero-allocation motion-preserving 30-frame matrix normalization
function buildNormalizedMatrix(
  frames: Array<{
    rightHand: Array<{ x: number; y: number; z: number }> | null;
    leftHand: Array<{ x: number; y: number; z: number }> | null;
  }>,
  winSize = 30
): Float32Array {
  const matrix = new Float32Array(winSize * 126);
  if (!frames || frames.length === 0) return matrix;

  const N = frames.length;

  // Find initial reference right hand wrist & palm scale directly from earliest frame
  let refRh0: { x: number; y: number; z: number } | null = null;
  let refRhScale = 1.0;
  for (let i = 0; i < N; i++) {
    const rh = frames[i]?.rightHand;
    if (rh && rh.length >= 21) {
      refRh0 = rh[0];
      const dx = rh[9].x - refRh0.x;
      const dy = rh[9].y - refRh0.y;
      const dz = rh[9].z - refRh0.z;
      refRhScale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;
      break;
    }
  }

  // Find initial reference left hand wrist & palm scale directly from earliest frame
  let refLh0: { x: number; y: number; z: number } | null = null;
  let refLhScale = 1.0;
  for (let i = 0; i < N; i++) {
    const lh = frames[i]?.leftHand;
    if (lh && lh.length >= 21) {
      refLh0 = lh[0];
      const dx = lh[9].x - refLh0.x;
      const dy = lh[9].y - refLh0.y;
      const dz = lh[9].z - refLh0.z;
      refLhScale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;
      break;
    }
  }

  for (let k = 0; k < winSize; k++) {
    const progress = winSize > 1 ? k / (winSize - 1) : 0;
    const srcIndex = progress * (N - 1);
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(N - 1, Math.ceil(srcIndex));
    const t = srcIndex - i0;
    const f0 = frames[i0];
    const f1 = frames[i1];
    const base = k * 126;

    // Right hand (dims 0..62)
    const rh0 = f0?.rightHand;
    const rh1 = f1?.rightHand || rh0;
    if (rh0 && rh0.length >= 21 && refRh0) {
      const h1 = rh1 && rh1.length >= 21 ? rh1 : rh0;
      // Instantaneous interpolated wrist
      const wt_x = rh0[0].x + (h1[0].x - rh0[0].x) * t;
      const wt_y = rh0[0].y + (h1[0].y - rh0[0].y) * t;
      const wt_z = rh0[0].z + (h1[0].z - rh0[0].z) * t;

      // Instantaneous interpolated MCP9
      const m9_x = rh0[9].x + (h1[9].x - rh0[9].x) * t;
      const m9_y = rh0[9].y + (h1[9].y - rh0[9].y) * t;
      const m9_z = rh0[9].z + (h1[9].z - rh0[9].z) * t;

      const dxt = m9_x - wt_x;
      const dyt = m9_y - wt_y;
      const dzt = m9_z - wt_z;
      const curScale = Math.sqrt(dxt * dxt + dyt * dyt + dzt * dzt) || refRhScale;

      // Joint 0: Wrist motion displacement relative to sequence initial frame
      matrix[base + 0] = (wt_x - refRh0.x) / refRhScale;
      matrix[base + 1] = (wt_y - refRh0.y) / refRhScale;
      matrix[base + 2] = (wt_z - refRh0.z) / refRhScale;

      // Joints 1..20: Skeletal configuration relative to instantaneous wrist
      for (let i = 1; i < 21; i++) {
        const px = rh0[i].x + (h1[i].x - rh0[i].x) * t;
        const py = rh0[i].y + (h1[i].y - rh0[i].y) * t;
        const pz = rh0[i].z + (h1[i].z - rh0[i].z) * t;
        matrix[base + i * 3 + 0] = (px - wt_x) / curScale;
        matrix[base + i * 3 + 1] = (py - wt_y) / curScale;
        matrix[base + i * 3 + 2] = (pz - wt_z) / curScale;
      }
    }

    // Left hand (dims 63..125)
    const lh0 = f0?.leftHand;
    const lh1 = f1?.leftHand || lh0;
    if (lh0 && lh0.length >= 21 && refLh0) {
      const h1 = lh1 && lh1.length >= 21 ? lh1 : lh0;
      const wt_x = lh0[0].x + (h1[0].x - lh0[0].x) * t;
      const wt_y = lh0[0].y + (h1[0].y - lh0[0].y) * t;
      const wt_z = lh0[0].z + (h1[0].z - lh0[0].z) * t;

      const m9_x = lh0[9].x + (h1[9].x - lh0[9].x) * t;
      const m9_y = lh0[9].y + (h1[9].y - lh0[9].y) * t;
      const m9_z = lh0[9].z + (h1[9].z - lh0[9].z) * t;

      const dxt = m9_x - wt_x;
      const dyt = m9_y - wt_y;
      const dzt = m9_z - wt_z;
      const curScale = Math.sqrt(dxt * dxt + dyt * dyt + dzt * dzt) || refLhScale;

      matrix[base + 63 + 0] = (wt_x - refLh0.x) / refLhScale;
      matrix[base + 63 + 1] = (wt_y - refLh0.y) / refLhScale;
      matrix[base + 63 + 2] = (wt_z - refLh0.z) / refLhScale;

      for (let i = 1; i < 21; i++) {
        const px = lh0[i].x + (h1[i].x - lh0[i].x) * t;
        const py = lh0[i].y + (h1[i].y - lh0[i].y) * t;
        const pz = lh0[i].z + (h1[i].z - lh0[i].z) * t;
        matrix[base + 63 + i * 3 + 0] = (px - wt_x) / curScale;
        matrix[base + 63 + i * 3 + 1] = (py - wt_y) / curScale;
        matrix[base + 63 + i * 3 + 2] = (pz - wt_z) / curScale;
      }
    }
  }

  return matrix;
}

function computeMotionVector(
  frames: Array<{
    rightHand: Array<{ x: number; y: number; z: number }> | null;
    leftHand: Array<{ x: number; y: number; z: number }> | null;
  }>
): [number, number, number] {
  if (frames.length < 2) return [0, 0, 0];
  let firstWrist: { x: number; y: number; z: number } | null = null;
  let lastWrist: { x: number; y: number; z: number } | null = null;
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
  const effectiveX = totalDeltaX * 4.0 > 0.16 ? (Math.abs(netX) > 0.04 ? netX : 0.08) : netX;
  return [effectiveX, netY, netZ];
}

export const OptimizedCameraView: React.FC<OptimizedCameraViewProps> = ({
  isCameraOn,
  onToggleCamera,
  confidenceThreshold = 0.48,
  onSignConfirmed,
  onCandidateChanged,
  onRecognitionStateChanged,
  onTopProbabilitiesChanged,
  isSigningPaused = false,
  onToggleSigningPaused,
  className = '',
}) => {
  // DOM Refs
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const mpRef = useRef<HandLandmarkerInstance | null>(null);

  // UI Viewport Controls
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [showPerfHUD, setShowPerfHUD] = useState<boolean>(true);
  const [cameraStatus, setCameraStatus] = useState<string>('Initializing...');
  const [handsCount, setHandsCount] = useState<number>(0);
  const [activeRecognitionState, setActiveRecognitionState] = useState<RecognitionState>('IDLE');
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Performance Telemetry (Throttled to 4Hz for smooth, non-jank UI updates)
  const [perfMetrics, setPerfMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    landmarkMs: 0,
    inferMs: 0,
    confirmMs: 0,
    latencyMs: 0,
    droppedFrames: 0,
  });

  // Mutable internal state (Zero React re-render overhead per frame)
  const frameCountRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());
  const droppedFramesRef = useRef<number>(0);
  const lastLandmarkMsRef = useRef<number>(0);
  const lastInferMsRef = useRef<number>(0);
  const lastConfirmMsRef = useRef<number>(0);
  const lastLatencyMsRef = useRef<number>(0);

  // Sliding Temporal Buffer
  const bufferRef = useRef<Array<{
    rightHand: Array<{ x: number; y: number; z: number }> | null;
    leftHand: Array<{ x: number; y: number; z: number }> | null;
  }>>([]);
  const inferenceInFlightRef = useRef<boolean>(false);

  // Adaptive Confirmation State
  const candidateSignRef = useRef<string | null>(null);
  const candidateCountRef = useRef<number>(0);
  const candidateStartAtRef = useRef<number>(0);
  const heldSignRef = useRef<{ sign: string; confirmedAt: number } | null>(null);
  const neutralFramesCountRef = useRef<number>(0);
  const latestRecognitionStateRef = useRef<RecognitionState>('IDLE');

  // Callback dispatchers (Stabilized with refs to prevent RAF rebuilds)
  const onSignConfirmedRef = useRef(onSignConfirmed);
  useEffect(() => { onSignConfirmedRef.current = onSignConfirmed; }, [onSignConfirmed]);

  const onCandidateChangedRef = useRef(onCandidateChanged);
  useEffect(() => { onCandidateChangedRef.current = onCandidateChanged; }, [onCandidateChanged]);

  const onRecognitionStateChangedRef = useRef(onRecognitionStateChanged);
  useEffect(() => { onRecognitionStateChangedRef.current = onRecognitionStateChanged; }, [onRecognitionStateChanged]);

  const onTopProbabilitiesChangedRef = useRef(onTopProbabilitiesChanged);
  useEffect(() => { onTopProbabilitiesChangedRef.current = onTopProbabilitiesChanged; }, [onTopProbabilitiesChanged]);

  const setRecognitionStateSafe = useCallback((state: RecognitionState) => {
    if (latestRecognitionStateRef.current !== state) {
      latestRecognitionStateRef.current = state;
      setActiveRecognitionState(state);
      onRecognitionStateChangedRef.current?.(state);
    }
  }, []);

  // Fullscreen / Expand Handler
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  // Performance Telemetry Periodic Sync (every 250ms)
  useEffect(() => {
    const timer = setInterval(() => {
      const now = performance.now();
      const elapsed = (now - lastFpsTimeRef.current) / 1000;
      const currentFps = elapsed > 0 ? Math.round((frameCountRef.current / elapsed) * 10) / 10 : 0;
      frameCountRef.current = 0;
      lastFpsTimeRef.current = now;

      setPerfMetrics({
        fps: currentFps,
        landmarkMs: Math.round(lastLandmarkMsRef.current * 10) / 10,
        inferMs: Math.round(lastInferMsRef.current * 10) / 10,
        confirmMs: Math.round(lastConfirmMsRef.current * 10) / 10,
        latencyMs: Math.round(lastLatencyMsRef.current * 10) / 10,
        droppedFrames: droppedFramesRef.current,
      });
    }, 250);

    return () => clearInterval(timer);
  }, []);

  // Direct 2D Canvas Landmark Drawing
  const drawLandmarksToCanvas = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    rh: Array<{ x: number; y: number; z: number }> | null,
    lh: Array<{ x: number; y: number; z: number }> | null,
    state: RecognitionState
  ) => {
    ctx.clearRect(0, 0, width, height);
    if (!showLandmarks) return;

    // State-based glowing accent colors
    let boneColor = 'rgba(18, 184, 194, 0.75)'; // Default cyan
    let jointColor = '#12B8C2';
    if (state === 'CONFIRMED') {
      boneColor = 'rgba(16, 185, 129, 0.9)'; // Green
      jointColor = '#10B981';
    } else if (state === 'CANDIDATE') {
      boneColor = 'rgba(245, 158, 11, 0.85)'; // Amber
      jointColor = '#F59E0B';
    } else if (state === 'UNKNOWN') {
      boneColor = 'rgba(244, 63, 94, 0.75)'; // Rose
      jointColor = '#F43F5E';
    }

    const drawHandMesh = (lms: Array<{ x: number; y: number; z: number }>) => {
      // Draw Connections (Batched in a single path stroke to eliminate canvas draw-call overhead)
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = boneColor;
      ctx.beginPath();
      for (const [startIdx, endIdx] of CONNECTIONS) {
        const p1 = lms[startIdx];
        const p2 = lms[endIdx];
        if (!p1 || !p2) continue;
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
      }
      ctx.stroke();

      // Draw Keypoints
      for (let i = 0; i < lms.length; i++) {
        const p = lms[i];
        if (!p) continue;
        const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
        const radius = isTip ? 4.5 : i === 0 ? 5.5 : 3.0;

        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isTip ? '#FFFFFF' : jointColor;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = isTip ? jointColor : '#FFFFFF';
        ctx.stroke();
      }
    };

    if (rh) drawHandMesh(rh);
    if (lh) drawHandMesh(lh);
  };

  // Main Camera & Perception Loop
  useEffect(() => {
    if (!isCameraOn) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      workerRef.current?.terminate();
      workerRef.current = null;
      setCameraStatus('Camera Off');
      setHandsCount(0);
      setRecognitionStateSafe('IDLE');
      return;
    }

    let isMounted = true;
    setCameraError(null);
    setCameraStatus('Starting Camera...');

    async function initCameraPipeline() {
      try {
        // 1. Initialize Webcam Stream (Optimized 640x480 ideal resolution for low-latency MediaPipe tracking)
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640, min: 480 },
            height: { ideal: 480, min: 360 },
            facingMode: 'user',
          },
          audio: false,
        });

        if (!isMounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setCameraStatus('Live Feed Active');

        // 2. Initialize Web Worker for Background TF.js Inference
        const worker = new Worker(new URL('../../workers/inferenceWorker.ts', import.meta.url), {
          type: 'module',
        });
        workerRef.current = worker;

        worker.onmessage = (ev) => {
          const msg = ev.data;
          if (msg.type === 'PONG') {
            worker.postMessage({
              type: 'LOAD_MODEL',
              modelUrl: '/models/sbv-gru-v1/model.json',
              metadataUrl: '/models/sbv-gru-v1/metadata.json',
            });
          } else if (msg.type === 'MODEL_LOADED') {
            console.log(`[OptimizedCamera] Model initialized (${msg.activeBackend || 'active'})`);
          } else if (msg.type === 'RESULT') {
            inferenceInFlightRef.current = false;
            const res: RecognitionResult = msg.result;
            const now = performance.now();

            if (msg.inferDurationMs) {
              lastInferMsRef.current = msg.inferDurationMs;
            }
            if (msg.sentAt) {
              lastLatencyMsRef.current = now - msg.sentAt;
            }

            if (res?.probabilities && res.probabilities.length > 0) {
              onTopProbabilitiesChangedRef.current?.(res.probabilities.slice(0, 5));
            }

            // Adaptive Confirmation Machine
            if (
              res &&
              res.confidence >= confidenceThreshold &&
              res.gesture !== 'NONE' &&
              res.gesture !== 'NONE_RESTING' &&
              res.gesture !== 'UNKNOWN'
            ) {
              neutralFramesCountRef.current = 0;

              // Check if user is holding already confirmed sign
              if (heldSignRef.current?.sign === res.gesture) {
                // Suppress duplicate triggers while held
                setRecognitionStateSafe('CONFIRMED');
                return;
              }

              // Fast confirmation:
              // Immediate confirmation if confidence is high (>= 0.58)
              // Or 2 consecutive frames if confidence is between 0.48 and 0.58
              const isHighConfidence = res.confidence >= 0.58;
              const isConsecutiveMatch = candidateSignRef.current === res.gesture;

              if (isHighConfidence || (isConsecutiveMatch && candidateCountRef.current >= 1)) {
                const confirmDuration = now - candidateStartAtRef.current;
                lastConfirmMsRef.current = confirmDuration;

                setRecognitionStateSafe('CONFIRMED');
                heldSignRef.current = { sign: res.gesture, confirmedAt: now };
                candidateSignRef.current = null;
                candidateCountRef.current = 0;
                onCandidateChangedRef.current?.(null);

                // Trigger verified sign confirmation callback
                onSignConfirmedRef.current(res);
              } else {
                // Evaluating candidate sign
                candidateSignRef.current = res.gesture;
                candidateCountRef.current = (candidateCountRef.current || 0) + 1;
                candidateStartAtRef.current = now;
                setRecognitionStateSafe('CANDIDATE');
                onCandidateChangedRef.current?.({
                  sign: res.gesture,
                  label: res.displayName,
                  confidence: res.confidence,
                });
              }
            } else if (res?.gesture === 'UNKNOWN') {
              // Sign actively being formed but unrecognized
              candidateSignRef.current = null;
              candidateCountRef.current = 0;
              onCandidateChangedRef.current?.(null);
              setRecognitionStateSafe('UNKNOWN');
            } else {
              // Neutral / Resting Hand State
              neutralFramesCountRef.current++;
              // Release held sign after 3 neutral frames (~100ms) allowing the same sign to be repeated
              if (neutralFramesCountRef.current >= 3) {
                heldSignRef.current = null;
              }
              candidateSignRef.current = null;
              candidateCountRef.current = 0;
              onCandidateChangedRef.current?.(null);

              if (bufferRef.current.length > 0) {
                setRecognitionStateSafe('ANALYZING');
              } else {
                setRecognitionStateSafe('HAND_DETECTED');
              }
            }
          } else if (msg.type === 'ERROR') {
            inferenceInFlightRef.current = false;
          }
        };

        worker.postMessage({ type: 'PING' });

        // 3. Obtain Reused Singleton MediaPipe HandLandmarker
        const hl = await createHandLandmarker();
        if (!isMounted) {
          worker.terminate();
          return;
        }
        mpRef.current = hl;

        // 4. Zero-React-Render RequestAnimationFrame Loop
        let lastVideoTime = -1;

        const processFrame = () => {
          if (!isMounted) return;
          const video = videoRef.current;
          const canvas = canvasRef.current;

          if (video && video.readyState >= 2 && mpRef.current && canvas) {
            // Check if frame has advanced
            if (video.currentTime !== lastVideoTime) {
              lastVideoTime = video.currentTime;
              frameCountRef.current++;

              // Match canvas resolution to video
              if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
              }

              // Extract Hand Landmarks with High-Precision Timing
              const tLm0 = performance.now();
              const result = mpRef.current.detectForVideo(video, tLm0);
              lastLandmarkMsRef.current = performance.now() - tLm0;

              let rh: Array<{ x: number; y: number; z: number }> | null = null;
              let lh: Array<{ x: number; y: number; z: number }> | null = null;

              if (result?.landmarks && result.landmarks.length > 0) {
                if (result.landmarks.length === 1) {
                  // Dominant hand routing: map single hand directly to dominant rightHand slot
                  rh = result.landmarks[0].map((lm: any) => ({ x: lm.x, y: lm.y, z: lm.z }));
                } else {
                  for (let i = 0; i < result.landmarks.length; i++) {
                    const isRight = result.handedness?.[i]?.[0]?.categoryName === 'Right';
                    const lms = result.landmarks[i].map((lm: any) => ({ x: lm.x, y: lm.y, z: lm.z }));
                    if (isRight) rh = lms;
                    else lh = lms;
                  }
                  if (!rh && lh) { rh = lh; lh = null; }
                }
              }

              const count = rh || lh ? (rh && lh ? 2 : 1) : 0;
              setHandsCount(count);

              // Render landmarks directly onto canvas context without React state changes
              const ctx = canvas.getContext('2d');
              if (ctx) {
                drawLandmarksToCanvas(ctx, canvas.width, canvas.height, rh, lh, latestRecognitionStateRef.current);
              }

              // Manage Temporal Buffer & Dispatch Worker Inference
              if (rh || lh) {
                bufferRef.current.push({ rightHand: rh, leftHand: lh });
                if (bufferRef.current.length > 30) bufferRef.current.shift();

                // Dispatch latest frame buffer to worker (non-blocking, drops backlogged frames)
                if (
                  bufferRef.current.length >= 3 &&
                  !inferenceInFlightRef.current &&
                  workerRef.current &&
                  !isSigningPaused
                ) {
                  inferenceInFlightRef.current = true;
                  const frames = [...bufferRef.current];
                  const matrix = buildNormalizedMatrix(frames, 30);
                  const motionVector = computeMotionVector(frames);

                  workerRef.current.postMessage(
                    {
                      type: 'INFER',
                      features: matrix,
                      motionVector,
                      windowSize: 30,
                      featureDim: 126,
                      sentAt: performance.now(),
                      tAcq: lastVideoTime,
                      tPrep: lastLandmarkMsRef.current,
                    },
                    [matrix.buffer]
                  );
                } else if (inferenceInFlightRef.current) {
                  droppedFramesRef.current++;
                }
              } else {
                // Clear buffer when hands leave frame
                bufferRef.current = [];
                heldSignRef.current = null;
                candidateSignRef.current = null;
                candidateCountRef.current = 0;
                setRecognitionStateSafe('IDLE');
              }
            }
          }

          rafRef.current = requestAnimationFrame(processFrame);
        };

        rafRef.current = requestAnimationFrame(processFrame);
      } catch (err: any) {
        console.error('[OptimizedCamera] Pipeline failure:', err);
        setCameraError(err.message || 'Camera permission denied or camera device not found.');
        setCameraStatus('Camera Error');
      }
    }

    initCameraPipeline();

    return () => {
      isMounted = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, [isCameraOn, confidenceThreshold, isSigningPaused, setRecognitionStateSafe]);

  return (
    <div
      ref={containerRef}
      className={`bg-white rounded-2xl border border-[#DCE5EF] shadow-sm flex flex-col overflow-hidden transition-all ${className}`}
    >
      {/* 1. Header Toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#DCE5EF] bg-white">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              {isCameraOn ? (
                <>
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#12B8C2] opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#12B8C2]" />
                </>
              ) : (
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-slate-300" />
              )}
            </span>
            <span className="text-xs font-bold tracking-wider text-[#14264F] uppercase flex items-center gap-1.5">
              Live ISL Camera Stream
            </span>
          </div>

          <span
            className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full ${
              isCameraOn
                ? handsCount > 0
                  ? 'bg-[#E7F4FC] text-[#12B8C2] border border-[#12B8C2]/30'
                  : 'bg-slate-100 text-[#71809A]'
                : 'bg-rose-50 text-rose-500'
            }`}
          >
            {isCameraOn ? (handsCount > 0 ? `${handsCount} Hand(s) Tracked` : 'Searching Hands') : 'Camera Paused'}
          </span>
        </div>

        {/* Viewport Action Controls */}
        <div className="flex items-center gap-2">
          {/* Skeleton Landmark Toggle */}
          <button
            type="button"
            onClick={() => setShowLandmarks((prev) => !prev)}
            title={showLandmarks ? 'Hide Hand Skeleton Overlay' : 'Show Hand Skeleton Overlay'}
            className={`p-1.5 rounded-lg border text-xs transition flex items-center gap-1 ${
              showLandmarks
                ? 'bg-[#E7F4FC] text-[#12B8C2] border-[#12B8C2]/40'
                : 'bg-slate-50 text-[#71809A] border-[#DCE5EF] hover:bg-slate-100'
            }`}
          >
            {showLandmarks ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline text-[11px] font-medium">Skeleton</span>
          </button>

          {/* Performance HUD Toggle */}
          <button
            type="button"
            onClick={() => setShowPerfHUD((prev) => !prev)}
            title="Toggle Live Performance Metrics"
            className={`p-1.5 rounded-lg border text-xs transition flex items-center gap-1 ${
              showPerfHUD
                ? 'bg-[#14264F] text-white border-[#14264F]'
                : 'bg-slate-50 text-[#71809A] border-[#DCE5EF] hover:bg-slate-100'
            }`}
          >
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span className="hidden sm:inline text-[11px] font-medium">Perf</span>
          </button>

          {/* Fullscreen Expand Toggle */}
          <button
            type="button"
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Expand to Fullscreen'}
            className="p-1.5 rounded-lg border border-[#DCE5EF] bg-slate-50 text-[#71809A] hover:bg-slate-100 transition"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>

          {/* Pause / Resume Tracking */}
          {onToggleSigningPaused && (
            <button
              type="button"
              onClick={onToggleSigningPaused}
              title={isSigningPaused ? 'Resume Tracking' : 'Pause Tracking'}
              className={`p-1.5 rounded-lg border text-xs transition flex items-center gap-1 ${
                isSigningPaused
                  ? 'bg-amber-50 text-amber-600 border-amber-300'
                  : 'bg-slate-50 text-[#71809A] border-[#DCE5EF] hover:bg-slate-100'
              }`}
            >
              {isSigningPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            </button>
          )}

          {/* Camera Power Toggle Switch */}
          <button
            type="button"
            onClick={onToggleCamera}
            aria-label="Toggle camera stream"
            className={`relative inline-flex h-5 w-10 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
              isCameraOn ? 'bg-[#12B8C2]' : 'bg-[#DCE5EF]'
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-md ring-0 transition duration-200 ease-in-out ${
                isCameraOn ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
        </div>
      </div>

      {/* 2. Primary Large Camera Feed Viewport (Min height 420px on desktop) */}
      <div className="relative flex-1 w-full min-h-[420px] sm:min-h-[460px] bg-slate-950 flex items-center justify-center overflow-hidden">
        {isCameraOn ? (
          <>
            {/* Mirrored Uncropped Video Feed */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-contain pointer-events-none"
              style={{ transform: 'scaleX(-1)' }}
            />

            {/* Synchronized Transparent Landmark Canvas */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              style={{ transform: 'scaleX(-1)' }}
            />

            {/* Live State Badge Overlay (Top Left of Camera View) */}
            <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-none">
              {activeRecognitionState === 'CONFIRMED' && (
                <div className="px-3 py-1 rounded-full bg-emerald-500/90 backdrop-blur-md text-white text-xs font-bold flex items-center gap-1.5 shadow-lg animate-pulse">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>SIGN CONFIRMED</span>
                </div>
              )}
              {activeRecognitionState === 'CANDIDATE' && (
                <div className="px-3 py-1 rounded-full bg-amber-500/90 backdrop-blur-md text-white text-xs font-bold flex items-center gap-1.5 shadow-lg">
                  <Activity className="w-3.5 h-3.5 animate-spin" />
                  <span>EVALUATING STABILITY</span>
                </div>
              )}
              {activeRecognitionState === 'ANALYZING' && (
                <div className="px-3 py-1 rounded-full bg-cyan-600/80 backdrop-blur-md text-white text-xs font-semibold flex items-center gap-1.5 shadow">
                  <Sparkles className="w-3 h-3 text-cyan-200" />
                  <span>ANALYZING GESTURE</span>
                </div>
              )}
              {activeRecognitionState === 'UNKNOWN' && (
                <div className="px-3 py-1 rounded-full bg-rose-500/90 backdrop-blur-md text-white text-xs font-bold flex items-center gap-1.5 shadow-lg">
                  <AlertCircle className="w-3.5 h-3.5" />
                  <span>UNKNOWN SIGN</span>
                </div>
              )}
            </div>

            {/* Developer Performance Metrics HUD (Part 5) */}
            {showPerfHUD && (
              <div className="absolute bottom-3 left-3 right-3 bg-slate-900/85 backdrop-blur-md border border-slate-700/80 rounded-xl p-2.5 text-white shadow-xl transition-all">
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center divide-x divide-slate-800">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Camera FPS</span>
                    <span className={`text-xs font-mono font-bold ${perfMetrics.fps >= 24 ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {perfMetrics.fps > 0 ? `${perfMetrics.fps} fps` : '--'}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Landmark</span>
                    <span className="text-xs font-mono font-bold text-cyan-400">
                      {perfMetrics.landmarkMs > 0 ? `${perfMetrics.landmarkMs}ms` : '--'}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Inference</span>
                    <span className="text-xs font-mono font-bold text-indigo-400">
                      {perfMetrics.inferMs > 0 ? `${perfMetrics.inferMs}ms` : '--'}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Confirmation</span>
                    <span className="text-xs font-mono font-bold text-amber-400">
                      {perfMetrics.confirmMs > 0 ? `${perfMetrics.confirmMs}ms` : '180ms (target)'}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Latency</span>
                    <span className="text-xs font-mono font-bold text-emerald-400">
                      {perfMetrics.latencyMs > 0 ? `${perfMetrics.latencyMs}ms` : '--'}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Dropped</span>
                    <span className="text-xs font-mono font-bold text-slate-300">
                      {perfMetrics.droppedFrames}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center p-8 text-center text-slate-400">
            <CameraOff className="w-12 h-12 mb-3 text-slate-600" />
            <p className="text-sm font-semibold text-slate-300 mb-1">Webcam Stream Paused</p>
            <p className="text-xs text-slate-500 max-w-sm mb-4">
              Toggle camera stream on using the switch above to resume real-time ISL tracking.
            </p>
            <button
              onClick={onToggleCamera}
              className="px-4 py-2 rounded-xl bg-[#12B8C2] hover:bg-[#0fa0a9] text-white text-xs font-bold transition shadow-sm"
            >
              Start Camera Feed
            </button>
          </div>
        )}

        {/* Camera Error Display */}
        {cameraError && (
          <div className="absolute inset-x-4 top-4 bg-rose-500/90 text-white p-3 rounded-xl backdrop-blur-md text-xs font-medium flex items-center justify-between shadow-lg">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{cameraError}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
