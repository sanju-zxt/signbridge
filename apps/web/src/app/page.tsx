'use client';
/**
 * page.tsx — SignBridge2Vision Live Meeting & Studio Platform
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from '../components/Header';
import { AudioControls } from '../components/AudioControls';
import { ConversationThread } from '../components/ConversationThread';
import { ConfidenceGate } from '../components/ConfidenceGate';
import { MetricsDashboard } from '../components/MetricsDashboard';
import { DemoControls } from '../components/DemoControls';
import { SignAvatar } from '../components/avatar/SignAvatar';
import LandmarkOverlay from '../components/vision/LandmarkOverlay';
import LiveCameraView from '../components/vision/LiveCameraView';
import DataCollectionStudio from '../components/studio/DataCollectionStudio';
import { useRealtimeSession } from '../hooks/useRealtimeSession';
import { useISLPerception } from '../hooks/useISLPerception';
import { StreamingSTTClient } from '../lib/speech/stt';
import { ttsPlayer } from '../lib/speech/tts';
import { ISL_VOCABULARY, RecognitionResult, CanonicalMessage, RawLandmarkFrame } from '@signbridge2vision/types';
import {
  Mic,
  MicOff,
  Hand,
  Volume2,
  Camera,
  CameraOff,
  Sparkles,
  LayoutDashboard,
  Video,
  Database,
  Send,
  AlertCircle,
} from 'lucide-react';

// ── Landmark Normalization and Feature Extraction Helpers ───────────────────
function normHand(lms: Array<{ x: number; y: number; z: number }> | null): Float32Array {
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

function buildMatrix(
  frames: Array<{ rightHand: Array<{ x: number; y: number; z: number }> | null; leftHand: Array<{ x: number; y: number; z: number }> | null }>,
  winSize = 30
): Float32Array {
  const matrix = new Float32Array(winSize * 126);
  const offset = Math.max(0, winSize - frames.length);
  for (let i = 0; i < frames.length && i < winSize; i++) {
    const rh = normHand(frames[i].rightHand);
    const lh = normHand(frames[i].leftHand);
    const base = (offset + i) * 126;
    matrix.set(rh, base);
    matrix.set(lh, base + 63);
  }
  return matrix;
}

function computeMotionVector(
  frames: Array<{ rightHand: Array<{ x: number; y: number; z: number }> | null; leftHand: Array<{ x: number; y: number; z: number }> | null }>
): [number, number, number] {
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

export default function SignBridge2VisionApp() {
  const [viewMode, setViewMode] = useState<'meeting' | 'studio'>('meeting');
  const [studioTab, setStudioTab] = useState<'recognition' | 'collection'>('recognition');
  const [studioRawFrame, setStudioRawFrame] = useState<RawLandmarkFrame | null>(null);

  const [isMicOn, setIsMicOn] = useState<boolean>(true);
  const [isGestureEnabled, setIsGestureEnabled] = useState<boolean>(true);
  const [isSignToVoiceEnabled, setIsSignToVoiceEnabled] = useState<boolean>(true);
  const [isCameraOn, setIsCameraOn] = useState<boolean>(true);

  const [recognizedSign, setRecognizedSign] = useState<string | null>(null);
  const [recognizedConfidence, setRecognizedConfidence] = useState<number>(0);
  const [handsDetected, setHandsDetected] = useState<number>(0);
  const [liveSubtitle, setLiveSubtitle] = useState<string>('Listening for speech... (speak or type below)');
  const [typedText, setTypedText] = useState<string>('');
  const [sttError, setSttError] = useState<string | null>(null);
  const [isListeningActive, setIsListeningActive] = useState<boolean>(false);
  const [avatarSign, setAvatarSign] = useState<string | null>('HELLO');

  // Subtitle latency & interim tracking
  const [speechStatus, setSpeechStatus] = useState<'idle' | 'listening' | 'interim' | 'final'>('idle');
  const speechStartRef = useRef<number>(0);
  const [lastSpeechLatencyMs, setLastSpeechLatencyMs] = useState<number | null>(null);
  const [speechTranscriptForAvatar, setSpeechTranscriptForAvatar] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const mpRef = useRef<any>(null);
  const [rightLandmarks, setRightLandmarks] = useState<Array<{ x: number; y: number; z: number }> | null>(null);
  const [leftLandmarks, setLeftLandmarks] = useState<Array<{ x: number; y: number; z: number }> | null>(null);
  const [isRecognizedFlash, setIsRecognizedFlash] = useState<boolean>(false);
  const [recognizedAt, setRecognizedAt] = useState<number | null>(null);

  // Meeting View Inference Worker & Temporal Buffer
  const meetingWorkerRef = useRef<Worker | null>(null);
  const meetingBufferRef = useRef<Array<{ rightHand: Array<{ x: number; y: number; z: number }> | null; leftHand: Array<{ x: number; y: number; z: number }> | null }>>([]);
  const inferenceRunningRef = useRef<boolean>(false);
  const stableSignRef = useRef<string | null>(null);
  const stableCountRef = useRef<number>(0);
  const lastEmitRef = useRef<{ sign: string; at: number } | null>(null);

  // Reset recognized sign display when hand leaves view
  useEffect(() => {
    if (handsDetected === 0 && recognizedSign) {
      const timer = setTimeout(() => {
        setRecognizedSign(null);
      }, 1800);
      return () => clearTimeout(timer);
    }
  }, [handsDetected, recognizedSign]);

  const [isMetricsOpen, setIsMetricsOpen] = useState(false);
  const [isPresetsOpen, setIsPresetsOpen] = useState(false);

  const sttClientRef = useRef<StreamingSTTClient | null>(null);

  const {
    messages,
    systemState,
    isConnected,
    activeLanguage,
    setActiveLanguage,
    partialSpeech,
    latencyMetrics,
    sendFinalISLSign,
    sendFinalSpeech,
    sendTextMessage,
    clearSession,
  } = useRealtimeSession('main-room', 'isl_signer');

  const triggerSpeechTranscript = useCallback((text: string) => {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();
    console.log(`[STT] Final transcript received in application: "${cleanText}"`);
    setLiveSubtitle(cleanText);
    setSttError(null);
    setSpeechStatus('final');
    sendFinalSpeech(cleanText, 0.95);
    setSpeechTranscriptForAvatar(cleanText);
  }, [sendFinalSpeech]);

  const triggerSpeechTranscriptRef = useRef(triggerSpeechTranscript);
  useEffect(() => {
    triggerSpeechTranscriptRef.current = triggerSpeechTranscript;
  }, [triggerSpeechTranscript]);

  const handleRecognizedSignFromCamera = useCallback(
    (result: RecognitionResult) => {
      if (!isGestureEnabled) return;
      if (
        !result.gesture ||
        result.gesture === 'NONE' ||
        result.gesture === 'NONE_RESTING' ||
        result.gesture === 'UNKNOWN' ||
        result.confidence < 0.60
      ) {
        return;
      }

      setRecognizedSign(result.displayName);
      setRecognizedConfidence(result.confidence);
      setIsRecognizedFlash(true);
      setRecognizedAt(Date.now());

      setAvatarSign(result.gesture);

      if (isSignToVoiceEnabled) {
        const vocab = ISL_VOCABULARY.find((v) => v.sign === result.gesture);
        const spokenText = vocab?.canonicalSentence || result.displayName;
        ttsPlayer.speak(spokenText, activeLanguage);
      }

      sendFinalISLSign({
        id: `sign_${Date.now()}`,
        sessionId: 'main-room',
        timestamp: Date.now(),
        sign: result.gesture,
        label: result.displayName,
        confidence: result.confidence,
        alternatives: [],
        modelVersion: 'SBV-ML-TEMPORAL-v1',
        latencyMs: result.latencyMs,
        isDebounced: true,
        status: 'final',
      });

      setTimeout(() => setIsRecognizedFlash(false), 800);
    },
    [isGestureEnabled, isSignToVoiceEnabled, activeLanguage, sendFinalISLSign]
  );

  const handleRecognizedSignRef = useRef(handleRecognizedSignFromCamera);
  useEffect(() => {
    handleRecognizedSignRef.current = handleRecognizedSignFromCamera;
  }, [handleRecognizedSignFromCamera]);

  // Camera & MediaPipe Initialization for Meeting View with live inference
  useEffect(() => {
    if (viewMode !== 'meeting') return;
    if (!isCameraOn) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setRightLandmarks(null);
      setLeftLandmarks(null);
      meetingWorkerRef.current?.terminate();
      meetingWorkerRef.current = null;
      return;
    }

    let isMounted = true;

    async function initMeetingCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        });
        if (!isMounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }

        // Initialize Web Worker for real-time inference in meeting view
        const worker = new Worker(new URL('../workers/inferenceWorker.ts', import.meta.url), { type: 'module' });
        meetingWorkerRef.current = worker;

        worker.onmessage = (ev) => {
          const msg = ev.data;
          if (msg.type === 'PONG') {
            worker.postMessage({
              type: 'LOAD_MODEL',
              modelUrl: '/models/sbv-gru-v1/model.json',
              metadataUrl: '/models/sbv-gru-v1/metadata.json',
            });
          } else if (msg.type === 'RESULT') {
            inferenceRunningRef.current = false;
            const res: RecognitionResult = msg.result;
            if (
              res &&
              res.confidence >= 0.60 &&
              res.gesture !== 'NONE' &&
              res.gesture !== 'NONE_RESTING' &&
              res.gesture !== 'UNKNOWN'
            ) {
              if (res.gesture === stableSignRef.current) {
                stableCountRef.current++;
              } else {
                stableSignRef.current = res.gesture;
                stableCountRef.current = 1;
              }
              if (stableCountRef.current >= 2) {
                const now = Date.now();
                const isDuplicate = lastEmitRef.current?.sign === res.gesture && now - (lastEmitRef.current?.at ?? 0) < 2500;
                if (!isDuplicate) {
                  lastEmitRef.current = { sign: res.gesture, at: now };
                  handleRecognizedSignRef.current(res);
                }
              }
            } else {
              stableCountRef.current = 0;
              stableSignRef.current = null;
            }
          } else if (msg.type === 'ERROR') {
            inferenceRunningRef.current = false;
          }
        };

        worker.postMessage({ type: 'PING' });

        const { createHandLandmarker } = await import('../lib/vision/handLandmarker');
        const hl = await createHandLandmarker();

        if (!isMounted) {
          worker.terminate();
          return;
        }
        mpRef.current = hl;

        let lastTime = -1;
        const loop = () => {
          if (!isMounted) return;
          const video = videoRef.current;
          if (video && video.readyState >= 2 && mpRef.current) {
            if (video.currentTime !== lastTime) {
              const tAcqStart = performance.now();
              lastTime = video.currentTime;
              const tAcq = performance.now() - tAcqStart;

              const tPrepStart = performance.now();
              const result = mpRef.current.detectForVideo(video, performance.now());

              let rh: Array<{ x: number; y: number; z: number }> | null = null;
              let lh: Array<{ x: number; y: number; z: number }> | null = null;

              if (result?.landmarks && result.landmarks.length > 0) {
                for (let i = 0; i < result.landmarks.length; i++) {
                  const isRight = result.handedness?.[i]?.[0]?.categoryName === 'Right';
                  const lms = result.landmarks[i].map((lm: any) => ({ x: lm.x, y: lm.y, z: lm.z }));
                  if (isRight) rh = lms;
                  else lh = lms;
                }
              }

              setRightLandmarks(rh);
              setLeftLandmarks(lh);
              setHandsDetected(rh || lh ? (rh && lh ? 2 : 1) : 0);

              if (rh || lh) {
                meetingBufferRef.current.push({ rightHand: rh, leftHand: lh });
                if (meetingBufferRef.current.length > 30) meetingBufferRef.current.shift();

                if (meetingBufferRef.current.length >= 5 && !inferenceRunningRef.current && meetingWorkerRef.current) {
                  inferenceRunningRef.current = true;
                  const frames = [...meetingBufferRef.current];
                  const matrix = buildMatrix(frames, 30);
                  const motionVector = computeMotionVector(frames);
                  const tPrep = performance.now() - tPrepStart;
                  meetingWorkerRef.current.postMessage(
                    {
                      type: 'INFER',
                      features: matrix,
                      motionVector,
                      windowSize: 30,
                      featureDim: 126,
                      sentAt: performance.now(),
                      tAcq,
                      tPrep,
                    },
                    [matrix.buffer]
                  );
                }
              } else {
                meetingBufferRef.current = [];
                stableCountRef.current = 0;
                stableSignRef.current = null;
              }
            }
          }
          rafRef.current = requestAnimationFrame(loop);
        };
        loop();
      } catch (err) {
        console.error('Meeting camera init error:', err);
      }
    }

    initMeetingCamera();

    return () => {
      isMounted = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      meetingWorkerRef.current?.terminate();
      meetingWorkerRef.current = null;
    };
  }, [viewMode, isCameraOn]);

  // STT Microphone Streaming (Active in Meeting View)
  useEffect(() => {
    if (viewMode === 'meeting' && isMicOn) {
      sttClientRef.current = new StreamingSTTClient({
        language: activeLanguage,
        onStateChange: (listening) => {
          setIsListeningActive(listening);
          setSpeechStatus(listening ? 'listening' : 'idle');
        },
        onPartial: (partial) => {
          if (!speechStartRef.current) speechStartRef.current = Date.now();
          setSpeechStatus('interim');
          setLiveSubtitle(partial);
          setSttError(null);
        },
        onFinal: (finalText) => {
          const latency = speechStartRef.current ? Math.round(Date.now() - speechStartRef.current) : 110;
          speechStartRef.current = 0;
          setLastSpeechLatencyMs(latency);
          setSpeechStatus('final');
          triggerSpeechTranscriptRef.current(finalText);
        },
        onError: (err) => {
          console.warn('STT Error:', err);
          setSttError(err);
          setSpeechStatus('idle');
        },
      });
      sttClientRef.current.start();
    } else {
      sttClientRef.current?.stop();
      setIsListeningActive(false);
      setSpeechStatus('idle');
      if (viewMode === 'meeting') {
        setLiveSubtitle('Mic is muted. Click Mic ON or type below.');
      }
    }

    return () => {
      sttClientRef.current?.stop();
    };
  }, [viewMode, isMicOn, activeLanguage]);

  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    if (typedText.trim()) {
      triggerSpeechTranscript(typedText.trim());
      setTypedText('');
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col">
      {/* Top Navbar */}
      <nav className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-black text-slate-900 tracking-tight">SignBridge2Vision</span>
          <span className="hidden sm:inline-block bg-sky-100 text-sky-800 text-xs font-bold px-2.5 py-0.5 rounded-full">
            Real-Time Multimodal AI
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('meeting')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
              viewMode === 'meeting'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Video className="w-3.5 h-3.5" /> Meeting View (3D Avatar)
          </button>
          <button
            onClick={() => setViewMode('studio')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
              viewMode === 'studio'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <LayoutDashboard className="w-3.5 h-3.5" /> Studio Workspace
          </button>
        </div>
      </nav>

      {/* 1. MEETING VIEW */}
      {viewMode === 'meeting' ? (
        <div className="flex-1 p-6 sm:p-10 max-w-7xl mx-auto w-full flex flex-col">
          <header className="mb-6">
            <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">SignBridge2Vision</h1>
            <p className="text-base text-slate-600 font-medium mt-1">Meeting ID : 12345</p>
          </header>

          <main className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start flex-1">
            {/* Left Column */}
            <section className="flex flex-col gap-4">
              <h2 className="text-xl font-bold text-slate-900">Camera Feed</h2>

              <div className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-slate-900 border border-slate-200 shadow-xl">
                {isCameraOn ? (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                      style={{ transform: 'scaleX(-1)' }}
                    />
                    <div className="absolute inset-0 pointer-events-none" style={{ transform: 'scaleX(-1)' }}>
                      <LandmarkOverlay
                        rightLandmarks={rightLandmarks}
                        leftLandmarks={leftLandmarks}
                        width={640}
                        height={480}
                        isRecognized={isRecognizedFlash}
                        recognizedAt={recognizedAt}
                      />
                    </div>
                  </>
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center bg-slate-800 text-slate-400 gap-2">
                    <CameraOff className="w-12 h-12" />
                    <span className="text-sm font-medium">Camera is Turned OFF</span>
                  </div>
                )}
              </div>

              <div>
                <h3 className="text-lg font-bold text-slate-900">Gesture Recognition</h3>
                <p className="text-base font-semibold text-slate-700 mt-1">
                  {handsDetected === 0 ? (
                    <span className="text-slate-500 bg-slate-100 px-3 py-1 rounded-md border border-slate-200">
                      No hand in view
                    </span>
                  ) : recognizedSign ? (
                    <span className="text-emerald-600 bg-emerald-50 px-3 py-1 rounded-md border border-emerald-200">
                      {recognizedSign} ({Math.round(recognizedConfidence * 100)}%)
                    </span>
                  ) : (
                    <span className="text-sky-600 bg-sky-50 px-3 py-1 rounded-md border border-sky-200 animate-pulse">
                      Detecting gesture…
                    </span>
                  )}
                </p>
              </div>

              <div>
                <h3 className="text-lg font-bold text-slate-900">Live Subtitles</h3>
                <div className="mt-2 bg-[#2a9d8f] text-white px-5 py-3.5 rounded-xl font-semibold text-base shadow-md flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5 flex-1 min-w-0">
                    <span
                      className={`w-3 h-3 rounded-full flex-shrink-0 ${
                        speechStatus === 'interim'
                          ? 'bg-amber-300 animate-ping'
                          : isListeningActive
                          ? 'bg-emerald-300 animate-pulse'
                          : 'bg-white/80'
                      }`}
                    />
                    <span className="truncate">{liveSubtitle}</span>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {speechStatus === 'interim' && (
                      <span className="text-xs bg-amber-400/30 border border-amber-300/40 px-2 py-0.5 rounded text-amber-100 font-semibold flex items-center gap-1">
                        Interim
                      </span>
                    )}
                    {speechStatus === 'final' && lastSpeechLatencyMs !== null && (
                      <span className="text-xs bg-white/20 px-2 py-0.5 rounded text-white font-mono">
                        {lastSpeechLatencyMs}ms
                      </span>
                    )}
                    {isListeningActive && (
                      <span className="text-xs bg-white/20 px-2 py-0.5 rounded text-white font-medium">Live Mic</span>
                    )}
                  </div>
                </div>

                {sttError && (
                  <div className="mt-2 text-xs text-rose-600 bg-rose-50 border border-rose-200 p-2.5 rounded-lg flex items-start gap-1.5">
                    <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <span>{sttError}</span>
                  </div>
                )}

                {/* Quick Speech Test Bar */}
                <form onSubmit={handleSendText} className="mt-3 flex gap-2">
                  <input
                    type="text"
                    value={typedText}
                    onChange={(e) => setTypedText(e.target.value)}
                    placeholder="Type speech message (e.g. Where is hospital?)..."
                    className="flex-1 bg-slate-100 border border-slate-300 rounded-xl px-3.5 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#2a9d8f]"
                  />
                  <button
                    type="submit"
                    className="bg-[#2a9d8f] hover:bg-[#21867a] text-white px-4 py-2 rounded-xl text-sm font-bold flex items-center gap-1 shadow-sm transition-all"
                  >
                    <Send className="w-3.5 h-3.5" /> Send
                  </button>
                </form>

                {/* Preset Speech Suggestions */}
                <div className="mt-2 flex items-center gap-1.5 overflow-x-auto text-xs py-1">
                  <span className="text-slate-500 font-medium text-[11px] whitespace-nowrap flex items-center gap-1">
                    <Sparkles className="w-3 h-3 text-[#2a9d8f]" /> Test Voice:
                  </span>
                  {[
                    'Hello, how can I help you?',
                    'Where is the hospital?',
                    'Please give me water and food',
                    'Thank you very much',
                  ].map((phrase) => (
                    <button
                      key={phrase}
                      type="button"
                      onClick={() => triggerSpeechTranscript(phrase)}
                      className="bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-md text-[11px] font-medium whitespace-nowrap transition-all border border-slate-200"
                    >
                      "{phrase}"
                    </button>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  onClick={() => setIsMicOn((prev) => !prev)}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-sm text-white shadow-md transition-all ${
                    isMicOn ? 'bg-[#2a9d8f] hover:bg-[#21867a]' : 'bg-slate-500 hover:bg-slate-600'
                  }`}
                >
                  {isMicOn ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
                  {isMicOn ? 'Mic ON' : 'Mic OFF'}
                </button>

                <button
                  onClick={() => setIsGestureEnabled((prev) => !prev)}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-sm text-white shadow-md transition-all ${
                    isGestureEnabled ? 'bg-[#2b7fff] hover:bg-[#1f6feb]' : 'bg-slate-500 hover:bg-slate-600'
                  }`}
                >
                  <Hand className="w-4 h-4" />
                  {isGestureEnabled ? 'Enable Gesture' : 'Gesture OFF'}
                </button>

                <button
                  onClick={() => setIsSignToVoiceEnabled((prev) => !prev)}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-sm text-white shadow-md transition-all ${
                    isSignToVoiceEnabled ? 'bg-[#8338ec] hover:bg-[#7226db]' : 'bg-slate-500 hover:bg-slate-600'
                  }`}
                >
                  <Volume2 className="w-4 h-4" />
                  {isSignToVoiceEnabled ? 'Enable Sign → Voice' : 'Voice Output OFF'}
                </button>

                <button
                  onClick={() => setIsCameraOn((prev) => !prev)}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-sm text-white shadow-md transition-all ${
                    isCameraOn ? 'bg-[#f4a261] hover:bg-[#e76f51]' : 'bg-slate-500 hover:bg-slate-600'
                  }`}
                >
                  {isCameraOn ? <Camera className="w-4 h-4" /> : <CameraOff className="w-4 h-4" />}
                  {isCameraOn ? 'Camera ON' : 'Camera OFF'}
                </button>
              </div>
            </section>

            {/* Right Column: 3D Sign Avatar */}
            <section className="flex flex-col gap-4">
              <h2 className="text-xl font-bold text-slate-900">Sign Avatar</h2>
              <SignAvatar currentSign={avatarSign} textToSign={speechTranscriptForAvatar} className="w-full" />
            </section>
          </main>
        </div>
      ) : (
        /* 2. ADVANCED STUDIO WORKSPACE */
        <div className="flex-1 bg-gray-950 text-gray-100 flex flex-col">
          <Header
            systemState={systemState}
            isConnected={isConnected}
            activeLanguage={activeLanguage}
            onLanguageChange={setActiveLanguage}
            latencySummary={latencyMetrics}
            onOpenMetrics={() => setIsMetricsOpen(true)}
            onOpenPresets={() => setIsPresetsOpen(true)}
          />

          <main className="flex-1 container mx-auto px-4 py-4 max-w-7xl flex flex-col gap-4">
            {/* Studio Workspace Tab Navigation */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-gray-900/60 p-2.5 rounded-2xl border border-gray-800">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setStudioTab('recognition')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition shadow-sm ${
                    studioTab === 'recognition'
                      ? 'bg-[#2b7fff] text-white shadow-blue-500/20'
                      : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                  }`}
                >
                  <Video className="w-4 h-4" /> Live Recognition & Chat
                </button>

                <button
                  type="button"
                  onClick={() => setStudioTab('collection')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition shadow-sm ${
                    studioTab === 'collection'
                      ? 'bg-amber-500 text-slate-950 shadow-amber-500/20'
                      : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                  }`}
                >
                  <Database className="w-4 h-4" /> Training Data Collection Studio
                </button>
              </div>

              <div className="text-xs text-gray-400 font-mono hidden md:inline">
                {studioTab === 'recognition'
                  ? 'Real-Time Full-Duplex Vision & Speech Stream'
                  : '30-Frame ISLRTC Landmark Sequence Recorder'}
              </div>
            </div>

            {studioTab === 'recognition' ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="flex flex-col gap-4">
                  <LiveCameraView
                    onSignRecognised={handleRecognizedSignFromCamera}
                    onRawFrame={setStudioRawFrame}
                    sessionId="main-room"
                    isActive={true}
                  />
                  <AudioControls
                    activeLanguage={activeLanguage}
                    partialSpeech={partialSpeech}
                    onSendSpeech={(text) => {
                      triggerSpeechTranscript(text);
                    }}
                    onSendText={(text) => {
                      triggerSpeechTranscript(text);
                      sendTextMessage(text);
                    }}
                  />
                </div>

                <div className="flex flex-col gap-4">
                  <SignAvatar currentSign={avatarSign} textToSign={speechTranscriptForAvatar} />
                  <ConversationThread
                    messages={messages}
                    activeLanguage={activeLanguage}
                    onClear={clearSession}
                  />
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                <div className="lg:col-span-5 flex flex-col gap-4">
                  <LiveCameraView
                    onSignRecognised={handleRecognizedSignFromCamera}
                    onRawFrame={setStudioRawFrame}
                    sessionId="main-room"
                    isActive={true}
                  />
                </div>
                <div className="lg:col-span-7 flex flex-col gap-4">
                  <DataCollectionStudio rawFrame={studioRawFrame} sessionId="studio_collector" />
                </div>
              </div>
            )}
          </main>
        </div>
      )}

      {/* Metrics Modal */}
      <MetricsDashboard
        isOpen={isMetricsOpen}
        onClose={() => setIsMetricsOpen(false)}
        latencyData={latencyMetrics}
      />

      {/* Demo Controls */}
      <DemoControls
        isOpen={isPresetsOpen}
        onClose={() => setIsPresetsOpen(false)}
        onRunScenarioStep={(speaker, sign, text) => {
          if (speaker === 'isl' && sign) {
            setAvatarSign(sign);
            handleRecognizedSignFromCamera({
              gesture: sign,
              displayName: sign,
              confidence: 0.95,
              probabilities: [{ sign, confidence: 0.95, label: sign }],
              timestamp: Date.now(),
              sequenceLength: 30,
              latencyMs: 3.5,
              modelId: 'SBV-ML-TEMPORAL-v1',
            });
          } else if (speaker === 'speech' && text) {
            triggerSpeechTranscript(text);
          }
        }}
      />
    </div>
  );
}
