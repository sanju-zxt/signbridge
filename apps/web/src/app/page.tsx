'use client';
/**
 * page.tsx — SignBridge Real-Time Multimodal ISL AI Communication Platform
 * Pixel-inspired implementation adhering to the official SaaS dashboard specification.
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { SignAvatar } from '../components/avatar/SignAvatar';
import LandmarkOverlay from '../components/vision/LandmarkOverlay';
import { OptimizedCameraView, RecognitionState } from '../components/vision/OptimizedCameraView';
import DataCollectionStudio from '../components/studio/DataCollectionStudio';
import { MetricsDashboard } from '../components/MetricsDashboard';
import { useRealtimeSession } from '../hooks/useRealtimeSession';
import { StreamingSTTClient } from '../lib/speech/stt';
import { ttsPlayer } from '../lib/speech/tts';
import { matchSpeechSequence } from '../lib/speech/phraseMatcher';
import { SentenceDemoMode } from '../components/demo/SentenceDemoMode';
import {
  ISL_VOCABULARY,
  translateISLGlossSequenceToEnglish,
  RecognitionResult,
  RawLandmarkFrame,
  SupportedLanguage,
} from '@signbridge/types';
import {
  LayoutDashboard,
  Video,
  Clock,
  Settings,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Camera,
  CameraOff,
  Send,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ShieldCheck,
  Eye,
  RefreshCw,
  Copy,
  Check,
  Sliders,
  Database,
  User,
  Hand,
  Play,
  Pause,
  Zap,
  Activity,
  RotateCcw,
} from 'lucide-react';

// ── Landmark Normalization and Feature Extraction Helpers ───────────────────
// ── Motion-Preserving Landmark Normalization & 30-Frame Resampling ───────────
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
  }>,
  winSize = 30
): Float32Array {
  const matrix = new Float32Array(winSize * 126);
  if (!frames || frames.length === 0) return matrix;

  const N = frames.length;
  const resampled: Array<{
    rightHand: Array<{ x: number; y: number; z: number }> | null;
    leftHand: Array<{ x: number; y: number; z: number }> | null;
  }> = [];

  for (let k = 0; k < winSize; k++) {
    const progress = winSize > 1 ? k / (winSize - 1) : 0;
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

  // Find reference initial right hand wrist & palm scale
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

  // Find reference initial left hand wrist & palm scale
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

  for (let step = 0; step < winSize; step++) {
    const f = resampled[step];
    const base = step * 126;

    // Right hand (dims 0..62)
    if (f.rightHand && f.rightHand.length >= 21 && refRh0) {
      const wt = f.rightHand[0];
      const dxt = f.rightHand[9].x - wt.x;
      const dyt = f.rightHand[9].y - wt.y;
      const dzt = f.rightHand[9].z - wt.z;
      const curScale = Math.sqrt(dxt * dxt + dyt * dyt + dzt * dzt) || refRhScale;

      // Joint 0: Wrist motion displacement relative to sequence initial frame
      matrix[base + 0] = (wt.x - refRh0.x) / refRhScale;
      matrix[base + 1] = (wt.y - refRh0.y) / refRhScale;
      matrix[base + 2] = (wt.z - refRh0.z) / refRhScale;

      // Joints 1..20: Skeletal configuration relative to instantaneous wrist
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

      // Joint 0: Wrist motion displacement relative to sequence initial frame
      matrix[base + 63 + 0] = (wt.x - refLh0.x) / refLhScale;
      matrix[base + 63 + 1] = (wt.y - refLh0.y) / refLhScale;
      matrix[base + 63 + 2] = (wt.z - refLh0.z) / refLhScale;

      // Joints 1..20: Skeletal configuration relative to instantaneous wrist
      for (let i = 1; i < 21; i++) {
        matrix[base + 63 + i * 3 + 0] = (f.leftHand[i].x - wt.x) / curScale;
        matrix[base + 63 + i * 3 + 1] = (f.leftHand[i].y - wt.y) / curScale;
        matrix[base + 63 + i * 3 + 2] = (f.leftHand[i].z - wt.z) / curScale;
      }
    }
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
  const effectiveX = totalDeltaX * 4.0 > 0.16 ? (Math.abs(netX) > 0.04 ? netX : 0.08) : netX;
  return [effectiveX, netY, netZ];
}

// ── 6 Canonical Quick Signs as Defined in Reference Specification ───────────
const QUICK_SIGNS = [
  { label: 'Hello', id: 'HELLO', englishText: 'Hello!' },
  { label: 'You', id: 'YOU', englishText: 'You.' },
  { label: 'Help', id: 'HELP', englishText: 'Help me.' },
  { label: 'What', id: 'WHAT', englishText: 'What?' },
  { label: 'Water', id: 'WATER', englishText: 'Water, please.' },
  { label: 'Thank you', id: 'THANK_YOU', englishText: 'Thank you.' },
];

export default function SignBridgeDashboard() {
  // Navigation & View State
  const [navSection, setNavSection] = useState<'dashboard' | 'meeting' | 'history' | 'settings' | 'demo'>('dashboard');
  const [meetingId] = useState<string>('SBV-12345');
  const [copiedMeeting, setCopiedMeeting] = useState<boolean>(false);

  // Control Toggles
  const [isCameraOn, setIsCameraOn] = useState<boolean>(true);
  const [isMicOn, setIsMicOn] = useState<boolean>(true);
  const [isSignToVoiceEnabled, setIsSignToVoiceEnabled] = useState<boolean>(true);
  const [isSubtitlesOn, setIsSubtitlesOn] = useState<boolean>(true);
  const [translationDirection, setTranslationDirection] = useState<'isl_to_speech' | 'speech_to_isl'>('isl_to_speech');
  const [accessibilityMode, setAccessibilityMode] = useState<boolean>(false);

  // Recognition & Perception State
  const [recognizedSign, setRecognizedSign] = useState<string | null>(null);
  const [recognizedConfidence, setRecognizedConfidence] = useState<number>(0);
  const [handsDetected, setHandsDetected] = useState<number>(0);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(true);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Transcription & Conversation State
  const [liveSubtitle, setLiveSubtitle] = useState<string>('Waiting for signs. Perform gestures in camera to build sentences.');
  const [transcriptionTimestamp, setTranscriptionTimestamp] = useState<string>('Ready');
  const [typedText, setTypedText] = useState<string>('');
  const [sttError, setSttError] = useState<string | null>(null);
  const [speechStatus, setSpeechStatus] = useState<'idle' | 'listening' | 'interim' | 'final'>('idle');
  const [historyLog, setHistoryLog] = useState<Array<{ id: string; time: string; text: string; source: 'isl' | 'speech' | 'text'; sign?: string }>>([]);

  // 3D Avatar State & Controls
  const [avatarSign, setAvatarSign] = useState<string | null>('HELLO');
  const [speechTranscriptForAvatar, setSpeechTranscriptForAvatar] = useState<string | null>(null);
  const [avatarStatus, setAvatarStatus] = useState<{ activeAction: string; isPaused: boolean; isGlbLoaded: boolean; queuedActions?: string[] }>({
    activeAction: 'HELLO',
    isPaused: false,
    isGlbLoaded: false,
    queuedActions: [],
  });
  const avatarPerformSignRef = useRef<((signId: string) => void) | null>(null);
  const avatarPerformSequenceRef = useRef<((signIds: string[]) => void) | null>(null);
  const avatarReplayRef = useRef<(() => boolean) | null>(null);
  const [lastExecutedSentence, setLastExecutedSentence] = useState<string | null>(null);

  // Camera & Adaptive Recognition State
  const [candidateSign, setCandidateSign] = useState<{ sign: string; label: string; confidence: number } | null>(null);
  const [recognitionState, setRecognitionState] = useState<RecognitionState>('IDLE');
  const [isRecognizedFlash, setIsRecognizedFlash] = useState<boolean>(false);
  const [recognizedAt, setRecognizedAt] = useState<number | null>(null);

  // Continuous Sentence Buffer & Controls
  const pauseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const accumulatedGlossesRef = useRef<string[]>([]);
  const [accumulatedGlosses, setAccumulatedGlosses] = useState<string[]>([]);
  const [topProbabilities, setTopProbabilities] = useState<Array<{ sign: string; label: string; confidence: number }>>([]);
  const [isSigningPaused, setIsSigningPaused] = useState<boolean>(false);
  const speechStartRef = useRef<number>(0);
  const sttClientRef = useRef<StreamingSTTClient | null>(null);

  // Settings configuration
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.48);
  const [ttsSpeed, setTtsSpeed] = useState<number>(1.0);

  // Realtime Session Hook
  const {
    messages,
    systemState,
    isConnected,
    activeLanguage,
    setActiveLanguage,
    sendFinalISLSign,
    sendFinalSpeech,
    sendTextMessage,
  } = useRealtimeSession('main-room', 'isl_signer');

  // Format current timestamp helper
  const getCurrentFormattedTime = useCallback(() => {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }, []);

  // Clear Sentence Callback
  const clearCurrentSentence = useCallback(() => {
    if (pauseTimerRef.current) {
      clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }
    accumulatedGlossesRef.current = [];
    setAccumulatedGlosses([]);
    setLiveSubtitle('Sentence cleared. Ready for next sign.');
  }, []);

  // Repeat Avatar Message Callback
  const repeatAvatarMessage = useCallback(() => {
    if (lastExecutedSentence) {
      triggerSpeechTranscript(lastExecutedSentence);
    } else if (accumulatedGlosses.length > 0 && avatarPerformSequenceRef.current) {
      avatarPerformSequenceRef.current(accumulatedGlosses);
    } else if (avatarSign && avatarPerformSignRef.current) {
      avatarPerformSignRef.current(avatarSign);
    }
  }, [lastExecutedSentence, accumulatedGlosses, avatarSign]);

  // Finalize Continuous ISL Sentence
  const finalizeCurrentSentence = useCallback(() => {
    if (pauseTimerRef.current) {
      clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }

    const glosses = [...accumulatedGlossesRef.current];
    if (glosses.length === 0) {
      return;
    }

    accumulatedGlossesRef.current = [];
    setAccumulatedGlosses([]);

    const englishSentence = translateISLGlossSequenceToEnglish(glosses);
    if (!englishSentence) return;
    setLastExecutedSentence(englishSentence);

    const timeStr = getCurrentFormattedTime();
    setLiveSubtitle(englishSentence);
    setTranscriptionTimestamp(timeStr);

    setHistoryLog((prev) => [
      { id: `hist_${Date.now()}`, time: timeStr, text: englishSentence, source: 'isl', sign: glosses.join(' ') },
      ...prev,
    ]);

    if (isSignToVoiceEnabled) {
      ttsPlayer.speak(englishSentence, activeLanguage);
    }

    sendFinalISLSign({
      id: `isl_seq_${Date.now()}`,
      sessionId: 'main-room',
      timestamp: Date.now(),
      sign: glosses.join(' '),
      label: englishSentence,
      confidence: 0.95,
      alternatives: [],
      modelVersion: 'SIGNBRIDGE-ISL-v8',
      latencyMs: 18.2,
      isDebounced: true,
      status: 'final',
    });
  }, [isSignToVoiceEnabled, activeLanguage, sendFinalISLSign, getCurrentFormattedTime]);

  const finalizeSentenceRef = useRef(finalizeCurrentSentence);
  useEffect(() => {
    finalizeSentenceRef.current = finalizeCurrentSentence;
  }, [finalizeCurrentSentence]);

  const finishSentenceRecognition = useCallback(() => {
    if (pauseTimerRef.current) {
      clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }
    finalizeSentenceRef.current();
  }, []);

  // Handle Confirmed Vision Recognition Result from Camera
  const handleRecognizedSignFromCamera = useCallback(
    (result: RecognitionResult) => {
      if (isSigningPaused) return;
      if (!result.gesture || result.gesture === 'NONE' || result.gesture === 'NONE_RESTING' || result.gesture === 'UNKNOWN' || result.confidence < confidenceThreshold) {
        return;
      }

      const sign = result.gesture;

      if (pauseTimerRef.current) {
        clearTimeout(pauseTimerRef.current);
      }
      pauseTimerRef.current = setTimeout(() => {
        finalizeSentenceRef.current();
      }, 2500);

      // Append verified sign to continuous sentence buffer
      accumulatedGlossesRef.current.push(sign);
      setAccumulatedGlosses([...accumulatedGlossesRef.current]);

      setRecognizedSign(result.displayName);
      setRecognizedConfidence(result.confidence);
      setIsRecognizedFlash(true);
      setRecognizedAt(Date.now());
      setAvatarSign(sign);

      if (avatarPerformSignRef.current) {
        avatarPerformSignRef.current(sign);
      }

      const timeStr = getCurrentFormattedTime();
      setTranscriptionTimestamp(timeStr);
      const partialSentence = translateISLGlossSequenceToEnglish(accumulatedGlossesRef.current);
      setLiveSubtitle(partialSentence ? partialSentence : `[ISL]: ${accumulatedGlossesRef.current.join(' ')}`);

      setTimeout(() => setIsRecognizedFlash(false), 500);
    },
    [confidenceThreshold, isSigningPaused, getCurrentFormattedTime]
  );

  const handleCandidateChanged = useCallback((cand: { sign: string; label: string; confidence: number } | null) => {
    setCandidateSign(cand);
  }, []);

  const handleRecognitionStateChanged = useCallback((state: RecognitionState) => {
    setRecognitionState(state);
    if (state === 'UNKNOWN') {
      setRecognizedSign('UNKNOWN');
      setRecognizedConfidence(0.40);
    }
  }, []);

  const handleTopProbabilitiesChanged = useCallback((probs: Array<{ sign: string; label: string; confidence: number }>) => {
    setTopProbabilities(probs);
  }, []);

  const handleRecognizedSignRef = useRef(handleRecognizedSignFromCamera);
  useEffect(() => {
    handleRecognizedSignRef.current = handleRecognizedSignFromCamera;
  }, [handleRecognizedSignFromCamera]);

  // Handle Speech Recognition Transcript (STT or Typed Message)
  const triggerSpeechTranscript = useCallback(
    (text: string) => {
      if (!text || !text.trim()) return;
      const cleanText = text.trim();
      const timeStr = getCurrentFormattedTime();

      setLiveSubtitle(cleanText);
      setTranscriptionTimestamp(timeStr);
      setSttError(null);
      setSpeechStatus('final');
      sendFinalSpeech(cleanText, 0.95);
      setSpeechTranscriptForAvatar(cleanText);
      setLastExecutedSentence(cleanText);

      setHistoryLog((prev) => [
        { id: `speech_${Date.now()}`, time: timeStr, text: cleanText, source: 'speech' },
        ...prev,
      ]);

      // Direct avatar full-sentence sequencing
      const matchedTokens = matchSpeechSequence(cleanText);
      if (matchedTokens.length > 0) {
        if (avatarPerformSequenceRef.current) {
          avatarPerformSequenceRef.current(matchedTokens);
        } else if (avatarPerformSignRef.current) {
          avatarPerformSignRef.current(matchedTokens[0]);
        }
        setAvatarSign(matchedTokens[0]);
      }
    },
    [sendFinalSpeech, getCurrentFormattedTime]
  );

  const triggerSpeechRef = useRef(triggerSpeechTranscript);
  useEffect(() => {
    triggerSpeechRef.current = triggerSpeechTranscript;
  }, [triggerSpeechTranscript]);

  // Quick Sign Action Handler - Chained into Continuous ISL Sentence Construction
  const handleTriggerQuickSign = (signId: string, englishText: string) => {
    if (avatarPerformSignRef.current) {
      avatarPerformSignRef.current(signId);
    }
    setAvatarSign(signId);
    setRecognizedSign(signId);
    setRecognizedConfidence(0.98);

    // Append to continuous sentence buffer so quick signs chain into complete sentences
    accumulatedGlossesRef.current.push(signId);
    const updatedGlosses = [...accumulatedGlossesRef.current];
    setAccumulatedGlosses(updatedGlosses);

    const fullSentence = translateISLGlossSequenceToEnglish(updatedGlosses);
    const timeStr = getCurrentFormattedTime();
    setLiveSubtitle(fullSentence || englishText);
    setTranscriptionTimestamp(timeStr);

    if (pauseTimerRef.current) {
      clearTimeout(pauseTimerRef.current);
    }
    pauseTimerRef.current = setTimeout(() => {
      finalizeSentenceRef.current();
    }, 2500);
  };



  // STT Microphone Streaming
  useEffect(() => {
    if (isMicOn) {
      sttClientRef.current = new StreamingSTTClient({
        language: activeLanguage,
        onStateChange: (listening) => {
          setSpeechStatus(listening ? 'listening' : 'idle');
        },
        onPartial: (partial) => {
          if (!speechStartRef.current) speechStartRef.current = Date.now();
          setSpeechStatus('interim');
          setLiveSubtitle(partial);
          setSttError(null);
        },
        onFinal: (finalText) => {
          speechStartRef.current = 0;
          setSpeechStatus('final');
          triggerSpeechRef.current(finalText);
        },
        onError: (err) => {
          setSttError(err);
          setSpeechStatus('idle');
        },
      });
      sttClientRef.current.start();
    } else {
      sttClientRef.current?.stop();
      setSpeechStatus('idle');
    }

    return () => {
      sttClientRef.current?.stop();
    };
  }, [isMicOn, activeLanguage]);

  // Handle Send Typed Text Message
  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    if (typedText.trim()) {
      triggerSpeechTranscript(typedText.trim());
      setTypedText('');
    }
  };

  // Copy Meeting ID helper
  const handleCopyMeetingId = () => {
    navigator.clipboard.writeText(meetingId);
    setCopiedMeeting(true);
    setTimeout(() => setCopiedMeeting(false), 2000);
  };

  return (
    <div className={`min-h-screen flex flex-col font-sans transition-colors ${accessibilityMode ? 'bg-white text-black' : 'bg-[#F7F9FC] text-[#14264F]'}`}>
      {/* ── TOP NAVIGATION (Height ~72px) ─────────────────────────────────── */}
      <header className="h-[72px] bg-white border-b border-[#DCE5EF] px-6 sm:px-8 flex items-center justify-between shrink-0 shadow-sm z-30">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-black text-[#14264F] tracking-tight">SignBridge</span>
          <span className="hidden sm:inline-flex items-center bg-[#E7F4FC] text-[#12B8C2] text-xs font-semibold px-3 py-1 rounded-full border border-[#12B8C2]/20">
            Real-Time Multimodal AI
          </span>
        </div>

        <div className="flex items-center gap-3 sm:gap-6">
          {/* System Online Status */}
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#12B8C2] animate-pulse" />
            <span className="text-xs font-bold uppercase tracking-wider text-[#12B8C2]">System Online</span>
          </div>

          {/* Meeting ID */}
          <div className="hidden md:flex items-center gap-1.5 text-xs font-medium text-[#71809A]">
            <span>Meeting ID:</span>
            <button
              onClick={handleCopyMeetingId}
              className="font-bold text-[#14264F] hover:text-[#12B8C2] transition flex items-center gap-1"
              title="Click to copy Meeting ID"
            >
              <span>{meetingId}</span>
              {copiedMeeting ? <Check className="w-3 h-3 text-[#12B8C2]" /> : <Copy className="w-3 h-3 text-[#71809A]" />}
            </button>
          </div>

          {/* Meeting View Button */}
          <button
            onClick={() => setNavSection((prev) => (prev === 'meeting' ? 'dashboard' : 'meeting'))}
            className={`px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all shadow-sm ${
              navSection === 'meeting'
                ? 'bg-[#12B8C2] text-white shadow-[#12B8C2]/30'
                : 'bg-[#14264F] hover:bg-[#1e3875] text-white'
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            <span>Meeting View</span>
          </button>
        </div>
      </header>

      {/* ── MAIN WORKSPACE CONTAINER ───────────────────────────────────────── */}
      <div className="flex-1 flex flex-col md:flex-row min-h-0">
        {/* ── LEFT SIDEBAR (Width ~220px) ─────────────────────────────────── */}
        <aside className="w-full md:w-[220px] bg-white border-r border-[#DCE5EF] p-4 flex md:flex-col justify-between shrink-0 shadow-sm z-20">
          <nav className="flex md:flex-col gap-1.5 w-full">
            {/* Dashboard Link (Active State) */}
            <button
              onClick={() => setNavSection('dashboard')}
              className={`relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all w-full text-left ${
                navSection === 'dashboard'
                  ? 'bg-[#E7F4FC] text-[#14264F] shadow-sm'
                  : 'text-[#71809A] hover:bg-[#F7F9FC] hover:text-[#14264F]'
              }`}
            >
              {navSection === 'dashboard' && (
                <span className="absolute left-0 top-2 bottom-2 w-1 bg-[#12B8C2] rounded-r-full" />
              )}
              <LayoutDashboard className={`w-4 h-4 ${navSection === 'dashboard' ? 'text-[#12B8C2]' : 'text-[#71809A]'}`} />
              <span>Dashboard</span>
            </button>

            {/* V14 Dedicated Bidirectional Sentence Demo */}
            <button
              onClick={() => setNavSection('demo')}
              className={`relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all w-full text-left ${
                navSection === 'demo'
                  ? 'bg-[#E7F4FC] text-[#14264F] shadow-sm'
                  : 'text-[#71809A] hover:bg-[#F7F9FC] hover:text-[#14264F]'
              }`}
            >
              {navSection === 'demo' && (
                <span className="absolute left-0 top-2 bottom-2 w-1 bg-[#12B8C2] rounded-r-full" />
              )}
              <Sparkles className={`w-4 h-4 ${navSection === 'demo' ? 'text-[#12B8C2]' : 'text-[#71809A]'}`} />
              <div className="flex items-center justify-between flex-1">
                <span>Sentence Demo</span>
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-[#12B8C2] text-white">
                  V14
                </span>
              </div>
            </button>

            {/* Meeting Link */}
            <button
              onClick={() => setNavSection('meeting')}
              className={`relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all w-full text-left ${
                navSection === 'meeting'
                  ? 'bg-[#E7F4FC] text-[#14264F] shadow-sm'
                  : 'text-[#71809A] hover:bg-[#F7F9FC] hover:text-[#14264F]'
              }`}
            >
              {navSection === 'meeting' && (
                <span className="absolute left-0 top-2 bottom-2 w-1 bg-[#12B8C2] rounded-r-full" />
              )}
              <Video className={`w-4 h-4 ${navSection === 'meeting' ? 'text-[#12B8C2]' : 'text-[#71809A]'}`} />
              <span>Meeting</span>
            </button>

            {/* History Link */}
            <button
              onClick={() => setNavSection('history')}
              className={`relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all w-full text-left ${
                navSection === 'history'
                  ? 'bg-[#E7F4FC] text-[#14264F] shadow-sm'
                  : 'text-[#71809A] hover:bg-[#F7F9FC] hover:text-[#14264F]'
              }`}
            >
              {navSection === 'history' && (
                <span className="absolute left-0 top-2 bottom-2 w-1 bg-[#12B8C2] rounded-r-full" />
              )}
              <Clock className={`w-4 h-4 ${navSection === 'history' ? 'text-[#12B8C2]' : 'text-[#71809A]'}`} />
              <span>History</span>
            </button>

            {/* Settings Link */}
            <button
              onClick={() => setNavSection('settings')}
              className={`relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all w-full text-left ${
                navSection === 'settings'
                  ? 'bg-[#E7F4FC] text-[#14264F] shadow-sm'
                  : 'text-[#71809A] hover:bg-[#F7F9FC] hover:text-[#14264F]'
              }`}
            >
              {navSection === 'settings' && (
                <span className="absolute left-0 top-2 bottom-2 w-1 bg-[#12B8C2] rounded-r-full" />
              )}
              <Settings className={`w-4 h-4 ${navSection === 'settings' ? 'text-[#12B8C2]' : 'text-[#71809A]'}`} />
              <span>Settings</span>
            </button>
          </nav>

          {/* Model Status Card on Sidebar Bottom */}
          <div className="hidden md:block p-3 rounded-xl bg-[#F7F9FC] border border-[#DCE5EF] text-xs">
            <div className="flex items-center justify-between text-[#71809A] mb-1">
              <span className="font-semibold text-[11px] uppercase tracking-wide">GRU V1 Engine</span>
              <span className="text-[10px] text-[#12B8C2] font-bold">97.5%</span>
            </div>
            <p className="text-[11px] text-[#71809A] leading-relaxed">
              50 ISL classes • 30-frame temporal buffer • Unseen signer calibrated
            </p>
          </div>
        </aside>

        {/* ── MAIN CONTENT VIEWPORT ─────────────────────────────────────────── */}
        <main className="flex-1 p-3.5 sm:p-4 lg:p-5 overflow-y-auto max-w-[1440px] mx-auto w-full flex flex-col gap-3.5">
          {navSection === 'dashboard' && (
            <>
              {/* ── RESPONSIVE TWO-COLUMN WORKSPACE: 62%–66% Camera, 34%–38% Recognition & Avatar ─── */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
                {/* ── LEFT COLUMN: LARGE CAMERA VIEWPORT (62%–66% Width on Desktop) ─── */}
                <div className="lg:col-span-7 xl:col-span-8 flex flex-col gap-3.5">
                  <OptimizedCameraView
                    isCameraOn={isCameraOn}
                    onToggleCamera={() => setIsCameraOn((prev) => !prev)}
                    confidenceThreshold={confidenceThreshold}
                    onSignConfirmed={handleRecognizedSignFromCamera}
                    onCandidateChanged={handleCandidateChanged}
                    onRecognitionStateChanged={handleRecognitionStateChanged}
                    onTopProbabilitiesChanged={handleTopProbabilitiesChanged}
                    isSigningPaused={isSigningPaused}
                    onToggleSigningPaused={() => setIsSigningPaused((prev) => !prev)}
                    className="w-full min-h-[440px] xl:min-h-[500px]"
                  />

                  {/* CARD: LIVE TRANSCRIPTION */}
                  <div className="bg-white rounded-2xl border border-[#DCE5EF] p-3.5 sm:p-4 shadow-sm flex flex-col justify-between min-h-[96px]">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[11px] font-bold tracking-wider text-[#71809A] uppercase">
                          Live Transcription
                        </span>
                        {speechStatus === 'interim' && (
                          <span className="text-[9px] text-amber-600 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full font-semibold animate-pulse">
                            Transcribing…
                          </span>
                        )}
                      </div>

                      {/* Transcribed Text */}
                      <p className="text-base sm:text-lg font-medium text-[#14264F] leading-snug">
                        {isSubtitlesOn ? liveSubtitle : 'Subtitles are turned off. Enable them below.'}
                      </p>
                    </div>

                    {/* Timestamp at bottom right */}
                    <div className="flex items-center justify-between text-[10px] text-[#71809A] pt-2 border-t border-[#DCE5EF]/60 mt-2">
                      <span>Indian Sign Language ↔ English</span>
                      <span className="font-mono">{transcriptionTimestamp}</span>
                    </div>
                  </div>
                </div>

                {/* ── RIGHT COLUMN: RECOGNITION, 3D AVATAR & QUICK ACTIONS (34%–38% Width on Desktop) ─── */}
                <div className="lg:col-span-5 xl:col-span-4 flex flex-col gap-3.5">
                  {/* CARD 1: GESTURE RECOGNITION & CONTINUOUS ISL ENGINE */}
                  <div className="bg-white rounded-2xl border border-[#DCE5EF] p-4 shadow-sm flex flex-col gap-3">
                    {/* Header with 5-State Status Badge and Pause/Resume Tracking */}
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-bold tracking-wider text-[#71809A] uppercase flex items-center gap-1.5">
                        <Activity className="w-3.5 h-3.5 text-[#12B8C2]" />
                        ISL Recognition Engine
                      </span>
                      <button
                        type="button"
                        onClick={() => setIsSigningPaused(!isSigningPaused)}
                        className={`text-[10px] font-bold px-2.5 py-1 rounded-full border transition-colors flex items-center gap-1 ${
                          isSigningPaused
                            ? 'bg-amber-100 text-amber-800 border-amber-300'
                            : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        }`}
                        title="Pause or resume continuous sign recognition"
                      >
                        {isSigningPaused ? <Play className="w-2.5 h-2.5" /> : <Pause className="w-2.5 h-2.5" />}
                        <span>{isSigningPaused ? 'Paused' : 'Active Tracking'}</span>
                      </button>
                    </div>

                    {/* Prominent Confirmed Sign / State Presentation */}
                    <div className="p-3.5 rounded-xl bg-[#F7F9FC] border border-[#DCE5EF]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-bold text-[#71809A] uppercase tracking-wider">
                          Current Sign
                        </span>
                        {/* 5-State Pill Badge */}
                        {recognitionState === 'CONFIRMED' && (
                          <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-bold text-[10px] flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" /> Confirmed
                          </span>
                        )}
                        {recognitionState === 'CANDIDATE' && (
                          <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-bold text-[10px] flex items-center gap-1">
                            <Activity className="w-3 h-3 animate-spin" /> Evaluating
                          </span>
                        )}
                        {recognitionState === 'ANALYZING' && (
                          <span className="px-2 py-0.5 rounded-full bg-cyan-100 text-cyan-700 font-bold text-[10px] flex items-center gap-1">
                            <Sparkles className="w-3 h-3" /> Analyzing
                          </span>
                        )}
                        {recognitionState === 'HAND_DETECTED' && (
                          <span className="px-2 py-0.5 rounded-full bg-slate-200 text-slate-700 font-bold text-[10px] flex items-center gap-1">
                            <Hand className="w-3 h-3" /> Hand Detected
                          </span>
                        )}
                        {recognitionState === 'UNKNOWN' && (
                          <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 font-bold text-[10px] flex items-center gap-1">
                            <AlertCircle className="w-3 h-3" /> Unknown Sign
                          </span>
                        )}
                        {recognitionState === 'IDLE' && (
                          <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 font-medium text-[10px]">
                            Waiting for Hand
                          </span>
                        )}
                      </div>

                      {/* Big Confirmed Sign Text */}
                      <div className="my-1">
                        {recognitionState === 'UNKNOWN' ? (
                          <span className="text-2xl sm:text-3xl font-extrabold text-rose-600 tracking-tight block">
                            UNKNOWN SIGN
                          </span>
                        ) : (
                          <span className="text-2xl sm:text-3xl font-extrabold text-[#14264F] tracking-tight block truncate">
                            {recognizedSign || (isSigningPaused ? 'PAUSED' : 'WAITING FOR SIGN')}
                          </span>
                        )}
                      </div>

                      {/* Candidate Callout if in Candidate State */}
                      {recognitionState === 'CANDIDATE' && candidateSign && (
                        <div className="mt-1 mb-2 px-2.5 py-1 rounded-lg bg-amber-50/80 border border-dashed border-amber-300 text-[11px] text-amber-800 flex items-center justify-between">
                          <span className="font-semibold">Candidate: {candidateSign.label}</span>
                          <span className="font-mono text-[10px]">Evaluating stability ({Math.round(candidateSign.confidence * 100)}%)</span>
                        </div>
                      )}

                      {/* Progress Bar & Confidence Gauge */}
                      <div className="w-full h-2 rounded-full bg-[#E7F4FC] overflow-hidden my-1.5">
                        <div
                          className={`h-full rounded-full transition-all duration-200 ${
                            recognitionState === 'CONFIRMED'
                              ? 'bg-emerald-500'
                              : recognitionState === 'CANDIDATE'
                              ? 'bg-amber-500'
                              : recognitionState === 'UNKNOWN'
                              ? 'bg-rose-400'
                              : 'bg-[#12B8C2]'
                          }`}
                          style={{
                            width: recognitionState === 'UNKNOWN'
                              ? '100%'
                              : recognizedSign
                              ? `${Math.max(15, Math.round(recognizedConfidence * 100))}%`
                              : candidateSign
                              ? `${Math.round(candidateSign.confidence * 100)}%`
                              : '0%',
                          }}
                        />
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-[#71809A]">
                        <span>
                          {recognitionState === 'UNKNOWN'
                            ? 'Pose does not match trained ISL vocabulary.'
                            : recognizedSign
                            ? `Confidence: ${Math.round(recognizedConfidence * 100)}%`
                            : 'Ready for next ISL gesture.'}
                        </span>
                        {recognizedSign && recognitionState !== 'UNKNOWN' && (
                          <span className="text-[#12B8C2] font-semibold text-[10px] flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" /> Validated ISL
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Top-4 Alternative Predictions with clear Candidate Styling */}
                    {topProbabilities.length > 0 && (
                      <div className="flex flex-col gap-1.5 pt-1 border-t border-[#DCE5EF]">
                        <span className="text-[10px] font-bold text-[#71809A] uppercase tracking-wider">
                          Candidate Probabilities
                        </span>
                        <div className="grid grid-cols-2 gap-1.5">
                          {topProbabilities.slice(0, 4).map((p, idx) => (
                            <div
                              key={idx}
                              className={`p-1.5 rounded-lg border text-xs flex items-center justify-between ${
                                idx === 0
                                  ? 'bg-[#E7F4FC] border-[#12B8C2]/40 text-[#14264F] font-semibold'
                                  : 'bg-slate-50 border-slate-200 text-slate-600'
                              }`}
                            >
                              <span className="truncate pr-1">{p.label}</span>
                              <span className="font-mono text-[10px] text-[#12B8C2] font-bold shrink-0">
                                {(p.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Continuous Accumulated Glosses Sentence Buffer */}
                    <div className="bg-[#F8FAFC] rounded-xl p-3 border border-[#E2E8F0]">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[10px] font-bold text-[#475569] uppercase tracking-wider">
                          Sentence Buffer ({accumulatedGlosses.length} {accumulatedGlosses.length === 1 ? 'sign' : 'signs'})
                        </span>
                        {accumulatedGlosses.length > 0 && (
                          <div className="flex items-center gap-1.5">
                            <button
                              type="button"
                              onClick={finishSentenceRecognition}
                              className="text-[10px] px-2 py-0.5 bg-[#12B8C2] hover:bg-[#0E8B93] text-white font-bold rounded-lg transition-colors shadow-xs"
                            >
                              Translate & Speak
                            </button>
                            <button
                              type="button"
                              onClick={clearCurrentSentence}
                              className="text-[10px] px-2 py-0.5 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold rounded-lg transition-colors flex items-center gap-1"
                              title="Clear current sentence buffer"
                            >
                              <RotateCcw className="w-2.5 h-2.5" />
                              <span>Clear</span>
                            </button>
                          </div>
                        )}
                      </div>

                      {accumulatedGlosses.length === 0 ? (
                        <p className="text-[11px] text-[#94A3B8] italic">
                          Perform consecutive signs to construct continuous ISL sentences.
                        </p>
                      ) : (
                        <div className="flex flex-col gap-2">
                          <div className="flex flex-wrap items-center gap-1.5">
                            {accumulatedGlosses.map((gloss, idx) => (
                              <span
                                key={idx}
                                className="inline-flex items-center gap-1 px-2.5 py-1 bg-white border border-[#CBD5E1] rounded-lg text-xs font-bold text-[#14264F] shadow-2xs"
                              >
                                <span className="text-[10px] text-[#94A3B8] font-normal">{idx + 1}.</span>
                                {gloss}
                              </span>
                            ))}
                          </div>

                          <div className="pt-2 border-t border-[#E2E8F0] flex items-center justify-between text-xs bg-white p-2 rounded-lg border border-[#DCE5EF]">
                            <div className="flex items-center gap-1.5 text-[#14264F] font-semibold truncate mr-2">
                              <Volume2 className="w-3.5 h-3.5 text-[#12B8C2] shrink-0" />
                              <span className="truncate">English: "{translateISLGlossSequenceToEnglish(accumulatedGlosses)}"</span>
                            </div>
                            <button
                              type="button"
                              onClick={() => {
                                const sentence = translateISLGlossSequenceToEnglish(accumulatedGlosses);
                                if (sentence) ttsPlayer.speak(sentence, activeLanguage);
                              }}
                              className="text-[10px] px-2 py-0.5 bg-[#E7F4FC] hover:bg-[#12B8C2] text-[#14264F] hover:text-white font-bold rounded transition-colors shrink-0"
                            >
                              Speak Now
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* CARD 2: 3D SIGN AVATAR CARD */}
                  <div className="bg-white rounded-2xl border border-[#DCE5EF] p-3.5 sm:p-4 shadow-sm flex flex-col">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[11px] font-bold tracking-wider text-[#71809A] uppercase">
                        3D Sign Avatar
                      </span>

                      {/* Status Indicator & Replay */}
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={repeatAvatarMessage}
                          className="text-[10px] font-semibold text-[#14264F] hover:text-[#12B8C2] bg-[#E7F4FC] hover:bg-[#d4edfa] px-2 py-0.5 rounded-md transition-colors flex items-center gap-1"
                          title="Repeat last sign or sentence animation"
                        >
                          <RefreshCw className="w-2.5 h-2.5" /> Replay
                        </button>
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-[#12B8C2]">
                          <span className="text-xs">{avatarStatus.isPaused ? 'Paused' : 'Active'}</span>
                          <span className={`w-2 h-2 rounded-full ${avatarStatus.isPaused ? 'bg-amber-400' : 'bg-[#12B8C2]'}`} />
                        </div>
                      </div>
                    </div>

                    {/* Clean 3D Avatar Rendering Container */}
                    <div className="w-full rounded-xl overflow-hidden bg-white">
                      <SignAvatar
                        currentSign={avatarSign}
                        textToSign={speechTranscriptForAvatar}
                        theme="light"
                        showControls={true}
                        showQuickActions={false}
                        onPerformSignReady={(fn) => {
                          avatarPerformSignRef.current = fn;
                        }}
                        onPerformSequenceReady={(fn) => {
                          avatarPerformSequenceRef.current = fn;
                        }}
                        onReplaySequenceReady={(fn) => {
                          avatarReplayRef.current = fn;
                        }}
                        onStatusChange={setAvatarStatus}
                        className="w-full border-0 shadow-none"
                      />
                    </div>
                  </div>

                  {/* CARD 3: QUICK SIGN PANEL (6 Buttons as Specified) ─────── */}
                  <div className="bg-white rounded-2xl border border-[#DCE5EF] p-3 shadow-sm">
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
                      {QUICK_SIGNS.map((item) => (
                        <button
                          key={item.id}
                          type="button"
                          onClick={() => handleTriggerQuickSign(item.id, item.englishText)}
                          className="px-2.5 py-2 rounded-xl bg-white hover:bg-[#E7F4FC] text-[#14264F] hover:text-[#12B8C2] border border-[#DCE5EF] hover:border-[#12B8C2] text-xs font-semibold transition-all shadow-sm active:scale-95 text-center truncate"
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* ── CARD 5.5: V14 DEDICATED BIDIRECTIONAL SENTENCE DEMONSTRATION ─── */}
              <SentenceDemoMode
                onDispatchSequenceToAvatar={(signIds) => {
                  if (avatarPerformSequenceRef.current) {
                    avatarPerformSequenceRef.current(signIds);
                  }
                  if (signIds.length > 0) setAvatarSign(signIds[0]);
                }}
                onReplayAvatarSequence={() => {
                  if (avatarReplayRef.current) return avatarReplayRef.current();
                  return false;
                }}
                onToggleAvatarPause={() => {
                  setAvatarStatus((prev) => ({ ...prev, isPaused: !prev.isPaused }));
                }}
                onResetAvatar={() => {
                  if (avatarPerformSignRef.current) avatarPerformSignRef.current('IDLE');
                  setAvatarSign('IDLE');
                }}
                isAvatarPaused={avatarStatus.isPaused}
                avatarActiveAction={avatarStatus.activeAction}
                avatarQueuedActions={avatarStatus.queuedActions || []}
                recognizedCameraSign={recognizedSign}
                recognizedCameraConfidence={recognizedConfidence}
                accumulatedCameraGlosses={accumulatedGlosses}
                onClearCameraSequence={clearCurrentSentence}
                isCameraActive={isCameraActive && isCameraOn}
                onToggleCamera={() => setIsCameraOn((prev) => !prev)}
              />

              {/* ── CARD 6: VOICE & ACCESSIBILITY FULL-WIDTH PANEL ──────────── */}
              <div className="bg-white rounded-2xl border border-[#DCE5EF] p-3.5 sm:p-4 shadow-sm flex flex-col gap-2.5">
                <span className="text-[11px] font-bold tracking-wider text-[#71809A] uppercase">
                  Voice &amp; Accessibility
                </span>

                <div className="flex flex-wrap items-center gap-3 lg:gap-5 justify-between">
                  {/* 1. Mic On / Off Toggle */}
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setIsMicOn((prev) => !prev)}
                      aria-label="Toggle Microphone"
                      className={`w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center transition-all border shadow-sm ${
                        isMicOn
                          ? 'bg-[#E7F4FC] border-[#12B8C2] text-[#12B8C2]'
                          : 'bg-white border-[#DCE5EF] text-[#71809A]'
                      }`}
                    >
                      {isMicOn ? <Mic className="w-4 h-4 text-[#12B8C2]" /> : <MicOff className="w-4 h-4" />}
                    </button>
                    <div className="flex flex-col text-left">
                      <span className="text-[10px] text-[#71809A] leading-tight">Mic</span>
                      <span className="text-xs font-bold text-[#14264F]">{isMicOn ? 'On' : 'Off'}</span>
                    </div>
                  </div>

                  {/* 2. Text Input & Send Button */}
                  <form onSubmit={handleSendText} className="flex-1 min-w-[240px] flex items-center gap-2">
                    <input
                      type="text"
                      value={typedText}
                      onChange={(e) => setTypedText(e.target.value)}
                      placeholder="Type a message to translate..."
                      className="flex-1 bg-white border border-[#DCE5EF] rounded-xl px-3.5 py-2 text-xs sm:text-sm text-[#14264F] placeholder-[#71809A] focus:outline-none focus:ring-2 focus:ring-[#12B8C2]/40 focus:border-[#12B8C2] transition-all"
                    />
                    <button
                      type="submit"
                      className="bg-[#14264F] hover:bg-[#1e3875] text-white px-4 py-2 rounded-xl text-xs sm:text-sm font-bold flex items-center gap-1.5 transition-all shadow-sm shrink-0"
                    >
                      <Send className="w-3.5 h-3.5" />
                      <span>Send</span>
                    </button>
                  </form>

                  {/* 3. Voice Output On / Off Toggle */}
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setIsSignToVoiceEnabled((prev) => !prev)}
                      aria-label="Toggle Voice Output"
                      className={`w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center transition-all border shadow-sm ${
                        isSignToVoiceEnabled
                          ? 'bg-[#E7F4FC] border-[#12B8C2] text-[#12B8C2]'
                          : 'bg-white border-[#DCE5EF] text-[#71809A]'
                      }`}
                    >
                      {isSignToVoiceEnabled ? (
                        <Volume2 className="w-4 h-4 text-[#12B8C2]" />
                      ) : (
                        <VolumeX className="w-4 h-4" />
                      )}
                    </button>
                    <div className="flex flex-col text-left">
                      <span className="text-[10px] text-[#71809A] leading-tight">Voice output</span>
                      <span className="text-xs font-bold text-[#14264F]">{isSignToVoiceEnabled ? 'On' : 'Off'}</span>
                    </div>
                  </div>

                  {/* 4. Subtitles On / Off Toggle */}
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setIsSubtitlesOn((prev) => !prev)}
                      aria-label="Toggle Subtitles"
                      className={`w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center transition-all border shadow-sm font-bold ${
                        isSubtitlesOn
                          ? 'bg-[#E7F4FC] border-[#12B8C2] text-[#12B8C2]'
                          : 'bg-white border-[#DCE5EF] text-[#71809A]'
                      }`}
                    >
                      <span className="text-xs font-black tracking-tighter">CC</span>
                    </button>
                    <div className="flex flex-col text-left">
                      <span className="text-[10px] text-[#71809A] leading-tight">Subtitles</span>
                      <span className="text-xs font-bold text-[#14264F]">{isSubtitlesOn ? 'On' : 'Off'}</span>
                    </div>
                  </div>

                  {/* 5. Translation Direction Dropdown */}
                  <div className="relative shrink-0">
                    <select
                      value={translationDirection}
                      onChange={(e) => setTranslationDirection(e.target.value as any)}
                      className="appearance-none bg-white border border-[#DCE5EF] rounded-xl pl-3 pr-7 py-2 text-xs font-semibold text-[#14264F] focus:outline-none focus:ring-2 focus:ring-[#12B8C2]/40 focus:border-[#12B8C2] cursor-pointer shadow-sm"
                    >
                      <option value="isl_to_speech">ISL → English</option>
                      <option value="speech_to_isl">English → ISL Avatar</option>
                    </select>
                    <ChevronDown className="w-3.5 h-3.5 text-[#71809A] absolute right-2 top-2.5 pointer-events-none" />
                  </div>
                </div>

                {sttError && (
                  <div className="text-[11px] text-rose-600 bg-rose-50 border border-rose-200 px-3 py-1.5 rounded-lg flex items-center gap-2">
                    <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                    <span>Microphone Notice: {sttError}</span>
                  </div>
                )}
              </div>
            </>
          )}

          {/* ── DEDICATED SENTENCE DEMO VIEW TAB (V14 Full Duplex) ─────────────── */}
          {navSection === 'demo' && (
            <div className="flex-1 flex flex-col gap-4">
              <SentenceDemoMode
                onDispatchSequenceToAvatar={(signIds) => {
                  if (avatarPerformSequenceRef.current) {
                    avatarPerformSequenceRef.current(signIds);
                  }
                  if (signIds.length > 0) setAvatarSign(signIds[0]);
                }}
                onReplayAvatarSequence={() => {
                  if (avatarReplayRef.current) return avatarReplayRef.current();
                  return false;
                }}
                onToggleAvatarPause={() => {
                  setAvatarStatus((prev) => ({ ...prev, isPaused: !prev.isPaused }));
                }}
                onResetAvatar={() => {
                  if (avatarPerformSignRef.current) avatarPerformSignRef.current('IDLE');
                  setAvatarSign('IDLE');
                }}
                isAvatarPaused={avatarStatus.isPaused}
                avatarActiveAction={avatarStatus.activeAction}
                avatarQueuedActions={avatarStatus.queuedActions || []}
                recognizedCameraSign={recognizedSign}
                recognizedCameraConfidence={recognizedConfidence}
                accumulatedCameraGlosses={accumulatedGlosses}
                onClearCameraSequence={clearCurrentSentence}
                isCameraActive={isCameraActive && isCameraOn}
                onToggleCamera={() => setIsCameraOn((prev) => !prev)}
              />

              {/* Side-by-Side Dual Visual Viewports */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1">
                {/* Visual Viewport 1: Live Camera with Skeleton Overlay */}
                <div className="flex flex-col flex-1">
                  <OptimizedCameraView
                    isCameraOn={isCameraOn}
                    onToggleCamera={() => setIsCameraOn((prev) => !prev)}
                    confidenceThreshold={confidenceThreshold}
                    onSignConfirmed={handleRecognizedSignFromCamera}
                    onCandidateChanged={handleCandidateChanged}
                    onRecognitionStateChanged={handleRecognitionStateChanged}
                    onTopProbabilitiesChanged={handleTopProbabilitiesChanged}
                    isSigningPaused={isSigningPaused}
                    onToggleSigningPaused={() => setIsSigningPaused((prev) => !prev)}
                    className="w-full flex-1 min-h-[360px]"
                  />
                </div>

                {/* Visual Viewport 2: 3D Sign Avatar with Skeletal Controller */}
                <div className="bg-white rounded-2xl border border-[#DCE5EF] p-4 shadow-sm flex flex-col">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold tracking-wider text-[#71809A] uppercase flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-[#12B8C2]" /> 3D ISL Avatar Output
                    </span>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-[#E7F4FC] text-[#12B8C2]">
                      {avatarStatus.activeAction === 'IDLE' ? 'Idle' : `Signing: ${avatarStatus.activeAction}`}
                    </span>
                  </div>
                  <div className="w-full rounded-xl overflow-hidden bg-white flex-1 min-h-[360px]">
                    <SignAvatar
                      currentSign={avatarSign}
                      textToSign={speechTranscriptForAvatar}
                      theme="light"
                      showControls={true}
                      showQuickActions={false}
                      onPerformSignReady={(fn) => {
                        avatarPerformSignRef.current = fn;
                      }}
                      onPerformSequenceReady={(fn) => {
                        avatarPerformSequenceRef.current = fn;
                      }}
                      onReplaySequenceReady={(fn) => {
                        avatarReplayRef.current = fn;
                      }}
                      onStatusChange={setAvatarStatus}
                      className="w-full border-0 shadow-none"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── MEETING VIEW TAB (Dedicated Collaborative Mode) ─────────────── */}
          {navSection === 'meeting' && (
            <div className="flex-1 flex flex-col gap-6">
              <div className="bg-white rounded-2xl border border-[#DCE5EF] p-6 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-[#14264F]">SignBridge Collaborative Meeting</h2>
                  <p className="text-sm text-[#71809A]">
                    Room ID: <span className="font-mono font-bold text-[#14264F]">{meetingId}</span> • Full-Duplex ISL ↔ Speech Channel
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleCopyMeetingId}
                    className="px-4 py-2 bg-[#E7F4FC] hover:bg-[#d9effa] text-[#12B8C2] border border-[#12B8C2]/30 text-xs font-bold rounded-xl flex items-center gap-2 transition"
                  >
                    {copiedMeeting ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                    <span>{copiedMeeting ? 'Copied Link' : 'Copy Room Link'}</span>
                  </button>
                  <button
                    onClick={() => setNavSection('dashboard')}
                    className="px-4 py-2 bg-[#14264F] hover:bg-[#1e3875] text-white text-xs font-bold rounded-xl transition"
                  >
                    Back to Dashboard
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1">
                <div className="flex flex-col flex-1">
                  <OptimizedCameraView
                    isCameraOn={isCameraOn}
                    onToggleCamera={() => setIsCameraOn((prev) => !prev)}
                    confidenceThreshold={confidenceThreshold}
                    onSignConfirmed={handleRecognizedSignFromCamera}
                    onCandidateChanged={handleCandidateChanged}
                    onRecognitionStateChanged={handleRecognitionStateChanged}
                    onTopProbabilitiesChanged={handleTopProbabilitiesChanged}
                    isSigningPaused={isSigningPaused}
                    onToggleSigningPaused={() => setIsSigningPaused((prev) => !prev)}
                    className="w-full flex-1 min-h-[380px]"
                  />
                </div>

                <div className="bg-white rounded-2xl border border-[#DCE5EF] p-5 shadow-sm flex flex-col">
                  <span className="text-xs font-bold tracking-wider text-[#71809A] uppercase mb-3">3D Sign Translator</span>
                  <SignAvatar
                    currentSign={avatarSign}
                    textToSign={speechTranscriptForAvatar}
                    theme="light"
                    showControls={true}
                    showQuickActions={true}
                    className="w-full flex-1 border-0 shadow-none"
                  />
                </div>
              </div>
            </div>
          )}

          {/* ── HISTORY VIEW TAB (Conversation Log) ─────────────────────────── */}
          {navSection === 'history' && (
            <div className="bg-white rounded-2xl border border-[#DCE5EF] p-6 shadow-sm flex flex-col gap-4">
              <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-4">
                <div>
                  <h2 className="text-2xl font-bold text-[#14264F]">Session Transcription History</h2>
                  <p className="text-xs text-[#71809A]">Chronological transcript of all bidirectional interpretations.</p>
                </div>
                <button
                  onClick={() => setHistoryLog([])}
                  className="px-3.5 py-1.5 bg-[#F7F9FC] hover:bg-slate-100 text-[#71809A] hover:text-rose-600 border border-[#DCE5EF] rounded-xl text-xs font-semibold transition"
                >
                  Clear History
                </button>
              </div>

              <div className="flex flex-col gap-3 max-h-[540px] overflow-y-auto pr-1">
                {historyLog.length === 0 ? (
                  <p className="text-sm text-[#71809A] py-8 text-center">No conversation records yet in this session.</p>
                ) : (
                  historyLog.map((item) => (
                    <div
                      key={item.id}
                      className="p-4 rounded-xl border border-[#DCE5EF] bg-[#F7F9FC] flex items-start justify-between gap-4"
                    >
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${
                              item.source === 'isl'
                                ? 'bg-[#E7F4FC] text-[#12B8C2]'
                                : 'bg-slate-200 text-[#14264F]'
                            }`}
                          >
                            {item.source === 'isl' ? 'ISL Vision' : 'Spoken Speech'}
                          </span>
                          {item.sign && (
                            <span className="text-[11px] font-mono text-[#71809A]">
                              Sign: {item.sign}
                            </span>
                          )}
                        </div>
                        <p className="text-base font-medium text-[#14264F]">{item.text}</p>
                      </div>
                      <span className="text-xs font-mono text-[#71809A] shrink-0">{item.time}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* ── SETTINGS VIEW TAB ───────────────────────────────────────────── */}
          {navSection === 'settings' && (
            <div className="bg-white rounded-2xl border border-[#DCE5EF] p-6 shadow-sm flex flex-col gap-6">
              <div>
                <h2 className="text-2xl font-bold text-[#14264F]">Platform Settings & Preferences</h2>
                <p className="text-xs text-[#71809A]">Configure perception thresholds, audio synthesis, and accessibility preferences.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Recognition Threshold */}
                <div className="p-5 rounded-xl border border-[#DCE5EF] bg-[#F7F9FC] flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-bold text-[#14264F]">Vision Confidence Threshold</label>
                    <span className="text-xs font-mono font-bold text-[#12B8C2]">{Math.round(confidenceThreshold * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.50"
                    max="0.95"
                    step="0.05"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#12B8C2] cursor-pointer"
                  />
                  <p className="text-[11px] text-[#71809A]">Higher threshold prevents false positives; lower threshold responds faster.</p>
                </div>

                {/* Spoken Language Selector */}
                <div className="p-5 rounded-xl border border-[#DCE5EF] bg-[#F7F9FC] flex flex-col gap-3">
                  <label className="text-sm font-bold text-[#14264F]">Speech Recognition Language</label>
                  <select
                    value={activeLanguage}
                    onChange={(e) => setActiveLanguage(e.target.value as SupportedLanguage)}
                    className="bg-white border border-[#DCE5EF] rounded-xl px-3 py-2 text-xs font-semibold text-[#14264F] focus:outline-none focus:ring-2 focus:ring-[#12B8C2]"
                  >
                    <option value="en">English (India)</option>
                    <option value="hi">Hindi (हिंदी)</option>
                    <option value="kn">Kannada (ಕನ್ನಡ)</option>
                    <option value="te">Telugu (తెలుగు)</option>
                  </select>
                  <p className="text-[11px] text-[#71809A]">Language utilized for speech-to-sign transcription matching.</p>
                </div>

                {/* TTS Speed */}
                <div className="p-5 rounded-xl border border-[#DCE5EF] bg-[#F7F9FC] flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-bold text-[#14264F]">Voice Synthesis Rate</label>
                    <span className="text-xs font-mono font-bold text-[#12B8C2]">{ttsSpeed.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.75"
                    max="1.5"
                    step="0.25"
                    value={ttsSpeed}
                    onChange={(e) => setTtsSpeed(parseFloat(e.target.value))}
                    className="w-full accent-[#12B8C2] cursor-pointer"
                  />
                  <p className="text-[11px] text-[#71809A]">Controls audio speech playback pace.</p>
                </div>

                {/* High Contrast Accessibility Mode */}
                <div className="p-5 rounded-xl border border-[#DCE5EF] bg-[#F7F9FC] flex items-center justify-between gap-4">
                  <div>
                    <span className="text-sm font-bold text-[#14264F] block">High-Contrast Accessibility Mode</span>
                    <span className="text-[11px] text-[#71809A]">Maximizes contrast ratios for visual accessibility.</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setAccessibilityMode((prev) => !prev)}
                    className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      accessibilityMode ? 'bg-[#12B8C2]' : 'bg-[#DCE5EF]'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-md ring-0 transition duration-200 ease-in-out ${
                        accessibilityMode ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* ── FOOTER BAR ─────────────────────────────────────────────────────── */}
      <footer className="h-9 bg-[#E7F4FC] border-t border-[#DCE5EF] px-6 sm:px-8 flex items-center justify-between text-xs text-[#5A7184] font-medium shrink-0">
        <div className="flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-[#12B8C2]" />
          <span>Privacy-first processing</span>
        </div>
        <button
          onClick={() => setAccessibilityMode((prev) => !prev)}
          className="flex items-center gap-1.5 hover:text-[#14264F] transition"
        >
          <Eye className="w-3.5 h-3.5 text-[#12B8C2]" />
          <span>{accessibilityMode ? 'Accessibility mode (Active)' : 'Accessibility mode'}</span>
        </button>
      </footer>
    </div>
  );
}
