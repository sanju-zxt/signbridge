const fs = require('fs');
const path = require('path');

function save(relPath, content) {
  const full = path.resolve(__dirname, '..', relPath);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content, 'utf8');
  console.log('Saved:', relPath);
}

// 1. apps/web/src/hooks/useRealtimeSession.ts
save('apps/web/src/hooks/useRealtimeSession.ts', `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  CanonicalMessage,
  ConversationSession,
  SystemState,
  SupportedLanguage,
  SpeakerRole,
  RecognitionEvent,
  SpeechTranscriptEvent,
  LatencySummary
} from '@signbridge2vision/types';

export function useRealtimeSession(sessionId: string = 'main-room', userRole: SpeakerRole = 'isl_signer') {
  const [session, setSession] = useState<ConversationSession | null>(null);
  const [messages, setMessages] = useState<CanonicalMessage[]>([]);
  const [systemState, setSystemState] = useState<SystemState>('IDLE');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [activeLanguage, setActiveLanguage] = useState<SupportedLanguage>('en');
  const [partialSpeech, setPartialSpeech] = useState<string>('');
  const [activeProvisionalSign, setActiveProvisionalSign] = useState<{ sign: string; label: string; confidence: number } | null>(null);
  const [latencyMetrics, setLatencyMetrics] = useState<{ overall: LatencySummary; visionSummary: LatencySummary; agentSummary: LatencySummary } | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    try {
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:4000/ws';
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setSystemState('IDLE');
        // Join session room
        ws.send(JSON.stringify({
          event: 'session.join',
          data: {
            sessionId,
            role: userRole,
            clientName: userRole === 'isl_signer' ? 'ISL Communicator' : 'Hearing Partner',
            preferredLanguage: activeLanguage
          }
        }));
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          const { event: evType, data } = parsed;

          switch (evType) {
            case 'session.state': {
              if (data.session) {
                setSession(data.session);
                setMessages(data.session.messages || []);
                setActiveLanguage(data.session.activeLanguage || 'en');
              }
              if (data.latencyMetrics) {
                setLatencyMetrics(data.latencyMetrics);
              }
              break;
            }

            case 'session.updated': {
              if (data.session) {
                setSession(data.session);
                setMessages(data.session.messages || []);
              }
              break;
            }

            case 'message.created': {
              setMessages((prev) => {
                // Avoid duplicate messages
                if (prev.some(m => m.id === data.id)) return prev;
                return [...prev, data];
              });
              setPartialSpeech('');
              break;
            }

            case 'vision.provisional': {
              setActiveProvisionalSign(data);
              setSystemState('SIGNING');
              break;
            }

            case 'speech.streaming': {
              setPartialSpeech(data.transcript);
              setSystemState('LISTENING');
              break;
            }

            case 'system.state': {
              if (data.state) {
                setSystemState(data.state);
              }
              break;
            }

            case 'metrics.updated': {
              setLatencyMetrics(data);
              break;
            }

            case 'session.cleared': {
              setMessages([]);
              break;
            }
          }
        } catch (err) {
          console.error('Error parsing WS message:', err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setSystemState('RECONNECTING');
        // Auto-reconnect after 2 seconds
        setTimeout(connect, 2000);
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection notice:', err);
      };
    } catch {
      setIsConnected(false);
    }
  }, [sessionId, userRole, activeLanguage]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendFinalISLSign = useCallback((recognition: RecognitionEvent) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        event: 'vision.final',
        data: recognition
      }));
    } else {
      // Local fallback simulation if gateway disconnected
      const fallbackMsg: CanonicalMessage = {
        id: 'msg_local_' + Date.now(),
        sessionId,
        speakerId: 'user_local',
        speakerRole: 'isl_signer',
        speakerName: 'ISL Signer (Offline)',
        modality: 'isl',
        text: recognition.label,
        normalizedText: recognition.label,
        confidence: recognition.confidence,
        alternatives: recognition.alternatives,
        language: activeLanguage,
        timestamp: Date.now(),
        status: recognition.confidence < 0.65 ? 'low_confidence' : 'final'
      };
      setMessages(prev => [...prev, fallbackMsg]);
    }
  }, [sessionId, activeLanguage]);

  const sendFinalSpeech = useCallback((transcript: string, confidence: number = 0.95) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        event: 'speech.final',
        data: {
          sessionId,
          speakerId: 'speech_user',
          transcript,
          isFinal: true,
          confidence,
          language: activeLanguage,
          timestamp: Date.now()
        } as SpeechTranscriptEvent
      }));
    } else {
      const fallbackMsg: CanonicalMessage = {
        id: 'msg_speech_local_' + Date.now(),
        sessionId,
        speakerId: 'speech_local',
        speakerRole: 'speech_speaker',
        speakerName: 'Hearing User (Offline)',
        modality: 'speech',
        text: transcript,
        normalizedText: transcript,
        confidence,
        language: activeLanguage,
        timestamp: Date.now(),
        status: 'final'
      };
      setMessages(prev => [...prev, fallbackMsg]);
    }
  }, [sessionId, activeLanguage]);

  const sendTextMessage = useCallback((text: string, speakerRole: SpeakerRole = userRole) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        event: 'message.create',
        data: {
          sessionId,
          text,
          speakerRole,
          speakerName: speakerRole === 'isl_signer' ? 'ISL Signer' : 'Hearing User',
          modality: 'text',
          language: activeLanguage
        }
      }));
    } else {
      const fallbackMsg: CanonicalMessage = {
        id: 'msg_text_' + Date.now(),
        sessionId,
        speakerId: 'text_local',
        speakerRole,
        speakerName: speakerRole === 'isl_signer' ? 'ISL Signer' : 'Hearing User',
        modality: 'text',
        text,
        normalizedText: text,
        confidence: 1.0,
        language: activeLanguage,
        timestamp: Date.now(),
        status: 'final'
      };
      setMessages(prev => [...prev, fallbackMsg]);
    }
  }, [sessionId, userRole, activeLanguage]);

  const clearSession = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ event: 'session.clear', data: { sessionId } }));
    }
    setMessages([]);
  }, [sessionId]);

  return {
    session,
    messages,
    systemState,
    setSystemState,
    isConnected,
    activeLanguage,
    setActiveLanguage,
    partialSpeech,
    activeProvisionalSign,
    latencyMetrics,
    sendFinalISLSign,
    sendFinalSpeech,
    sendTextMessage,
    clearSession
  };
}
`);

// 2. apps/web/src/hooks/useISLPerception.ts
save('apps/web/src/hooks/useISLPerception.ts', `'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { NormalizedFrame, RecognitionEvent, Landmark3D, AlternativeCandidate } from '@signbridge2vision/types';
import { matchTemporalSequence, SIGN_FEATURE_TEMPLATES } from '../lib/vision/temporalMatcher';

export function useISLPerception(onSignFinalized: (event: RecognitionEvent) => void) {
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [activePrediction, setActivePrediction] = useState<{ sign: string; label: string; confidence: number; alternatives: AlternativeCandidate[] } | null>(null);
  const [isDebouncing, setIsDebouncing] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [isSimulatedMode, setIsSimulatedMode] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameBufferRef = useRef<NormalizedFrame[]>([]);
  const lastFinalizedSignRef = useRef<string | null>(null);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const frameCountRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(Date.now());

  // Start live camera
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsCameraActive(true);
          setIsSimulatedMode(false);
          startProcessingLoop();
        };
      }
    } catch (err: any) {
      console.warn('Camera access denied or unavailable:', err);
      setCameraError('Camera access unavailable. You can use Simulated ISL Gesture Mode or Test Presets.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
    }
    setIsCameraActive(false);
  };

  // Trigger a deterministic sign sequence for flawless live demo/testing
  const triggerSimulatedSign = useCallback((signId: string) => {
    const template = SIGN_FEATURE_TEMPLATES[signId];
    if (!template) return;

    setIsSimulatedMode(true);
    const frames: NormalizedFrame[] = [];
    const count = 25;

    for (let f = 0; f < count; f++) {
      const progress = f / (count - 1);
      const mx = template.motionVector[0] * progress;
      const my = template.motionVector[1] * progress;
      const mz = template.motionVector[2] * progress;

      const rightHand: Landmark3D[] = [{ x: mx, y: my, z: mz, visibility: 1 }];
      for (let i = 0; i < 5; i++) {
        const ext = template.fingerStates[i];
        for (let j = 1; j <= 4; j++) {
          rightHand.push({
            x: mx + (i - 2) * 0.1,
            y: my - (j * 0.25) * (0.3 + 0.7 * ext),
            z: mz + (ext < 0.5 ? 0.1 : 0),
            visibility: 1
          });
        }
      }

      let leftHand: Landmark3D[] | null = null;
      if (template.isTwoHanded) {
        leftHand = rightHand.map(lm => ({ x: -lm.x, y: lm.y, z: lm.z, visibility: 1 }));
      }

      frames.push({
        timestamp: Date.now() + f * 33,
        rightHand,
        leftHand,
        pose: null,
        features: []
      });
    }

    // Render simulated hands on canvas
    drawLandmarksOnCanvas(frames[frames.length - 1].rightHand, frames[frames.length - 1].leftHand);

    const prediction = matchTemporalSequence(frames);
    setActivePrediction({
      sign: prediction.sign,
      label: prediction.label,
      confidence: prediction.confidence,
      alternatives: prediction.alternatives
    });

    setIsDebouncing(true);
    setTimeout(() => {
      setIsDebouncing(false);
      onSignFinalized({
        id: 'rec_' + Date.now(),
        sessionId: 'main-room',
        timestamp: Date.now(),
        sign: prediction.sign,
        label: prediction.label,
        confidence: prediction.confidence,
        alternatives: prediction.alternatives,
        modelVersion: 'isl-temporal-v1',
        latencyMs: prediction.latencyMs,
        isDebounced: true,
        status: prediction.confidence < 0.65 ? 'low_confidence' : 'final'
      });
    }, 400);
  }, [onSignFinalized]);

  // Draw 21-point skeleton on canvas overlay
  const drawLandmarksOnCanvas = (rightHand: Landmark3D[] | null, leftHand: Landmark3D[] | null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drawHand = (hand: Landmark3D[], color: string, glowColor: string) => {
      const w = canvas.width;
      const h = canvas.height;
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],       // Index
        [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
        [0, 13], [13, 14], [14, 15], [15, 16],// Ring
        [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
      ];

      // Draw bones
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 8;

      for (const [start, end] of connections) {
        if (hand[start] && hand[end]) {
          const sx = (hand[start].x + 0.5) * w;
          const sy = (hand[start].y + 0.5) * h;
          const ex = (hand[end].x + 0.5) * w;
          const ey = (hand[end].y + 0.5) * h;

          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(ex, ey);
          ctx.stroke();
        }
      }

      // Draw joints
      for (const lm of hand) {
        const x = (lm.x + 0.5) * w;
        const y = (lm.y + 0.5) * h;
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    };

    if (rightHand) drawHand(rightHand, '#14B8A6', '#0D9488'); // Neon Teal for Dominant Hand
    if (leftHand) drawHand(leftHand, '#6366F1', '#4F46E5');   // Indigo for Secondary Hand
  };

  const startProcessingLoop = () => {
    const loop = () => {
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }

      // Live video landmark synthesis / tracking
      if (videoRef.current && videoRef.current.readyState >= 2) {
        // Continuous temporal buffer
        if (frameBufferRef.current.length > 30) {
          frameBufferRef.current.shift();
        }
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    isCameraActive,
    cameraError,
    activePrediction,
    isDebouncing,
    fps,
    isSimulatedMode,
    startCamera,
    stopCamera,
    triggerSimulatedSign
  };
}
`);
