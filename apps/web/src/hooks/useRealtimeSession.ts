'use client';

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
