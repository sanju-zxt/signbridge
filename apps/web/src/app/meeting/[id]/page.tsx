'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { SignAvatar } from '../../../components/avatar/SignAvatar';
import LandmarkOverlay from '../../../components/vision/LandmarkOverlay';
import { useISLPerception } from '../../../hooks/useISLPerception';
import { useRealtimeSession } from '../../../hooks/useRealtimeSession';
import { StreamingSTTClient } from '../../../lib/speech/stt';
import { ttsPlayer } from '../../../lib/speech/tts';
import { ISL_VOCABULARY, RecognitionResult, CanonicalMessage } from '@signbridge/types';
import { Mic, MicOff, Hand, Volume2, Camera, CameraOff, Send, Sparkles, AlertCircle } from 'lucide-react';

export default function MeetingPage() {
  const params = useParams();
  const meetingId = (params?.id as string) || '12345';

  const [isMicOn, setIsMicOn] = useState<boolean>(true);
  const [isGestureEnabled, setIsGestureEnabled] = useState<boolean>(true);
  const [isSignToVoiceEnabled, setIsSignToVoiceEnabled] = useState<boolean>(true);
  const [isCameraOn, setIsCameraOn] = useState<boolean>(true);

  const [recognizedSign, setRecognizedSign] = useState<string | null>(null);
  const [recognizedConfidence, setRecognizedConfidence] = useState<number>(0);
  const [liveSubtitle, setLiveSubtitle] = useState<string>('Listening for speech... (speak or type below)');
  const [typedText, setTypedText] = useState<string>('');
  const [sttError, setSttError] = useState<string | null>(null);
  const [isListeningActive, setIsListeningActive] = useState<boolean>(false);
  const [avatarSign, setAvatarSign] = useState<string | null>(null);
  const [speechTranscriptForAvatar, setSpeechTranscriptForAvatar] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const mpRef = useRef<any>(null);
  const [rightLandmarks, setRightLandmarks] = useState<Array<{ x: number; y: number; z: number }> | null>(null);
  const [leftLandmarks, setLeftLandmarks] = useState<Array<{ x: number; y: number; z: number }> | null>(null);
  const [isRecognizedFlash, setIsRecognizedFlash] = useState<boolean>(false);
  const [recognizedAt, setRecognizedAt] = useState<number | null>(null);

  const sttClientRef = useRef<StreamingSTTClient | null>(null);

  const {
    messages,
    sendFinalISLSign,
    sendFinalSpeech,
    activeLanguage,
  } = useRealtimeSession(`meeting-${meetingId}`, 'isl_signer');

  // Broadcast spoken sentence / subtitle
  const triggerSpeechTranscript = useCallback((text: string) => {
    if (!text.trim()) return;
    const cleanText = text.trim();
    setLiveSubtitle(cleanText);
    setSttError(null);
    sendFinalSpeech(cleanText, 0.95);
    setSpeechTranscriptForAvatar(cleanText);
  }, [sendFinalSpeech]);

  const handleRecognizedSign = useCallback((result: RecognitionResult) => {
    if (!isGestureEnabled) return;

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
      sessionId: `meeting-${meetingId}`,
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
  }, [isGestureEnabled, isSignToVoiceEnabled, activeLanguage, meetingId, sendFinalISLSign]);

  // Camera & MediaPipe Initialization
  useEffect(() => {
    if (!isCameraOn) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setRightLandmarks(null);
      setLeftLandmarks(null);
      return;
    }

    let isMounted = true;

    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        });
        if (!isMounted) return;

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }

        const { HandLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm'
        );
        const hl = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (!isMounted) return;
        mpRef.current = hl;

        let lastTime = -1;
        const loop = () => {
          if (!isMounted) return;
          const video = videoRef.current;
          if (video && video.readyState >= 2 && mpRef.current) {
            if (video.currentTime !== lastTime) {
              lastTime = video.currentTime;
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
            }
          }
          rafRef.current = requestAnimationFrame(loop);
        };
        loop();
      } catch (err) {
        console.error('Camera init error:', err);
      }
    }

    initCamera();

    return () => {
      isMounted = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [isCameraOn]);

  // STT Microphone Streaming
  useEffect(() => {
    if (isMicOn) {
      sttClientRef.current = new StreamingSTTClient({
        language: activeLanguage,
        onStateChange: (listening) => {
          setIsListeningActive(listening);
        },
        onPartial: (partial) => {
          setLiveSubtitle(partial);
          setSttError(null);
        },
        onFinal: (finalText) => {
          triggerSpeechTranscript(finalText);
        },
        onError: (err) => {
          console.warn('STT Error:', err);
          setSttError(err);
        },
      });
      sttClientRef.current.start();
    } else {
      sttClientRef.current?.stop();
      setIsListeningActive(false);
      setLiveSubtitle('Mic is muted. Click Mic ON or type below.');
    }

    return () => {
      sttClientRef.current?.stop();
    };
  }, [isMicOn, activeLanguage, triggerSpeechTranscript]);

  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    if (typedText.trim()) {
      triggerSpeechTranscript(typedText.trim());
      setTypedText('');
    }
  };

  return (
    <div className="min-h-screen bg-white text-slate-900 font-sans p-6 sm:p-10 flex flex-col">
      <header className="mb-6">
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">SignBridge</h1>
        <p className="text-base text-slate-600 font-medium mt-1">Meeting ID : {meetingId}</p>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start flex-1 max-w-7xl">
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
              {recognizedSign ? (
                <span className="text-emerald-600 bg-emerald-50 px-3 py-1 rounded-md border border-emerald-200">
                  {recognizedSign} ({Math.round(recognizedConfidence * 100)}%)
                </span>
              ) : (
                'No sign detected'
              )}
            </p>
          </div>

          <div>
            <h3 className="text-lg font-bold text-slate-900">Live Subtitles</h3>
            <div className="mt-2 bg-[#2a9d8f] text-white px-5 py-3.5 rounded-xl font-semibold text-base shadow-md flex items-center justify-between gap-3">
              <div className="flex items-center gap-2.5 flex-1 min-w-0">
                <span className={`w-3 h-3 rounded-full flex-shrink-0 ${isListeningActive ? 'bg-emerald-300 animate-ping' : 'bg-white/80'}`} />
                <span className="truncate">{liveSubtitle}</span>
              </div>
              {isListeningActive && (
                <span className="text-xs bg-white/20 px-2 py-0.5 rounded text-white font-medium flex-shrink-0">Live Mic</span>
              )}
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

          {/* Control Buttons */}
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

        {/* Right Column */}
        <section className="flex flex-col gap-4">
          <h2 className="text-xl font-bold text-slate-900">Sign Avatar</h2>
          <SignAvatar currentSign={avatarSign} textToSign={speechTranscriptForAvatar} className="w-full" />
        </section>
      </main>
    </div>
  );
}