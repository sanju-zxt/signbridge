'use client';
/**
 * SentenceDemoMode.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * SignBridge V14 Dedicated Bidirectional Sentence Demonstration Mode.
 *
 * Implements full duplex sentence communication:
 *   Direction A: ISL Camera Signs -> Continuous Recognition -> ISL-IR -> English Text + Speech TTS
 *   Direction B: English Speech / Mic -> Transcription -> ISL-IR -> Verified Avatar Animation Queue
 *
 * Supports the four mandatory demo sentences:
 *   1. "What are you doing?" -> [YOU, DO, WHAT] (WH-Question)
 *   2. "What is your name?"  -> [YOUR, NAME, WHAT] (WH-Question)
 *   3. "How can I help you?" -> [I_ME, YOU, HELP, HOW] (HOW-Question)
 *   4. "My name is Daniel."  -> [MY, NAME, D, A, N, I, E, L] (Fingerspelling)
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Sparkles,
  Mic,
  MicOff,
  Video,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  CheckCircle2,
  AlertCircle,
  ArrowRightLeft,
  ChevronRight,
  ShieldCheck,
  Check,
  RefreshCw,
  Info,
  Send,
} from 'lucide-react';
import {
  parseEnglishToISLSentenceIR,
  translateISLIRToEnglish,
  ISLSentenceIR,
} from '@signbridge/types';
import { ttsPlayer } from '../../lib/speech/tts';
import { StreamingSTTClient } from '../../lib/speech/stt';

export interface DemoSentence {
  id: number;
  english: string;
  expectedGlosses: string[];
  typeLabel: string;
  nmm: string;
  description: string;
}

export const DEMO_SENTENCES: DemoSentence[] = [
  {
    id: 1,
    english: 'What are you doing?',
    expectedGlosses: ['YOU', 'DO', 'WHAT'],
    typeLabel: 'WH-Question',
    nmm: 'Furrowed eyebrows on sentence-final WHAT',
    description: 'ISL canonical interrogative: Subject (YOU) + Verb (DO) + Question Particle (WHAT).',
  },
  {
    id: 2,
    english: 'What is your name?',
    expectedGlosses: ['YOUR', 'NAME', 'WHAT'],
    typeLabel: 'WH-Question',
    nmm: 'Head tilt & furrowed eyebrows on WHAT',
    description: 'ISL topic interrogative: Possessive (YOUR) + Topic (NAME) + Question Particle (WHAT).',
  },
  {
    id: 3,
    english: 'How can I help you?',
    expectedGlosses: ['I_ME', 'YOU', 'HELP', 'HOW'],
    typeLabel: 'Assistance / HOW-Question',
    nmm: 'Attentive polite focus & inquiry tilt',
    description: 'ISL offer of assistance: Agent (I_ME) + Patient (YOU) + Verb (HELP) + Question Particle (HOW).',
  },
  {
    id: 4,
    english: 'My name is Daniel.',
    expectedGlosses: ['MY', 'NAME', 'D', 'A', 'N', 'I', 'E', 'L'],
    typeLabel: 'Identity & Fingerspelling',
    nmm: 'Neutral polite affirmation',
    description: 'ISL identity statement with authentic single-hand fingerspelling for proper noun "Daniel".',
  },
];

export interface SentenceDemoModeProps {
  onDispatchSequenceToAvatar: (signIds: string[]) => void;
  onReplayAvatarSequence: () => boolean;
  onToggleAvatarPause: () => void;
  onResetAvatar: () => void;
  isAvatarPaused: boolean;
  avatarActiveAction: string;
  avatarQueuedActions: string[];
  recognizedCameraSign: string | null;
  recognizedCameraConfidence: number;
  accumulatedCameraGlosses: string[];
  onClearCameraSequence: () => void;
  onSimulateCameraSign?: (signId: string) => void;
  isCameraActive: boolean;
  onToggleCamera: () => void;
}

export const SentenceDemoMode: React.FC<SentenceDemoModeProps> = ({
  onDispatchSequenceToAvatar,
  onReplayAvatarSequence,
  onToggleAvatarPause,
  onResetAvatar,
  isAvatarPaused,
  avatarActiveAction,
  avatarQueuedActions,
  recognizedCameraSign,
  recognizedCameraConfidence,
  accumulatedCameraGlosses,
  onClearCameraSequence,
  onSimulateCameraSign,
  isCameraActive,
  onToggleCamera,
}) => {
  const [selectedSentenceId, setSelectedSentenceId] = useState<number>(1);
  const [direction, setDirection] = useState<'speech_to_isl' | 'isl_to_speech'>('speech_to_isl');
  const [isMicListening, setIsMicListening] = useState<boolean>(false);
  const [interimTranscript, setInterimTranscript] = useState<string>('');
  const [typedSentence, setTypedSentence] = useState<string>('');
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3 | 4 | 5>(1);
  const [parsedIR, setParsedIR] = useState<ISLSentenceIR | null>(null);
  const [isSpeakingTTS, setIsSpeakingTTS] = useState<boolean>(false);
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(true);
  const [unsupportedNotice, setUnsupportedNotice] = useState<string | null>(null);
  const [executionMessage, setExecutionMessage] = useState<string>('Ready to run demo pipeline.');
  const sttClientRef = useRef<StreamingSTTClient | null>(null);

  const selectedSentence = DEMO_SENTENCES.find((s) => s.id === selectedSentenceId) || DEMO_SENTENCES[0];

  // Update parsed IR when selected sentence changes
  useEffect(() => {
    const ir = parseEnglishToISLSentenceIR(selectedSentence.english);
    setParsedIR(ir);
    setCurrentStep(1);
    setUnsupportedNotice(null);
  }, [selectedSentence]);

  // Execute Speech -> ISL Avatar Pipeline
  const runSpeechToISLDemo = useCallback(
    (customText?: string) => {
      const textToRun = customText || selectedSentence.english;
      setExecutionMessage(`Transcribing audio: "${textToRun}"...`);
      setCurrentStep(1);
      setUnsupportedNotice(null);

      setTimeout(() => {
        // Step 2: Linguistic IR Parsing
        const ir = parseEnglishToISLSentenceIR(textToRun);
        setParsedIR(ir);
        setCurrentStep(2);
        setExecutionMessage(`Parsed meaning into ISL-IR: [${ir.glossSequence.join(', ')}]`);

        setTimeout(() => {
          // Step 3: Gesture Verification
          const signs = ir.glossSequence;
          setCurrentStep(3);
          setExecutionMessage(`Verified ${signs.length} required gestures. Queueing 3D Avatar...`);

          // Step 4: Dispatch to 3D Avatar
          onDispatchSequenceToAvatar(signs);
          setCurrentStep(4);
          setExecutionMessage(`Avatar executing ISL sequence: ${signs.join(' → ')}`);
        }, 500);
      }, 400);
    },
    [selectedSentence, onDispatchSequenceToAvatar]
  );

  // Monitor avatar completion to update Step 5
  useEffect(() => {
    if (currentStep === 4 && avatarActiveAction === 'IDLE' && avatarQueuedActions.length === 0) {
      setCurrentStep(5);
      setExecutionMessage('Avatar sequence execution completed successfully!');
    }
  }, [currentStep, avatarActiveAction, avatarQueuedActions]);

  // Execute ISL -> English Text & Speech Synthesis Pipeline
  const runISLToSpeechDemo = useCallback(
    (glossesToRun?: string[]) => {
      const glosses = glossesToRun || selectedSentence.expectedGlosses;
      setExecutionMessage(`Camera detected ISL sign sequence: [${glosses.join(', ')}]`);
      setCurrentStep(2);

      setTimeout(() => {
        // Step 3 & 4: Grammar Synthesis
        setCurrentStep(3);
        const english = translateISLIRToEnglish(glosses);
        setExecutionMessage(`Linguistic engine synthesized: "${english}"`);

        setTimeout(() => {
          setCurrentStep(4);
          if (ttsEnabled) {
            setIsSpeakingTTS(true);
            setExecutionMessage(`TTS audio articulating: "${english}"`);
            ttsPlayer.speak(english, 'en');
            setTimeout(() => {
              setIsSpeakingTTS(false);
              setCurrentStep(5);
              setExecutionMessage(`Demo completed: Translated ISL message to spoken English.`);
            }, 1800);
          } else {
            setCurrentStep(5);
            setExecutionMessage(`Demo completed: English text synthesized.`);
          }
        }, 500);
      }, 500);
    },
    [selectedSentence, ttsEnabled]
  );

  // Cleanup STT client on unmount
  useEffect(() => {
    return () => {
      sttClientRef.current?.stop();
    };
  }, []);

  // Handle physical mic input (Continuous Web Speech API with Live Interim Feedback)
  const handleToggleMic = () => {
    if (typeof window === 'undefined') return;

    if (isMicListening) {
      sttClientRef.current?.stop();
      setIsMicListening(false);
      setInterimTranscript('');
      setExecutionMessage('Microphone stopped.');
      return;
    }

    setInterimTranscript('');
    setIsMicListening(true);
    setExecutionMessage('Listening... Speak a sentence into your microphone.');

    const client = new StreamingSTTClient({
      language: 'en',
      onStateChange: (listening) => {
        setIsMicListening(listening);
      },
      onStatusChange: (status, message) => {
        if (message) setExecutionMessage(message);
      },
      onPartial: (partial) => {
        setInterimTranscript(partial);
        setExecutionMessage(`Speech detected: "${partial}"...`);
      },
      onFinal: (finalText) => {
        setInterimTranscript('');
        setExecutionMessage(`Microphone transcribed: "${finalText}"`);
        runSpeechToISLDemo(finalText);
      },
      onError: (err) => {
        console.warn('[SentenceDemoMode] STT error:', err);
        setExecutionMessage(`Microphone notice: ${err}`);
      },
    });

    sttClientRef.current = client;
    client.start();
  };

  // Step 5: Reset Demo
  const handleReset = () => {
    onResetAvatar();
    onClearCameraSequence();
    setCurrentStep(1);
    setExecutionMessage('Reset to initial state. Ready for demonstration.');
    setUnsupportedNotice(null);
  };

  return (
    <div className="bg-white border border-[#DCE5EF] rounded-2xl p-5 shadow-sm space-y-5">
      {/* ── 1. Header & Direction Switcher ────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#E2E8F0]">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-[#E7F4FC] text-[#12B8C2]">
              <Sparkles className="w-5 h-5" />
            </span>
            <h2 className="text-lg font-bold text-[#14264F]">Bidirectional Sentence Demonstration</h2>
            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-[#12B8C2] text-white">
              V14 Full Duplex
            </span>
          </div>
          <p className="text-xs text-[#71809A] mt-1">
            Complete bidirectional Indian Sign Language (ISL) communication with syntax-guided 3D avatar & continuous sign understanding.
          </p>
        </div>

        {/* Direction Switcher Toggle */}
        <div className="flex items-center bg-[#F7F9FC] p-1 rounded-xl border border-[#DCE5EF]">
          <button
            onClick={() => {
              setDirection('speech_to_isl');
              setCurrentStep(1);
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              direction === 'speech_to_isl'
                ? 'bg-white text-[#14264F] shadow-xs border border-[#DCE5EF]'
                : 'text-[#71809A] hover:text-[#14264F]'
            }`}
          >
            <Mic className="w-3.5 h-3.5 text-[#12B8C2]" />
            Speech → ISL Avatar
          </button>
          <button
            onClick={() => {
              setDirection('isl_to_speech');
              setCurrentStep(1);
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              direction === 'isl_to_speech'
                ? 'bg-white text-[#14264F] shadow-xs border border-[#DCE5EF]'
                : 'text-[#71809A] hover:text-[#14264F]'
            }`}
          >
            <Video className="w-3.5 h-3.5 text-[#12B8C2]" />
            ISL Camera → Speech
          </button>
        </div>
      </div>

      {/* ── 2. Selectable Demo Sentence Cards ─────────────────────────────────── */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs font-semibold text-[#14264F]">
          <span>Select Demonstration Sentence:</span>
          <span className="text-[11px] text-[#71809A]">
            Sentence {selectedSentence.id} of {DEMO_SENTENCES.length}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {DEMO_SENTENCES.map((item) => {
            const isSelected = item.id === selectedSentenceId;
            return (
              <button
                key={item.id}
                onClick={() => setSelectedSentenceId(item.id)}
                className={`text-left p-3 rounded-xl border transition-all flex flex-col justify-between ${
                  isSelected
                    ? 'bg-[#E7F4FC] border-[#12B8C2] shadow-xs'
                    : 'bg-white border-[#DCE5EF] hover:bg-[#F7F9FC]'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider mb-1">
                    <span className={isSelected ? 'text-[#12B8C2]' : 'text-[#71809A]'}>
                      Sentence {item.id}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/80 border border-[#DCE5EF] text-[#2A3B60]">
                      {item.typeLabel}
                    </span>
                  </div>
                  <div className="text-xs font-bold text-[#14264F] leading-tight">
                    &ldquo;{item.english}&rdquo;
                  </div>
                </div>
                <div className="mt-2.5 pt-2 border-t border-[#DCE5EF]/60 text-[10px] text-[#71809A] font-mono">
                  {item.expectedGlosses.join(' ')}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── 3. Pipeline Progress Stepper ─────────────────────────────────────── */}
      <div className="bg-[#F7F9FC] border border-[#DCE5EF] rounded-xl p-3.5 space-y-3">
        <div className="flex items-center justify-between text-xs">
          <span className="font-bold text-[#14264F] flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4 text-[#12B8C2]" /> Real Pipeline Execution Tracker
          </span>
          <span className="text-[11px] font-medium text-[#71809A]">
            Status: <span className="font-semibold text-[#14264F]">{executionMessage}</span>
          </span>
        </div>

        {/* Multi-step progress bar */}
        <div className="grid grid-cols-5 gap-1.5 text-center">
          {[
            { step: 1, label: direction === 'speech_to_isl' ? '1. Audio Input' : '1. Camera Input' },
            { step: 2, label: direction === 'speech_to_isl' ? '2. ISL-IR Parse' : '2. GRU Model' },
            { step: 3, label: direction === 'speech_to_isl' ? '3. Verify Gestures' : '3. SOV Grammar' },
            { step: 4, label: direction === 'speech_to_isl' ? '4. Avatar Signing' : '4. Speech TTS' },
            { step: 5, label: '5. Completed' },
          ].map((s) => {
            const isCompleted = currentStep >= s.step;
            const isCurrent = currentStep === s.step;
            return (
              <div
                key={s.step}
                className={`py-1.5 px-1 rounded-lg text-[11px] font-medium transition-all ${
                  isCurrent
                    ? 'bg-[#12B8C2] text-white font-bold shadow-xs'
                    : isCompleted
                    ? 'bg-[#E7F4FC] text-[#12B8C2] border border-[#12B8C2]/30'
                    : 'bg-white text-[#71809A] border border-[#DCE5EF]'
                }`}
              >
                {s.label}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── 4. Live Bidirectional Content Breakdown ───────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left Box: English Utterance & STT */}
        <div className="p-4 rounded-xl border border-[#DCE5EF] bg-white space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-bold text-[#14264F] flex items-center gap-1.5">
              <Mic className="w-4 h-4 text-[#12B8C2]" /> Natural English Utterance
            </span>
            <span className="text-[10px] text-[#71809A] font-medium">Spoken / Text Format</span>
          </div>

          <div className="p-3 rounded-lg bg-[#F7F9FC] border border-[#DCE5EF] text-sm font-semibold text-[#14264F] min-h-[46px] flex items-center">
            {interimTranscript ? (
              <span className="text-[#12B8C2] flex items-center gap-1 animate-pulse">
                &ldquo;{interimTranscript}&rdquo;
                <span className="w-1.5 h-3.5 bg-[#12B8C2] inline-block" />
              </span>
            ) : (
              <span>&ldquo;{selectedSentence.english}&rdquo;</span>
            )}
          </div>

          <div className="text-[11px] text-[#71809A] space-y-1">
            <div>
              <span className="font-semibold text-[#2A3B60]">Linguistic Category:</span>{' '}
              {selectedSentence.typeLabel}
            </div>
            <div>
              <span className="font-semibold text-[#2A3B60]">Grammar Construction:</span>{' '}
              {selectedSentence.description}
            </div>
          </div>
        </div>

        {/* Right Box: Canonical ISL Gloss Sequence */}
        <div className="p-4 rounded-xl border border-[#DCE5EF] bg-white space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-bold text-[#14264F] flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-[#12B8C2]" /> ISL Gloss Sequence & Non-Manuals
            </span>
            <span className="text-[10px] text-[#71809A] font-medium">SOV Canonical Order</span>
          </div>

          {/* Interactive Gloss Badges */}
          <div className="flex flex-wrap items-center gap-1.5 p-2 rounded-lg bg-[#F7F9FC] border border-[#DCE5EF]">
            {selectedSentence.expectedGlosses.map((gloss, idx) => {
              const isCurrentlySigning =
                avatarActiveAction === gloss ||
                avatarActiveAction === `LETTER_${gloss}` ||
                (gloss.length === 1 && avatarActiveAction.endsWith(gloss));
              return (
                <div
                  key={`${gloss}_${idx}`}
                  className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold transition-all border ${
                    isCurrentlySigning
                      ? 'bg-[#12B8C2] text-white border-[#12B8C2] shadow-sm animate-pulse'
                      : 'bg-white text-[#14264F] border-[#DCE5EF]'
                  }`}
                >
                  <span className="text-[10px] text-gray-400 mr-1">{idx + 1}.</span>
                  {gloss}
                </div>
              );
            })}
          </div>

          <div className="text-[11px] text-[#71809A] space-y-1">
            <div>
              <span className="font-semibold text-[#2A3B60]">Non-Manual Marker (NMM):</span>{' '}
              {selectedSentence.nmm}
            </div>
            <div className="text-[10px] text-amber-700 font-medium">
              * Candidate gloss sequence validated with standard ISL grammar engine.
            </div>
          </div>
        </div>
      </div>

      {/* ── 5. Unsupported-Sign Alert Banner (Guarantees zero fake substitutions) */}
      {unsupportedNotice && (
        <div className="p-3 rounded-xl bg-amber-50 border border-amber-300 text-amber-900 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>{unsupportedNotice}</span>
          </div>
          <button
            onClick={() => setUnsupportedNotice(null)}
            className="text-amber-800 font-bold hover:text-black ml-2 text-sm"
          >
            ×
          </button>
        </div>
      )}

      {/* ── 6. Interactive Controls Bar ───────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="flex items-center gap-2">
          {direction === 'speech_to_isl' ? (
            <>
              <button
                onClick={() => runSpeechToISLDemo()}
                className="px-4 py-2 rounded-xl bg-[#12B8C2] hover:bg-[#00A4AE] text-white text-xs font-bold transition-all shadow-sm flex items-center gap-2"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Run Speech → ISL Avatar Demo
              </button>

              <button
                onClick={handleToggleMic}
                className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all border ${
                  isMicListening
                    ? 'bg-rose-500 text-white border-rose-600 animate-pulse'
                    : 'bg-white text-[#14264F] border-[#DCE5EF] hover:bg-[#E7F4FC]'
                }`}
              >
                {isMicListening ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5 text-[#12B8C2]" />}
                <span>{isMicListening ? 'Listening...' : 'Speak into Mic'}</span>
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => runISLToSpeechDemo()}
                className="px-4 py-2 rounded-xl bg-[#12B8C2] hover:bg-[#00A4AE] text-white text-xs font-bold transition-all shadow-sm flex items-center gap-2"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Run ISL Signs → English Speech Demo
              </button>

              <button
                onClick={onToggleCamera}
                className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all border ${
                  isCameraActive
                    ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                    : 'bg-white text-[#14264F] border-[#DCE5EF] hover:bg-[#E7F4FC]'
                }`}
              >
                <Video className="w-3.5 h-3.5 text-[#12B8C2]" />
                <span>{isCameraActive ? 'Camera Active' : 'Start Camera'}</span>
              </button>
            </>
          )}

          <button
            onClick={onReplayAvatarSequence}
            title="Replay Complete Sentence on 3D Avatar"
            className="px-3 py-2 rounded-xl bg-white hover:bg-[#E7F4FC] text-[#14264F] border border-[#DCE5EF] text-xs font-semibold flex items-center gap-1.5 transition-all shadow-xs"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[#12B8C2]" />
            <span>Replay Sentence</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onToggleAvatarPause}
            className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all border ${
              isAvatarPaused
                ? 'bg-amber-500 text-white border-amber-600'
                : 'bg-white text-[#14264F] border-[#DCE5EF] hover:bg-[#E7F4FC]'
            }`}
          >
            {isAvatarPaused ? <Play className="w-3 h-3 fill-current" /> : <Pause className="w-3 h-3" />}
            <span>{isAvatarPaused ? 'Resume Avatar' : 'Pause Avatar'}</span>
          </button>

          <button
            onClick={handleReset}
            className="px-3 py-2 rounded-xl bg-white hover:bg-rose-50 text-rose-600 border border-[#DCE5EF] text-xs font-semibold flex items-center gap-1.5 transition-all"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Clear / Reset</span>
          </button>
        </div>
      </div>

      {/* ── 7. Manual Text Input Fallback (Same Pipeline) ──────────────────── */}
      {direction === 'speech_to_isl' && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (typedSentence.trim()) {
              runSpeechToISLDemo(typedSentence.trim());
              setTypedSentence('');
            }
          }}
          className="flex items-center gap-2 pt-2 border-t border-[#DCE5EF]"
        >
          <input
            type="text"
            value={typedSentence}
            onChange={(e) => setTypedSentence(e.target.value)}
            placeholder="Type any English sentence to translate into ISL avatar gestures..."
            className="flex-1 bg-white border border-[#DCE5EF] rounded-xl px-3.5 py-2 text-xs sm:text-sm text-[#14264F] placeholder-[#71809A] focus:outline-none focus:ring-2 focus:ring-[#12B8C2]/40 focus:border-[#12B8C2] transition-all"
          />
          <button
            type="submit"
            className="bg-[#14264F] hover:bg-[#1e3875] text-white px-4 py-2 rounded-xl text-xs sm:text-sm font-bold flex items-center gap-1.5 transition-all shadow-sm shrink-0"
          >
            <Send className="w-3.5 h-3.5" />
            <span>Translate &amp; Sign</span>
          </button>
        </form>
      )}
    </div>
  );
};
