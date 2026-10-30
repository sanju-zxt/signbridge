const fs = require('fs');
const path = require('path');

function save(relPath, content) {
  const full = path.resolve(__dirname, '..', relPath);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content, 'utf8');
  console.log('Saved:', relPath);
}

// 1. apps/web/src/components/AudioControls.tsx
save('apps/web/src/components/AudioControls.tsx', `'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Mic, Send, Activity } from 'lucide-react';
import { SupportedLanguage } from '@signbridge2vision/types';
import { StreamingSTTClient } from '../lib/speech/stt';

interface AudioControlsProps {
  activeLanguage: SupportedLanguage;
  partialSpeech: string;
  onSendSpeech: (transcript: string, confidence?: number) => void;
  onSendText: (text: string) => void;
}

export const AudioControls: React.FC<AudioControlsProps> = ({
  activeLanguage,
  partialSpeech,
  onSendSpeech,
  onSendText
}) => {
  const [isListening, setIsListening] = useState<boolean>(false);
  const [typedText, setTypedText] = useState<string>('');
  const [sttError, setSttError] = useState<string | null>(null);
  const sttClientRef = useRef<StreamingSTTClient | null>(null);

  useEffect(() => {
    sttClientRef.current = new StreamingSTTClient({
      language: activeLanguage,
      onPartial: () => {},
      onFinal: (transcript, confidence) => {
        onSendSpeech(transcript, confidence);
      },
      onError: (err) => {
        setSttError(err);
      },
      onStateChange: (listening) => {
        setIsListening(listening);
      }
    });

    return () => {
      if (sttClientRef.current) {
        sttClientRef.current.stop();
      }
    };
  }, [activeLanguage, onSendSpeech]);

  const toggleMic = () => {
    setSttError(null);
    if (sttClientRef.current) {
      if (isListening) {
        sttClientRef.current.stop();
      } else {
        sttClientRef.current.start();
      }
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!typedText.trim()) return;
    onSendText(typedText.trim());
    setTypedText('');
  };

  const quickPhrases = [
    'How can I help you today?',
    'The registration desk is on the left.',
    'It is located on the second floor.',
    'Please proceed to Counter 4.',
    'You are very welcome.'
  ];

  return (
    <div className="flex flex-col bg-surface rounded-xl border border-surface-highlight/70 overflow-hidden shadow-xl p-4 gap-4">
      <div className="flex items-center justify-between pb-3 border-b border-surface-highlight/60">
        <div className="flex items-center gap-2">
          <div className={\`p-1.5 rounded-lg \${isListening ? 'bg-amber-500/20 text-amber-400' : 'bg-surface-elevated text-gray-400'}\`}>
            <Mic className="h-4 w-4" />
          </div>
          <div>
            <h4 className="text-xs font-semibold text-gray-200 uppercase tracking-wider">Hearing Partner Speech</h4>
            <p className="text-[11px] text-gray-400">Streaming Voice-to-Text Input</p>
          </div>
        </div>

        <button
          onClick={toggleMic}
          className={\`px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-2 \${
            isListening
              ? 'bg-amber-500 text-black font-semibold shadow-lg shadow-amber-500/20'
              : 'bg-surface-elevated hover:bg-surface-highlight text-gray-300 border border-surface-highlight'
          }\`}
        >
          {isListening ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-black opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-black"></span>
              </span>
              <span>Listening Active</span>
            </>
          ) : (
            <>
              <Mic className="h-3.5 w-3.5 text-gray-400" />
              <span>Turn On Mic</span>
            </>
          )}
        </button>
      </div>

      {partialSpeech && (
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center gap-2">
          <Activity className="h-4 w-4 text-amber-400 animate-pulse flex-shrink-0" />
          <div className="flex flex-col">
            <span className="text-[10px] text-amber-300 font-semibold uppercase tracking-wider">Transcribing Voice</span>
            <span className="text-xs text-gray-100 font-medium italic">"{partialSpeech}..."</span>
          </div>
        </div>
      )}

      {sttError && (
        <div className="text-[11px] text-amber-300 bg-amber-500/10 p-2 rounded border border-amber-500/20">
          {sttError}
        </div>
      )}

      <div className="flex flex-col gap-1.5">
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Quick Spoken Responses</span>
        <div className="flex flex-wrap gap-1.5">
          {quickPhrases.map((phrase, idx) => (
            <button
              key={idx}
              onClick={() => onSendSpeech(phrase, 0.98)}
              className="px-2.5 py-1 rounded bg-surface-elevated hover:bg-surface-highlight border border-surface-highlight text-[11px] text-gray-300 hover:text-white transition text-left"
            >
              {phrase}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleTextSubmit} className="flex items-center gap-2 pt-2 border-t border-surface-highlight/40">
        <input
          type="text"
          value={typedText}
          onChange={(e) => setTypedText(e.target.value)}
          placeholder="Or type a message to the shared conversation..."
          className="flex-1 bg-surface-elevated border border-surface-highlight/80 rounded-lg px-3 py-2 text-xs text-gray-100 placeholder-gray-500 outline-none focus:border-brand-500 transition"
        />
        <button
          type="submit"
          disabled={!typedText.trim()}
          className="p-2 rounded-lg bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white transition"
        >
          <Send className="h-3.5 w-3.5" />
        </button>
      </form>
    </div>
  );
};
`);

// 2. apps/web/src/components/ConversationThread.tsx
save('apps/web/src/components/ConversationThread.tsx', `'use client';

import React, { useRef, useEffect } from 'react';
import { CanonicalMessage, SupportedLanguage } from '@signbridge2vision/types';
import { Camera, Mic, Bot, Volume2, Globe, AlertTriangle } from 'lucide-react';
import { ttsPlayer } from '../lib/speech/tts';

interface ConversationThreadProps {
  messages: CanonicalMessage[];
  activeLanguage: SupportedLanguage;
  onClarify?: (message: CanonicalMessage) => void;
  onClear: () => void;
}

export const ConversationThread: React.FC<ConversationThreadProps> = ({
  messages,
  activeLanguage,
  onClarify,
  onClear
}) => {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const getModalityMeta = (modality: string) => {
    switch (modality) {
      case 'isl':
        return {
          label: 'ISL Camera',
          icon: <Camera className="h-3 w-3 text-teal-400" />,
          chip: 'bg-teal-500/20 text-teal-300 border-teal-500/30'
        };
      case 'speech':
        return {
          label: 'Voice Mic',
          icon: <Mic className="h-3 w-3 text-amber-400" />,
          chip: 'bg-amber-500/20 text-amber-300 border-amber-500/30'
        };
      default:
        return {
          label: 'Assistant / Text',
          icon: <Bot className="h-3 w-3 text-indigo-400" />,
          chip: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30'
        };
    }
  };

  const handleSpeak = (text: string) => {
    ttsPlayer.speak(text, activeLanguage);
  };

  return (
    <div className="flex flex-col h-full bg-surface rounded-xl border border-surface-highlight/70 overflow-hidden shadow-xl">
      <div className="flex items-center justify-between px-4 py-3 bg-surface-elevated/70 border-b border-surface-highlight/60">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-brand-500 animate-pulse"></div>
          <h3 className="text-xs font-semibold text-gray-200 uppercase tracking-wider">Shared Semantic Conversation</h3>
          <span className="text-[11px] text-gray-400 font-mono">({messages.length} messages)</span>
        </div>
        <button
          onClick={onClear}
          className="text-[11px] text-gray-400 hover:text-gray-200 px-2 py-1 rounded bg-surface border border-surface-highlight/50 transition"
        >
          Clear Thread
        </button>
      </div>

      <div className="flex-1 p-4 overflow-y-auto space-y-4 max-h-[520px] min-h-[380px]">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full py-12 text-center text-gray-400">
            <Bot className="h-10 w-10 text-gray-600 mb-2" />
            <h4 className="text-sm font-medium text-gray-300">Conversation Thread Ready</h4>
            <p className="text-xs text-gray-500 max-w-sm mt-1">
              Sign in front of the camera, speak into the microphone, or choose a demo scenario to start communicating.
            </p>
          </div>
        ) : (
          messages.map((msg) => {
            const meta = getModalityMeta(msg.modality);
            const isISL = msg.modality === 'isl';
            const isAssistant = msg.speakerRole === 'assistant';
            const translation = msg.translations?.[activeLanguage];

            return (
              <div
                key={msg.id}
                className={\`flex flex-col p-3.5 rounded-xl border transition \${
                  isISL
                    ? 'bg-teal-950/20 border-teal-500/30'
                    : isAssistant
                    ? 'bg-indigo-950/20 border-indigo-500/30'
                    : 'bg-amber-950/20 border-amber-500/30'
                }\`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-white">{msg.speakerName}</span>
                    <div className={\`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border \${meta.chip}\`}>
                      {meta.icon}
                      <span>{meta.label}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {msg.confidence !== undefined && (
                      <span className={\`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded \${
                        msg.confidence >= 0.75
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : 'bg-rose-500/20 text-rose-300'
                      }\`}>
                        {(msg.confidence * 100).toFixed(0)}% conf
                      </span>
                    )}
                    <button
                      onClick={() => handleSpeak(translation || msg.text)}
                      title="Read aloud with Text-to-Speech"
                      className="p-1 rounded hover:bg-surface-highlight text-gray-400 hover:text-white transition"
                    >
                      <Volume2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                <p className="text-sm text-gray-100 font-medium leading-relaxed">
                  {msg.text}
                </p>

                {activeLanguage !== 'en' && translation && translation !== msg.text && (
                  <div className="mt-2 pt-2 border-t border-surface-highlight/40 flex items-start gap-1.5 text-xs text-brand-200">
                    <Globe className="h-3.5 w-3.5 mt-0.5 flex-shrink-0 text-brand-400" />
                    <span>{translation}</span>
                  </div>
                )}

                {msg.status === 'low_confidence' && (
                  <div className="mt-2 p-2 rounded bg-rose-500/10 border border-rose-500/30 flex items-center justify-between text-xs text-rose-300">
                    <div className="flex items-center gap-1.5">
                      <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />
                      <span>Low confidence recognition. Please verify.</span>
                    </div>
                    {onClarify && (
                      <button
                        onClick={() => onClarify(msg)}
                        className="px-2 py-0.5 rounded bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-[11px] font-semibold"
                      >
                        Clarify
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};
`);

// 3. apps/web/src/components/ConfidenceGate.tsx
save('apps/web/src/components/ConfidenceGate.tsx', `'use client';

import React from 'react';
import { AlertCircle, Check, RefreshCw } from 'lucide-react';
import { AlternativeCandidate } from '@signbridge2vision/types';

interface ConfidenceGateProps {
  provisionalSign: string;
  confidence: number;
  alternatives: AlternativeCandidate[];
  onConfirm: (sign: string) => void;
  onRepeat: () => void;
}

export const ConfidenceGate: React.FC<ConfidenceGateProps> = ({
  provisionalSign,
  confidence,
  alternatives,
  onConfirm,
  onRepeat
}) => {
  return (
    <div className="p-4 rounded-xl bg-surface-elevated border border-rose-500/40 shadow-xl flex flex-col gap-3">
      <div className="flex items-center gap-2 text-rose-400">
        <AlertCircle className="h-4 w-4 flex-shrink-0" />
        <span className="text-xs font-semibold uppercase tracking-wider">Confidence Resolution Required</span>
      </div>
      <p className="text-xs text-gray-300">
        The system detected <span className="font-bold text-white">"{provisionalSign}"</span> with {(confidence * 100).toFixed(0)}% confidence.
        Please confirm or select an alternative before adding to the shared thread.
      </p>

      <div className="flex flex-wrap gap-2 pt-1">
        <button
          onClick={() => onConfirm(provisionalSign)}
          className="px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium transition flex items-center gap-1.5"
        >
          <Check className="h-3.5 w-3.5" />
          <span>Confirm "{provisionalSign}"</span>
        </button>

        {alternatives.map((alt) => (
          <button
            key={alt.sign}
            onClick={() => onConfirm(alt.sign)}
            className="px-3 py-1.5 rounded-lg bg-surface hover:bg-surface-highlight border border-surface-highlight text-gray-300 hover:text-white text-xs font-medium transition"
          >
            "{alt.label.split('/')[0]}" ({(alt.confidence * 100).toFixed(0)}%)
          </button>
        ))}

        <button
          onClick={onRepeat}
          className="px-3 py-1.5 rounded-lg bg-surface-elevated hover:bg-surface-highlight border border-surface-highlight text-gray-400 hover:text-gray-200 text-xs font-medium transition flex items-center gap-1"
        >
          <RefreshCw className="h-3 w-3" />
          <span>Repeat Sign</span>
        </button>
      </div>
    </div>
  );
};
`);
