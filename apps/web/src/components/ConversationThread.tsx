'use client';

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
                className={`flex flex-col p-3.5 rounded-xl border transition ${
                  isISL
                    ? 'bg-teal-950/20 border-teal-500/30'
                    : isAssistant
                    ? 'bg-indigo-950/20 border-indigo-500/30'
                    : 'bg-amber-950/20 border-amber-500/30'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-white">{msg.speakerName}</span>
                    <div className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border ${meta.chip}`}>
                      {meta.icon}
                      <span>{meta.label}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {msg.confidence !== undefined && (
                      <span className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded ${
                        msg.confidence >= 0.75
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : 'bg-rose-500/20 text-rose-300'
                      }`}>
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
