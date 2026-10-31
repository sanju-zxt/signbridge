'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Mic, Send, Activity } from 'lucide-react';
import { SupportedLanguage } from '@signbridge/types';
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
          <div className={`p-1.5 rounded-lg ${isListening ? 'bg-amber-500/20 text-amber-400' : 'bg-surface-elevated text-gray-400'}`}>
            <Mic className="h-4 w-4" />
          </div>
          <div>
            <h4 className="text-xs font-semibold text-gray-200 uppercase tracking-wider">Hearing Partner Speech</h4>
            <p className="text-[11px] text-gray-400">Streaming Voice-to-Text Input</p>
          </div>
        </div>

        <button
          onClick={toggleMic}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-2 ${
            isListening
              ? 'bg-amber-500 text-black font-semibold shadow-lg shadow-amber-500/20'
              : 'bg-surface-elevated hover:bg-surface-highlight text-gray-300 border border-surface-highlight'
          }`}
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
