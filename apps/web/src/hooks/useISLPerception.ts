'use client';
/**
 * useISLPerception.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * React hook that connects the ISL recognition pipeline output to the
 * application's conversation state.
 *
 * This hook does NOT perform recognition — that is handled by LiveCameraView
 * which runs MediaPipe + GRU inference internally.  This hook simply exposes
 * a callback that components can register to receive recognition events.
 */

import { useState, useCallback } from 'react';
import type { RecognitionResult, RecognitionState, CanonicalMessage } from '@signbridge/types';
import { ISL_VOCABULARY } from '@signbridge/types';

export interface ISLPerceptionState {
  lastRecognition: RecognitionResult | null;
  recognitionState: RecognitionState;
  onSignRecognised: (result: RecognitionResult) => void;
  clearLastRecognition: () => void;
}

/**
 * Converts a RecognitionResult into a CanonicalMessage for the conversation.
 * Separates the vision layer from the language layer — the ML model only
 * outputs a sign ID; this hook maps it to a natural-language sentence.
 */
function toCanonicalMessage(
  result: RecognitionResult,
  sessionId: string,
  speakerId: string
): CanonicalMessage {
  const vocab = ISL_VOCABULARY.find(v => v.sign === result.gesture);
  const text = vocab?.canonicalSentence ?? result.displayName;
  const translations = vocab?.translations;

  return {
    id: `msg_isl_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    sessionId,
    speakerId,
    speakerRole: 'isl_signer',
    speakerName: 'ISL Signer',
    modality: 'isl',
    text,
    normalizedText: text,
    confidence: result.confidence,
    alternatives: result.probabilities.slice(1, 4).map(p => ({
      sign: p.sign,
      confidence: p.confidence,
      label: p.label,
    })),
    language: 'en',
    timestamp: result.timestamp,
    status: result.confidence >= 0.70 ? 'final' : 'low_confidence',
    latencyMs: result.latencyMs,
    translations,
  };
}

export function useISLPerception(
  sessionId: string,
  speakerId: string,
  onMessage?: (msg: CanonicalMessage) => void
): ISLPerceptionState {
  const [lastRecognition, setLastRecognition] = useState<RecognitionResult | null>(null);
  const [recognitionState, setRecognitionState] = useState<RecognitionState>('IDLE');

  const onSignRecognised = useCallback((result: RecognitionResult) => {
    setLastRecognition(result);
    setRecognitionState(result.confidence >= 0.70 ? 'RECOGNIZED' : 'LOW_CONFIDENCE');

    // Convert to canonical message and propagate to conversation layer
    const msg = toCanonicalMessage(result, sessionId, speakerId);
    onMessage?.(msg);
  }, [sessionId, speakerId, onMessage]);

  const clearLastRecognition = useCallback(() => {
    setLastRecognition(null);
    setRecognitionState('IDLE');
  }, []);

  return { lastRecognition, recognitionState, onSignRecognised, clearLastRecognition };
}