/**
 * RecognitionHUD.tsx
 * Real-time recognition HUD overlay for SignBridge2Vision.
 *
 * Positioned absolute top-right over the camera feed. Shows the currently
 * recognised sign, confidence, pipeline state, and sequence buffer fill.
 */

import React from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RecognitionHUDProps {
  gesture: string | null;
  displayName: string | null;
  confidence: number; // 0.0 – 1.0
  state: string;      // e.g. 'STABLE' | 'ANALYZING' | 'UNCLEAR' | 'IDLE'
  modelId: string;
  sequenceLength: number; // current frames in buffer
  windowSize: number;     // total frames required
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function confidenceBarColor(confidence: number): string {
  if (confidence >= 0.8) return 'bg-green-500';
  if (confidence >= 0.6) return 'bg-yellow-400';
  return 'bg-red-500';
}

function stateBadgeStyle(state: string): string {
  switch (state) {
    case 'RECOGNIZED':
    case 'STABLE':
      return 'bg-green-600 text-green-100';
    case 'CLASSIFYING':
    case 'ANALYZING':
      return 'bg-blue-600 text-blue-100';
    case 'DETECTING':
    case 'LOW_CONFIDENCE':
    case 'UNCLEAR':
      return 'bg-yellow-600 text-yellow-100';
    default:
      return 'bg-gray-600 text-gray-100';
  }
}

function stateLabel(state: string): string {
  switch (state) {
    case 'RECOGNIZED':
    case 'STABLE':    return 'RECOGNIZED';
    case 'CLASSIFYING':
    case 'ANALYZING': return 'CLASSIFYING';
    case 'DETECTING': return 'DETECTING';
    case 'LOW_CONFIDENCE':
    case 'UNCLEAR':   return 'UNCLEAR';
    case 'IDLE':      return 'IDLE';
    default:          return state;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const RecognitionHUD: React.FC<RecognitionHUDProps> = ({
  gesture,
  displayName,
  confidence,
  state,
  modelId,
  sequenceLength,
  windowSize,
}) => {
  const confidencePct = Math.round(confidence * 100);
  const barWidth = `${confidencePct}%`;

  return (
    <div
      className={[
        'absolute top-3 right-3 z-20',
        'w-52 rounded-xl',
        'bg-black/70 backdrop-blur-sm border border-white/10',
        'p-3 flex flex-col gap-2',
        'select-none',
      ].join(' ')}
    >
      {/* ── Sign ID ─────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-1">
        <span
          className="font-mono text-2xl font-bold leading-tight text-white tracking-widest truncate"
          title={gesture ?? '—'}
        >
          {gesture ?? '—'}
        </span>

        {/* Status badge */}
        <span
          className={`shrink-0 mt-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${stateBadgeStyle(state)}`}
        >
          {stateLabel(state)}
        </span>
      </div>

      {/* ── Display name ─────────────────────────────────── */}
      <p className="text-sm text-gray-300 leading-snug -mt-1 truncate">
        {displayName ?? 'No sign detected'}
      </p>

      {/* ── Confidence bar ───────────────────────────────── */}
      <div>
        <div className="flex justify-between text-[11px] text-gray-400 mb-1">
          <span>Confidence</span>
          <span className="font-mono text-white">{confidencePct}%</span>
        </div>
        <div className="relative h-2 w-full rounded-full bg-white/10 overflow-hidden">
          <div
            className={`absolute left-0 top-0 h-full rounded-full transition-all duration-200 ease-out ${confidenceBarColor(confidence)}`}
            style={{ width: barWidth }}
          />
        </div>
      </div>

      {/* ── Divider ──────────────────────────────────────── */}
      <div className="border-t border-white/10" />

      {/* ── Sequence buffer fill ─────────────────────────── */}
      <div className="flex justify-between text-[11px] text-gray-400">
        <span>Buffer</span>
        <span className="font-mono text-gray-200">
          {sequenceLength}/{windowSize}
          <span className="text-gray-500"> frames</span>
        </span>
      </div>

      {/* ── Model ID ─────────────────────────────────────── */}
      <div className="flex justify-between text-[11px] text-gray-500">
        <span>Model</span>
        <span className="font-mono text-gray-400 truncate max-w-[110px]" title={modelId}>
          {modelId}
        </span>
      </div>
    </div>
  );
};

export default RecognitionHUD;
