/**
 * DebugPanel.tsx
 * Developer debug panel for SignBridge2Vision.
 *
 * Collapsible high-visibility panel fixed to the bottom-left of the viewport.
 * Displays comprehensive real-time pipeline diagnostics:
 *   - Camera status & FPS
 *   - Hand detection (Left / Right / Both) & confidence
 *   - Total 3D landmarks tracked (42/42 points)
 *   - Sequence buffer status (frames / 30)
 *   - State machine status (IDLE, ACCUMULATING, CLASSIFYING, RECOGNIZED)
 *   - Top-1 prediction with full confidence bar
 *   - Top-5 predictions with percentages
 *   - Triggered avatar animation ID & execution status
 *   - End-to-end inference latency (ms)
 */

import React from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProbabilityEntry {
  sign: string;
  confidence: number;
  label: string;
}

export interface DebugPanelProps {
  fps: number;
  handsDetected: number;
  rightConfidence?: number;
  leftConfidence?: number;
  landmarkCount?: number;
  sequenceFill: number;
  windowSize: number;
  inferenceLatencyMs: number;
  probabilities: ProbabilityEntry[];
  state: string;
  triggeredSign?: string | null;
  avatarActionStatus?: string;
  isOpen: boolean;
  onToggle: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function confBarColor(confidence: number): string {
  if (confidence >= 0.75) return '#22c55e'; // green-500
  if (confidence >= 0.50) return '#eab308'; // yellow-500
  return '#ef4444';                         // red-500
}

function fpsColor(fps: number): string {
  if (fps >= 25) return '#22c55e';
  if (fps >= 15) return '#eab308';
  return '#ef4444';
}

function stateColor(state: string): string {
  switch (state.toUpperCase()) {
    case 'RECOGNIZED':
    case 'STABLE':
      return '#22c55e';
    case 'CLASSIFYING':
    case 'PREDICTING':
      return '#38bdf8';
    case 'ACCUMULATING':
    case 'DETECTING':
      return '#eab308';
    default:
      return '#9ca3af';
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface MetricRowProps {
  label: string;
  value: string;
  valueColor?: string;
}

const MetricRow: React.FC<MetricRowProps> = ({ label, value, valueColor }) => (
  <div className="flex justify-between items-center text-[10px]">
    <span className="text-gray-400 font-medium">{label}</span>
    <span
      className="font-mono font-semibold"
      style={{ color: valueColor ?? '#e5e7eb' }}
    >
      {value}
    </span>
  </div>
);

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DebugPanel: React.FC<DebugPanelProps> = ({
  fps,
  handsDetected,
  rightConfidence = 0,
  leftConfidence = 0,
  landmarkCount,
  sequenceFill,
  windowSize,
  inferenceLatencyMs,
  probabilities,
  state,
  triggeredSign,
  avatarActionStatus = 'IDLE',
  isOpen,
  onToggle,
}) => {
  const top5 = probabilities.slice(0, 5);
  const top1 = top5[0];
  const calculatedLandmarks = landmarkCount ?? (handsDetected === 2 ? 42 : handsDetected === 1 ? 21 : 0);

  let handPresenceText = 'None';
  if (handsDetected === 2) {
    handPresenceText = `Both (R:${Math.round(rightConfidence * 100)}% L:${Math.round(leftConfidence * 100)}%)`;
  } else if (handsDetected === 1) {
    handPresenceText = rightConfidence > 0 ? `Right (${Math.round(rightConfidence * 100)}%)` : `Left (${Math.round(leftConfidence * 100)}%)`;
  }

  return (
    <div
      className="fixed bottom-3 left-3 z-50 select-none shadow-2xl transition-all"
      style={{ width: isOpen ? '280px' : '180px' }}
    >
      {/* ── Toggle button ─────────────────────────────────── */}
      <button
        onClick={onToggle}
        className={[
          'w-full flex items-center justify-between',
          'rounded-lg px-3 py-2',
          'bg-gray-900/95 border border-white/15 backdrop-blur-md',
          'text-[11px] font-mono font-semibold text-gray-200',
          'hover:bg-gray-800 transition-colors',
          isOpen ? 'rounded-b-none border-b-0' : '',
        ].join(' ')}
      >
        <span className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: fpsColor(fps) }}
          />
          <span>ISL VISION DEBUG</span>
        </span>
        <span className="text-gray-400 text-xs">{isOpen ? '▲' : '▼'}</span>
      </button>

      {/* ── Expanded panel ────────────────────────────────── */}
      {isOpen && (
        <div
          className={[
            'bg-gray-950/95 border border-white/15 border-t-0 backdrop-blur-md',
            'rounded-b-lg p-3 flex flex-col gap-2.5 max-h-[460px] overflow-y-auto',
          ].join(' ')}
        >
          {/* ── Pipeline Diagnostics ─────────────────────── */}
          <div className="flex flex-col gap-1 bg-white/5 p-2 rounded border border-white/5">
            <MetricRow
              label="Camera FPS"
              value={`${fps.toFixed(1)} FPS`}
              valueColor={fpsColor(fps)}
            />
            <MetricRow
              label="Detected Hands"
              value={handPresenceText}
              valueColor={handsDetected > 0 ? '#38bdf8' : '#9ca3af'}
            />
            <MetricRow
              label="3D Landmarks"
              value={`${calculatedLandmarks}/42 points`}
              valueColor={calculatedLandmarks === 42 ? '#22c55e' : calculatedLandmarks === 21 ? '#38bdf8' : '#9ca3af'}
            />
            <MetricRow
              label="Sequence Buffer"
              value={`${sequenceFill}/${windowSize} frames (${Math.round((sequenceFill / windowSize) * 100)}%)`}
              valueColor={sequenceFill >= windowSize ? '#22c55e' : '#eab308'}
            />
            <MetricRow
              label="Inference Latency"
              value={`${inferenceLatencyMs.toFixed(1)} ms`}
              valueColor={
                inferenceLatencyMs < 40
                  ? '#22c55e'
                  : inferenceLatencyMs < 80
                  ? '#eab308'
                  : '#ef4444'
              }
            />
            <MetricRow
              label="State Machine"
              value={state.toUpperCase()}
              valueColor={stateColor(state)}
            />
            <MetricRow
              label="Avatar Action"
              value={triggeredSign ? `${triggeredSign} (${avatarActionStatus})` : avatarActionStatus}
              valueColor={triggeredSign ? '#a855f7' : '#9ca3af'}
            />
          </div>

          {/* ── Top-1 Recognition Hero ───────────────────── */}
          {top1 && top1.confidence > 0 && (
            <div className="flex flex-col gap-1 bg-white/5 p-2 rounded border border-white/5">
              <div className="flex justify-between items-center text-[10px]">
                <span className="text-gray-400 font-semibold uppercase tracking-wider">Top-1 Recognized Sign</span>
                <span className="font-mono font-bold text-sm" style={{ color: confBarColor(top1.confidence) }}>
                  {Math.round(top1.confidence * 100)}%
                </span>
              </div>
              <div className="font-mono font-bold text-xs text-white truncate">
                {top1.sign} <span className="text-gray-400 font-normal">({top1.label})</span>
              </div>
              <div className="relative h-2 w-full rounded-full bg-white/10 overflow-hidden mt-0.5">
                <div
                  className="absolute left-0 top-0 h-full rounded-full transition-all duration-150"
                  style={{
                    width: `${Math.round(top1.confidence * 100)}%`,
                    backgroundColor: confBarColor(top1.confidence),
                  }}
                />
              </div>
            </div>
          )}

          {/* ── Top-5 Predictions List ───────────────────── */}
          {top5.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <span className="text-[9px] uppercase tracking-widest text-gray-500 font-semibold px-0.5">
                Top-5 Neural Probabilities
              </span>

              {top5.map((entry, idx) => {
                const pct = Math.round(entry.confidence * 100);
                const barColor = confBarColor(entry.confidence);

                return (
                  <div key={entry.sign} className="flex flex-col gap-0.5 bg-black/30 p-1.5 rounded border border-white/5">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="font-mono text-gray-200 truncate max-w-[170px]">
                        <span className="text-gray-500 mr-1.5">#{idx + 1}</span>
                        {entry.sign}
                      </span>
                      <span className="font-mono font-semibold" style={{ color: barColor }}>
                        {pct}%
                      </span>
                    </div>
                    {/* Confidence bar */}
                    <div className="relative h-1 w-full rounded-full bg-white/5 overflow-hidden">
                      <div
                        className="absolute left-0 top-0 h-full rounded-full transition-all duration-150"
                        style={{ width: `${pct}%`, backgroundColor: barColor }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DebugPanel;
