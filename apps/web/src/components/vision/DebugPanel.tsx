/**
 * DebugPanel.tsx
 * Developer debug panel for SignBridge2Vision.
 *
 * Collapsible dark panel fixed to the bottom-left of the viewport.
 * Shows real-time pipeline diagnostics: FPS, hand count, sequence buffer,
 * inference latency, top-3 predictions, and current pipeline state.
 */

import React from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ProbabilityEntry {
  sign: string;
  confidence: number;
  label: string;
}

interface DebugPanelProps {
  fps: number;
  handsDetected: number;
  sequenceFill: number;       // current frames in buffer
  windowSize: number;         // total frames required
  inferenceLatencyMs: number; // last inference duration in ms
  probabilities: ProbabilityEntry[];
  state: string;              // pipeline state string
  isOpen: boolean;
  onToggle: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function confBarColor(confidence: number): string {
  if (confidence >= 0.8) return '#22c55e'; // green-500
  if (confidence >= 0.6) return '#eab308'; // yellow-500
  return '#ef4444';                         // red-500
}

function fpsColor(fps: number): string {
  if (fps >= 25) return '#22c55e';
  if (fps >= 15) return '#eab308';
  return '#ef4444';
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
  <div className="flex justify-between items-center">
    <span className="text-gray-500 text-[10px]">{label}</span>
    <span
      className="font-mono text-[11px] font-semibold"
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
  sequenceFill,
  windowSize,
  inferenceLatencyMs,
  probabilities,
  state,
  isOpen,
  onToggle,
}) => {
  const top3 = probabilities.slice(0, 3);

  return (
    <div
      className="fixed bottom-3 left-3 z-50 select-none"
      style={{ minWidth: '180px', maxWidth: '220px' }}
    >
      {/* ── Toggle button ─────────────────────────────────── */}
      <button
        onClick={onToggle}
        className={[
          'w-full flex items-center justify-between',
          'rounded-lg px-2.5 py-1.5',
          'bg-gray-900/90 border border-white/10',
          'text-[10px] font-mono text-gray-400',
          'hover:bg-gray-800/90 hover:text-gray-200 transition-colors',
          isOpen ? 'rounded-b-none border-b-0' : '',
        ].join(' ')}
      >
        <span className="flex items-center gap-1.5">
          {/* Live indicator dot */}
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: fpsColor(fps) }}
          />
          DEBUG
        </span>
        <span className="text-gray-600">{isOpen ? '▲' : '▼'}</span>
      </button>

      {/* ── Expanded panel ────────────────────────────────── */}
      {isOpen && (
        <div
          className={[
            'bg-gray-950/95 border border-white/10 border-t-0',
            'rounded-b-lg px-3 py-2.5 flex flex-col gap-2',
          ].join(' ')}
        >
          {/* ── Core metrics ──────────────────────────────── */}
          <div className="flex flex-col gap-1">
            <MetricRow
              label="FPS"
              value={fps.toFixed(1)}
              valueColor={fpsColor(fps)}
            />
            <MetricRow
              label="Hands"
              value={`${handsDetected}`}
              valueColor={handsDetected > 0 ? '#22c55e' : '#9ca3af'}
            />
            <MetricRow
              label="Buffer"
              value={`${sequenceFill}/${windowSize}`}
            />
            <MetricRow
              label="Latency"
              value={`${inferenceLatencyMs.toFixed(1)} ms`}
              valueColor={
                inferenceLatencyMs < 50
                  ? '#22c55e'
                  : inferenceLatencyMs < 100
                  ? '#eab308'
                  : '#ef4444'
              }
            />
            <MetricRow
              label="State"
              value={state}
              valueColor={
                state === 'STABLE'
                  ? '#22c55e'
                  : state === 'ANALYZING'
                  ? '#60a5fa'
                  : '#9ca3af'
              }
            />
          </div>

          {/* ── Divider ───────────────────────────────────── */}
          {top3.length > 0 && (
            <>
              <div className="border-t border-white/10" />

              {/* ── Top-3 predictions ─────────────────────── */}
              <div className="flex flex-col gap-1.5">
                <span className="text-[9px] uppercase tracking-widest text-gray-600">
                  Top Predictions
                </span>

                {top3.map((entry) => {
                  const pct = Math.round(entry.confidence * 100);
                  const barColor = confBarColor(entry.confidence);

                  return (
                    <div key={entry.sign} className="flex flex-col gap-0.5">
                      <div className="flex justify-between items-center">
                        <span className="font-mono text-[10px] text-gray-300 truncate max-w-[120px]">
                          {entry.sign}
                        </span>
                        <span className="font-mono text-[10px]" style={{ color: barColor }}>
                          {pct}%
                        </span>
                      </div>
                      {/* Mini confidence bar */}
                      <div className="relative h-1 w-full rounded-full bg-white/5 overflow-hidden">
                        <div
                          className="absolute left-0 top-0 h-full rounded-full transition-all duration-150"
                          style={{ width: `${pct}%`, backgroundColor: barColor }}
                        />
                      </div>
                      <span className="text-[9px] text-gray-600 truncate">{entry.label}</span>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default DebugPanel;
