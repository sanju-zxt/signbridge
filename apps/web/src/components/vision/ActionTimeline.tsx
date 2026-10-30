/**
 * ActionTimeline.tsx
 * Recognition history timeline for SignBridge2Vision.
 *
 * Shows the 8 most recent recognised signs, newest first, with
 * timestamp, sign ID chip, display name, and colour-coded confidence badge.
 */

import React from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TimelineEvent {
  gesture: string;
  displayName: string;
  confidence: number; // 0.0 – 1.0
  timestamp: number;  // Date.now()
}

interface ActionTimelineProps {
  events: TimelineEvent[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MAX_EVENTS = 8;

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

function confidenceBadgeClass(confidence: number): string {
  if (confidence >= 0.8) return 'bg-green-500/20 text-green-400 border-green-500/30';
  if (confidence >= 0.6) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
  return 'bg-red-500/20 text-red-400 border-red-500/30';
}

function signChipClass(confidence: number): string {
  if (confidence >= 0.8) return 'bg-green-900/50 text-green-300';
  if (confidence >= 0.6) return 'bg-yellow-900/50 text-yellow-300';
  return 'bg-red-900/50 text-red-300';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const ActionTimeline: React.FC<ActionTimelineProps> = ({ events }) => {
  // Newest first, capped at MAX_EVENTS
  const displayed = [...events].reverse().slice(0, MAX_EVENTS);

  return (
    <div className="flex flex-col gap-1 w-full">
      {/* ── Header ──────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-1 mb-1">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          Recognition History
        </h3>
        <span className="text-[10px] text-gray-600">
          {events.length > 0 ? `${Math.min(events.length, MAX_EVENTS)} / ${events.length}` : ''}
        </span>
      </div>

      {/* ── Event list ──────────────────────────────────────── */}
      <div className="flex flex-col gap-1 overflow-y-auto max-h-60 pr-0.5">
        {displayed.length === 0 ? (
          /* Empty state */
          <div className="flex items-center justify-center h-20 rounded-lg border border-dashed border-white/10">
            <p className="text-xs text-gray-500 italic">No signs recognised yet</p>
          </div>
        ) : (
          displayed.map((event, idx) => (
            <div
              key={`${event.timestamp}-${idx}`}
              className="flex items-center gap-2 rounded-lg bg-white/5 border border-white/[0.08] px-2.5 py-1.5 hover:bg-white/[0.08] transition-colors"
            >
              {/* Timestamp */}
              <span className="shrink-0 font-mono text-[10px] text-gray-500 w-[52px]">
                {formatTimestamp(event.timestamp)}
              </span>

              {/* Sign ID chip */}
              <span
                className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold tracking-wider ${signChipClass(event.confidence)}`}
              >
                {event.gesture}
              </span>

              {/* Display name */}
              <span className="flex-1 text-xs text-gray-300 truncate" title={event.displayName}>
                {event.displayName}
              </span>

              {/* Confidence badge */}
              <span
                className={`shrink-0 rounded border text-[10px] font-mono font-semibold px-1.5 py-0.5 ${confidenceBadgeClass(event.confidence)}`}
              >
                {Math.round(event.confidence * 100)}%
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ActionTimeline;
