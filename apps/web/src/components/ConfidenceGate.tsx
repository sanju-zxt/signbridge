'use client';

import React from 'react';
import { AlertCircle, Check, RefreshCw } from 'lucide-react';
import { AlternativeCandidate } from '@signbridge/types';

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
