'use client';

import React from 'react';
import { X, Sparkles, Play, HeartPulse, GraduationCap, Users } from 'lucide-react';
import { DEMO_SCENARIO_PRESETS } from '@signbridge2vision/types';

interface DemoControlsProps {
  isOpen: boolean;
  onClose: () => void;
  onRunScenarioStep: (speaker: 'isl' | 'speech', sign?: string, text?: string) => void;
}

export const DemoControls: React.FC<DemoControlsProps> = ({
  isOpen,
  onClose,
  onRunScenarioStep
}) => {
  if (!isOpen) return null;

  const getIcon = (id: string) => {
    if (id.includes('hospital')) return <HeartPulse className="h-5 w-5 text-rose-400" />;
    if (id.includes('college')) return <GraduationCap className="h-5 w-5 text-indigo-400" />;
    return <Users className="h-5 w-5 text-teal-400" />;
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-surface border border-surface-highlight rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh]">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-highlight bg-surface-elevated">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-brand-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Live Demo Presentation Scenarios</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-surface-highlight text-gray-400 hover:text-white transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto space-y-4">
          {DEMO_SCENARIO_PRESETS.map((preset) => (
            <div key={preset.id} className="p-4 rounded-xl bg-surface-elevated border border-surface-highlight/80 flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-surface border border-surface-highlight">
                  {getIcon(preset.id)}
                </div>
                <div>
                  <h4 className="text-sm font-bold text-white">{preset.title}</h4>
                  <p className="text-xs text-gray-400">{preset.description}</p>
                </div>
              </div>

              <div className="space-y-2 pt-2 border-t border-surface-highlight/40">
                <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Interactive Turn-by-Turn Steps:</span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {preset.sampleDialogue.map((step, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        onRunScenarioStep(step.speaker, step.sign, step.text);
                        onClose();
                      }}
                      className={`p-2.5 rounded-lg border text-left transition flex items-center justify-between ${
                        step.speaker === 'isl'
                          ? 'bg-teal-950/30 border-teal-500/40 hover:bg-teal-900/40'
                          : 'bg-amber-950/30 border-amber-500/40 hover:bg-amber-900/40'
                      }`}
                    >
                      <div className="flex flex-col pr-2">
                        <span className={`text-[10px] font-bold uppercase ${
                          step.speaker === 'isl' ? 'text-teal-400' : 'text-amber-400'
                        }`}>
                          Turn {idx + 1}: {step.speaker === 'isl' ? `ISL [${step.sign}]` : 'Voice Reply'}
                        </span>
                        <span className="text-xs text-gray-200 truncate max-w-[200px]">{step.text}</span>
                      </div>
                      <Play className="h-3.5 w-3.5 text-gray-400 flex-shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
