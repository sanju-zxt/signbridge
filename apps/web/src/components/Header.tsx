'use client';

import React from 'react';
import { Activity, Radio, Volume2, ShieldCheck, Languages, BarChart3, Sparkles } from 'lucide-react';
import { SystemState, SupportedLanguage, LatencySummary } from '@signbridge/types';

interface HeaderProps {
  systemState: SystemState;
  isConnected: boolean;
  activeLanguage: SupportedLanguage;
  onLanguageChange: (lang: SupportedLanguage) => void;
  latencySummary?: { overall: LatencySummary } | null;
  onOpenMetrics: () => void;
  onOpenPresets: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  systemState,
  isConnected,
  activeLanguage,
  onLanguageChange,
  latencySummary,
  onOpenMetrics,
  onOpenPresets
}) => {
  const getStateBadge = (state: SystemState) => {
    switch (state) {
      case 'SIGNING':
        return { label: 'ISL Signing Detected', bg: 'bg-teal-500/20 text-teal-300 border-teal-500/30' };
      case 'LISTENING':
        return { label: 'Speech Listening', bg: 'bg-amber-500/20 text-amber-300 border-amber-500/30' };
      case 'THINKING':
        return { label: 'AI Agent Reasoning', bg: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30 animate-pulse' };
      case 'RESPONDING':
        return { label: 'Agent Response Ready', bg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' };
      case 'SPEAKING':
        return { label: 'TTS Synthesizing', bg: 'bg-purple-500/20 text-purple-300 border-purple-500/30' };
      case 'LOW_CONFIDENCE':
        return { label: 'Verifying Confidence', bg: 'bg-rose-500/20 text-rose-300 border-rose-500/30' };
      case 'RECONNECTING':
        return { label: 'Reconnecting Gateway', bg: 'bg-amber-500/20 text-amber-300 border-amber-500/30 animate-pulse' };
      default:
        return { label: 'Ready & Synchronized', bg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' };
    }
  };

  const badge = getStateBadge(systemState);

  return (
    <header className="w-full bg-surface/90 backdrop-blur border-b border-surface-highlight/50 px-4 lg:px-8 py-3 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Left: Branding & Tagline */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-brand-600 to-isl-teal flex items-center justify-center shadow-lg shadow-brand-600/20">
            <Radio className="h-5 w-5 text-white animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight text-white">SignBridge</span>
              <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded bg-brand-500/20 text-brand-100 border border-brand-500/30">
                Multimodal AI
              </span>
            </div>
            <p className="text-xs text-gray-400 hidden sm:block">ISL & Speech Semantic Communication Platform</p>
          </div>
        </div>

        {/* Center: Live System Status Pill */}
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium ${badge.bg}`}>
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isConnected ? 'bg-emerald-400' : 'bg-rose-400'}`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isConnected ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
            </span>
            <span>{badge.label}</span>
          </div>

          {latencySummary?.overall && latencySummary.overall.p50 > 0 && (
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-surface-elevated text-[11px] text-gray-300 border border-surface-highlight/60 font-mono">
              <Activity className="h-3 w-3 text-isl-teal" />
              <span>p50: {latencySummary.overall.p50}ms</span>
            </div>
          )}
        </div>

        {/* Right: Actions & Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenPresets}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-600/20 hover:bg-brand-600/30 text-brand-100 border border-brand-500/40 text-xs font-medium transition"
          >
            <Sparkles className="h-3.5 w-3.5 text-brand-100" />
            <span>Demo Scenarios</span>
          </button>

          <button
            onClick={onOpenMetrics}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-elevated hover:bg-surface-highlight text-gray-300 border border-surface-highlight text-xs font-medium transition"
          >
            <BarChart3 className="h-3.5 w-3.5 text-gray-400" />
            <span className="hidden md:inline">Telemetry</span>
          </button>

          <div className="flex items-center gap-1 bg-surface-elevated px-2 py-1 rounded-lg border border-surface-highlight">
            <Languages className="h-3.5 w-3.5 text-gray-400" />
            <select
              value={activeLanguage}
              onChange={(e) => onLanguageChange(e.target.value as SupportedLanguage)}
              className="bg-transparent text-xs text-gray-200 outline-none cursor-pointer"
            >
              <option value="en" className="bg-surface text-gray-200">English</option>
              <option value="hi" className="bg-surface text-gray-200">?????? (Hindi)</option>
              <option value="kn" className="bg-surface text-gray-200">????? (Kannada)</option>
              <option value="te" className="bg-surface text-gray-200">?????? (Telugu)</option>
            </select>
          </div>
        </div>
      </div>
    </header>
  );
};
