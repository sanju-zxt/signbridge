const fs = require('fs');
const path = require('path');

function save(relPath, content) {
  const full = path.resolve(__dirname, '..', relPath);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content, 'utf8');
  console.log('Saved:', relPath);
}

// 1. apps/web/src/components/Header.tsx
save('apps/web/src/components/Header.tsx', `'use client';

import React from 'react';
import { Activity, Radio, Volume2, ShieldCheck, Languages, BarChart3, Sparkles } from 'lucide-react';
import { SystemState, SupportedLanguage, LatencySummary } from '@signbridge2vision/types';

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
              <span className="font-bold text-lg tracking-tight text-white">SignBridge<span className="text-isl-teal">2</span>Vision</span>
              <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded bg-brand-500/20 text-brand-100 border border-brand-500/30">
                Multimodal AI
              </span>
            </div>
            <p className="text-xs text-gray-400 hidden sm:block">ISL & Speech Semantic Communication Platform</p>
          </div>
        </div>

        {/* Center: Live System Status Pill */}
        <div className="flex items-center gap-2">
          <div className={\`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium \${badge.bg}\`}>
            <span className="relative flex h-2 w-2">
              <span className={\`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 \${isConnected ? 'bg-emerald-400' : 'bg-rose-400'}\`}></span>
              <span className={\`relative inline-flex rounded-full h-2 w-2 \${isConnected ? 'bg-emerald-500' : 'bg-rose-500'}\`}></span>
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
`);

// 2. apps/web/src/components/LiveCameraView.tsx
save('apps/web/src/components/LiveCameraView.tsx', `'use client';

import React from 'react';
import { Camera, CameraOff, Sparkles, RefreshCw, Eye, EyeOff, Zap } from 'lucide-react';
import { ISL_VOCABULARY } from '@signbridge2vision/types';

interface LiveCameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  isCameraActive: boolean;
  cameraError: string | null;
  activePrediction: { sign: string; label: string; confidence: number } | null;
  isDebouncing: boolean;
  fps: number;
  isSimulatedMode: boolean;
  onStartCamera: () => void;
  onStopCamera: () => void;
  onTriggerSign: (signId: string) => void;
}

export const LiveCameraView: React.FC<LiveCameraViewProps> = ({
  videoRef,
  canvasRef,
  isCameraActive,
  cameraError,
  activePrediction,
  isDebouncing,
  fps,
  isSimulatedMode,
  onStartCamera,
  onStopCamera,
  onTriggerSign
}) => {
  return (
    <div className="flex flex-col bg-surface rounded-xl border border-surface-highlight/70 overflow-hidden shadow-xl">
      <div className="flex items-center justify-between px-4 py-2.5 bg-surface-elevated/70 border-b border-surface-highlight/60">
        <div className="flex items-center gap-2">
          <div className="p-1 rounded bg-teal-500/20 text-teal-400">
            <Camera className="h-4 w-4" />
          </div>
          <span className="text-xs font-semibold text-gray-200 uppercase tracking-wider">ISL Vision Perception</span>
        </div>
        <div className="flex items-center gap-2">
          {fps > 0 && (
            <span className="text-[11px] font-mono text-gray-400 bg-surface px-2 py-0.5 rounded border border-surface-highlight/40">
              {fps} FPS
            </span>
          )}
          <button
            onClick={isCameraActive ? onStopCamera : onStartCamera}
            className={\`px-2.5 py-1 rounded text-xs font-medium transition flex items-center gap-1.5 \${
              isCameraActive
                ? 'bg-rose-500/20 text-rose-300 hover:bg-rose-500/30 border border-rose-500/40'
                : 'bg-teal-500/20 text-teal-300 hover:bg-teal-500/30 border border-teal-500/40'
            }\`}
          >
            {isCameraActive ? <CameraOff className="h-3 w-3" /> : <Camera className="h-3 w-3" />}
            <span>{isCameraActive ? 'Stop Cam' : 'Start Cam'}</span>
          </button>
        </div>
      </div>

      <div className="relative aspect-[4/3] sm:aspect-video w-full bg-black/80 flex items-center justify-center overflow-hidden">
        <video
          ref={videoRef}
          playsInline
          muted
          className={\`w-full h-full object-cover transform -scale-x-100 \${isCameraActive ? 'block' : 'hidden'}\`}
        />

        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100 z-10"
        />

        {!isCameraActive && (
          <div className="flex flex-col items-center justify-center p-6 text-center z-0">
            <div className="h-16 w-16 rounded-2xl bg-surface-elevated border border-surface-highlight flex items-center justify-center mb-3">
              <Camera className="h-8 w-8 text-gray-500" />
            </div>
            <h4 className="text-sm font-semibold text-gray-300 mb-1">Live Camera Ready</h4>
            <p className="text-xs text-gray-400 max-w-xs mb-4">
              Connect webcam for real-time 3D hand landmark tracking, or trigger gestures below.
            </p>
            <button
              onClick={onStartCamera}
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-teal-600 to-teal-500 hover:from-teal-500 hover:to-teal-400 text-white font-medium text-xs shadow-lg shadow-teal-500/20 transition flex items-center gap-2"
            >
              <Camera className="h-4 w-4" />
              <span>Enable Live Camera</span>
            </button>
          </div>
        )}

        {activePrediction && activePrediction.confidence > 0.35 && (
          <div className="absolute top-3 left-3 z-20 flex flex-col gap-1">
            <div className="flex items-center gap-2 bg-surface/90 backdrop-blur px-3 py-1.5 rounded-lg border border-teal-500/40 shadow-lg">
              <Zap className="h-4 w-4 text-teal-400 animate-pulse" />
              <div className="flex flex-col">
                <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Live Recognition</span>
                <span className="text-sm font-bold text-white tracking-tight">{activePrediction.label}</span>
              </div>
              <div className="ml-2 pl-2 border-l border-surface-highlight flex flex-col items-end">
                <span className="text-[10px] text-gray-400 font-mono">CONF</span>
                <span className="text-xs font-bold text-teal-400 font-mono">
                  {(activePrediction.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {isDebouncing && (
              <div className="flex items-center gap-1.5 bg-brand-600/80 backdrop-blur px-2.5 py-1 rounded text-[11px] text-white font-medium animate-pulse">
                <RefreshCw className="h-3 w-3 animate-spin" />
                <span>Finalizing sign...</span>
              </div>
            )}
          </div>
        )}

        <div className="absolute bottom-3 left-3 z-20 bg-surface/80 backdrop-blur px-2 py-0.5 rounded text-[10px] text-gray-400 border border-surface-highlight/40">
          Client-side landmarks only • Zero video stored
        </div>
      </div>

      <div className="p-3 bg-surface-elevated/40 border-t border-surface-highlight/50 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1">
            <Sparkles className="h-3 w-3 text-teal-400" />
            <span>Constrained ISL Gesture Shortcuts</span>
          </span>
          <span className="text-[10px] text-gray-500">1-Click Sign Simulator</span>
        </div>

        <div className="grid grid-cols-3 sm:grid-cols-5 gap-1.5">
          {ISL_VOCABULARY.slice(0, 10).map((item) => (
            <button
              key={item.id}
              onClick={() => onTriggerSign(item.sign)}
              className="px-2 py-1.5 rounded bg-surface hover:bg-surface-highlight border border-surface-highlight/70 text-left transition flex flex-col group"
            >
              <span className="text-xs font-semibold text-teal-400 group-hover:text-teal-300 truncate">
                {item.sign}
              </span>
              <span className="text-[10px] text-gray-400 truncate">
                {item.label.split('/')[0]}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
`);
