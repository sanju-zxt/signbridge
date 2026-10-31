'use client';
/**
 * DataCollectionStudio.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Development-only training-data collection studio for SignBridge2Vision.
 *
 * Integrates with the existing MediaPipe webcam stream to record real 30-frame
 * sequences of 3D landmarks for small-vocabulary ISL prototype training.
 *
 * Adheres to official ISLRTC standards with authoritative sign reference cards,
 * pre-record countdown, frame capture progress bar, post-capture review/discard,
 * session tally dashboard, and one-click JSON/CSV dataset export.
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { RawLandmarkFrame, CollectionSequence, ISL_VOCABULARY, ASL_VOCABULARY } from '@signbridge/types';
import {
  PROTOTYPE_VOCABULARY,
  PrototypeSignInfo,
  validateSequence,
  ValidationReport,
} from '../../lib/gesture/datasetValidator';
import {
  Video,
  Play,
  RotateCcw,
  CheckCircle2,
  XCircle,
  Download,
  FileSpreadsheet,
  FileJson,
  BookOpen,
  Info,
  Award,
  Layers,
  Database,
  RefreshCw,
  Hand,
  Check,
  AlertTriangle,
} from 'lucide-react';

interface DataCollectionStudioProps {
  rawFrame: RawLandmarkFrame | null;
  sessionId?: string;
}

export default function DataCollectionStudio({
  rawFrame,
  sessionId = `session_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`,
}: DataCollectionStudioProps) {
  const [selectedSign, setSelectedSign] = useState<string>('HELLO');
  const [customSignInput, setCustomSignInput] = useState<string>('');
  const [activeSigner, setActiveSigner] = useState<string>('studio_signer_01');
  const [recordingLighting, setRecordingLighting] = useState<string>('Webcam Natural');
  const [recordingState, setRecordingState] = useState<'idle' | 'countdown' | 'recording' | 'review'>('idle');
  const [countdown, setCountdown] = useState<number>(3);
  const [recordedFrames, setRecordedFrames] = useState<RawLandmarkFrame[]>([]);
  const [sampleCounts, setSampleCounts] = useState<Record<string, number>>({});
  const [sessionSequences, setSessionSequences] = useState<CollectionSequence[]>([]);
  const [pendingSequence, setPendingSequence] = useState<CollectionSequence | null>(null);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const recordingStartRef = useRef<number>(0);
  const framesBufferRef = useRef<RawLandmarkFrame[]>([]);

  // Fetch disk sample counts on mount
  const refreshDiskCounts = useCallback(async () => {
    try {
      const res = await fetch('/api/dataset');
      const data = await res.json();
      if (data.success && data.counts) {
        setSampleCounts(data.counts);
      }
    } catch {
      // Ignore background fetch error
    }
  }, []);

  useEffect(() => {
    refreshDiskCounts();
  }, [refreshDiskCounts]);

  // Lookup authoritative ISLRTC definition
  const currentSignDef: PrototypeSignInfo = useMemo(() => {
    const found = PROTOTYPE_VOCABULARY.find((v) => v.sign === selectedSign);
    if (found) return found;

    // Check full ISL vocabulary first
    const islVocab = ISL_VOCABULARY.find((v) => v.sign === selectedSign);
    if (islVocab) {
      return {
        sign: islVocab.sign,
        label: islVocab.label,
        category: (islVocab.category as any) || 'needs',
        islrtcDescription: islVocab.description,
        handsRequired: islVocab.keyFeatures.hands,
        dominantHand: islVocab.keyFeatures.dominantHand as any,
        location: (islVocab.keyFeatures.location as any) || 'chest',
        movement: (islVocab.keyFeatures.primaryMotion as any) || 'linear',
        tips: islVocab.nonManualMarkers || `Perform canonical ISL gesture: ${islVocab.canonicalSentence}`,
      };
    }

    // Check legacy ASL vocabulary as fallback
    const fullVocab = ASL_VOCABULARY.find((v) => v.sign === selectedSign);
    if (fullVocab) {
      return {
        sign: fullVocab.sign,
        label: fullVocab.label,
        category: (fullVocab.category as any) || 'needs',
        islrtcDescription: fullVocab.description,
        handsRequired: fullVocab.keyFeatures.hands,
        dominantHand: fullVocab.keyFeatures.dominantHand as any,
        location: fullVocab.keyFeatures.location as any,
        movement: fullVocab.keyFeatures.primaryMotion as any,
        tips: `Perform with canonical movement: ${fullVocab.canonicalSentence}`,
      };
    }

    return {
      sign: selectedSign,
      label: selectedSign,
      category: 'needs',
      islrtcDescription: 'Custom gesture defined by user.',
      handsRequired: 'single',
      dominantHand: 'right',
      location: 'neutral_space',
      movement: 'linear',
      tips: 'Hold handshape clearly in front of camera.',
    };
  }, [selectedSign]);

  // Start 3-second countdown
  const startRecordingCountdown = useCallback(() => {
    if (recordingState !== 'idle') return;
    setRecordingState('countdown');
    setCountdown(3);
    framesBufferRef.current = [];
    setRecordedFrames([]);
    setPendingSequence(null);
    setStatusMessage(null);

    let count = 3;
    const interval = setInterval(() => {
      count--;
      if (count > 0) {
        setCountdown(count);
      } else {
        clearInterval(interval);
        setRecordingState('recording');
        recordingStartRef.current = Date.now();
      }
    }, 1000);
  }, [recordingState]);

  // Global Spacebar listener to trigger recording
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && (e.target as HTMLElement)?.tagName !== 'INPUT') {
        e.preventDefault();
        if (recordingState === 'idle') {
          startRecordingCountdown();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [recordingState, startRecordingCountdown]);

  // Accumulate frames during recording
  useEffect(() => {
    if (recordingState !== 'recording' || !rawFrame) return;

    framesBufferRef.current.push(rawFrame);
    setRecordedFrames([...framesBufferRef.current]);

    if (framesBufferRef.current.length >= 30) {
      const durationMs = Date.now() - recordingStartRef.current;
      const finishedFrames = [...framesBufferRef.current];

      const currentCount = sampleCounts[selectedSign] || 0;
      const seq: CollectionSequence = {
        sign: selectedSign,
        sessionId,
        sampleIndex: currentCount + 1,
        frames: finishedFrames,
        recordedAt: new Date().toISOString(),
        durationMs,
      };

      const report = validateSequence(seq, 30);
      setPendingSequence(seq);
      setValidationReport(report);
      setRecordingState('review');
    }
  }, [recordingState, rawFrame, selectedSign, sessionId, sampleCounts]);

  // Keep and save sample
  const handleKeepSample = async () => {
    if (!pendingSequence) return;
    setIsSaving(true);
    try {
      const res = await fetch('/api/dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sequence: pendingSequence,
          signerId: activeSigner,
          recordingConditions: {
            lighting: recordingLighting,
            background: 'indoor_ergonomic',
          },
        }),
      });
      const data = await res.json();

      if (data.success) {
        setSessionSequences((prev) => [...prev, pendingSequence]);
        setSampleCounts((prev) => ({
          ...prev,
          [pendingSequence.sign]: (prev[pendingSequence.sign] || 0) + 1,
        }));
        setStatusMessage(`✅ Saved sequence for ${pendingSequence.sign} (Total: ${data.count})`);
      } else {
        setStatusMessage(`❌ Error saving: ${data.error}`);
      }
    } catch (err: any) {
      setStatusMessage(`❌ Network error saving sequence: ${err?.message}`);
    } finally {
      setIsSaving(false);
      setPendingSequence(null);
      setValidationReport(null);
      setRecordedFrames([]);
      framesBufferRef.current = [];
      setRecordingState('idle');
    }
  };

  // Discard sample
  const handleDiscardSample = () => {
    setPendingSequence(null);
    setValidationReport(null);
    setRecordedFrames([]);
    framesBufferRef.current = [];
    setRecordingState('idle');
    setStatusMessage('Sample discarded. Ready for re-recording.');
  };

  // Export Dataset JSON
  const handleExportJSON = () => {
    const exportData = sessionSequences.length > 0 ? sessionSequences : [];
    if (exportData.length === 0) {
      alert('No recorded sequences in current session to export yet.');
      return;
    }
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `isl_dataset_${selectedSign.toLowerCase()}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export Dataset CSV
  const handleExportCSV = () => {
    if (sessionSequences.length === 0) {
      alert('No recorded sequences in current session to export yet.');
      return;
    }

    // Build headers: sign, sessionId, sampleIndex, frameIndex, hand, landmarkIndex, x, y, z
    const headers = ['sign', 'sessionId', 'sampleIndex', 'recordedAt', 'durationMs', 'frameIndex', 'timestamp', 'hand', 'landmarkIdx', 'x', 'y', 'z'];
    const rows: string[] = [headers.join(',')];

    for (const seq of sessionSequences) {
      seq.frames.forEach((frame, fIdx) => {
        if (frame.rightHand) {
          frame.rightHand.forEach((lm, lmIdx) => {
            rows.push(
              `"${seq.sign}","${seq.sessionId}",${seq.sampleIndex},"${seq.recordedAt}",${seq.durationMs},${fIdx},${frame.timestamp},"right",${lmIdx},${lm.x.toFixed(5)},${lm.y.toFixed(5)},${lm.z.toFixed(5)}`
            );
          });
        }
        if (frame.leftHand) {
          frame.leftHand.forEach((lm, lmIdx) => {
            rows.push(
              `"${seq.sign}","${seq.sessionId}",${seq.sampleIndex},"${seq.recordedAt}",${seq.durationMs},${fIdx},${frame.timestamp},"left",${lmIdx},${lm.x.toFixed(5)},${lm.y.toFixed(5)},${lm.z.toFixed(5)}`
            );
          });
        }
      });
    }

    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `isl_dataset_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 flex flex-col gap-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-amber-500/10 text-amber-400 rounded-xl border border-amber-500/20">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              ISL Training Data Collection Studio
              <span className="text-[11px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30">
                Dev Tool
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Collect 30-frame MediaPipe landmark sequences adhering to ISLRTC standards for prototype model training.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={refreshDiskCounts}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition"
            title="Refresh sample count from disk"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh Counts
          </button>
          <div className="text-xs font-mono bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-400">
            Session: <span className="text-slate-200">{sessionId}</span>
          </div>
        </div>
      </div>

      {statusMessage && (
        <div className="text-xs px-3.5 py-2.5 rounded-xl bg-slate-800/80 border border-slate-700 text-slate-200 flex items-center justify-between">
          <span>{statusMessage}</span>
          <button onClick={() => setStatusMessage(null)} className="text-slate-400 hover:text-slate-200 text-xs font-bold">
            ✕
          </button>
        </div>
      )}

      {/* Main Grid: Sign Selector & Reference Card (Left) vs Recording Monitor (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Sign Selection & Reference Standard */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
            <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center justify-between">
              <span>Target Sign Vocabulary</span>
              <span className="text-slate-500 font-normal">7 Target Classes</span>
            </label>

            {/* V5 Authentic Signer & Lighting Metadata */}
            <div className="grid grid-cols-2 gap-2 p-2 rounded-lg bg-slate-900/60 border border-slate-800">
              <div>
                <label className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Authentic Signer</label>
                <select
                  value={activeSigner}
                  onChange={(e) => setActiveSigner(e.target.value)}
                  disabled={recordingState !== 'idle'}
                  className="w-full bg-slate-900 border border-slate-700/80 rounded-md text-xs text-slate-200 px-2 py-1 focus:border-amber-400 outline-none"
                >
                  <option value="studio_signer_01">Signer 01 (Deaf Contributor A)</option>
                  <option value="studio_signer_02">Signer 02 (Fluent Signer B)</option>
                  <option value="studio_signer_03">Signer 03 (Fluent Signer C)</option>
                  <option value="studio_signer_04">Signer 04 (Val Held-out D)</option>
                  <option value="studio_signer_05">Signer 05 (Test Held-out E)</option>
                  <option value="studio_signer_06">Signer 06 (Test Held-out F)</option>
                </select>
              </div>
              <div>
                <label className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Lighting</label>
                <select
                  value={recordingLighting}
                  onChange={(e) => setRecordingLighting(e.target.value)}
                  disabled={recordingState !== 'idle'}
                  className="w-full bg-slate-900 border border-slate-700/80 rounded-md text-xs text-slate-200 px-2 py-1 focus:border-amber-400 outline-none"
                >
                  <option value="Webcam Natural">Webcam Natural</option>
                  <option value="Warm Indoor">Warm Indoor</option>
                  <option value="Low Contrast">Low Contrast</option>
                  <option value="Studio Direct">Studio Direct</option>
                </select>
              </div>
            </div>

            {/* Quick Chips */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
              {PROTOTYPE_VOCABULARY.map((v) => {
                const count = sampleCounts[v.sign] || 0;
                const isSelected = selectedSign === v.sign;
                return (
                  <button
                    key={v.sign}
                    onClick={() => {
                      if (recordingState === 'idle') setSelectedSign(v.sign);
                    }}
                    disabled={recordingState !== 'idle'}
                    className={`flex flex-col items-start p-2 rounded-lg border text-left transition ${
                      isSelected
                        ? 'bg-amber-500/20 border-amber-400 text-amber-200 shadow-sm'
                        : 'bg-slate-900/80 border-slate-800 hover:border-slate-700 text-slate-300'
                    } ${recordingState !== 'idle' ? 'opacity-60 cursor-not-allowed' : ''}`}
                  >
                    <span className="font-bold text-xs truncate w-full">{v.sign}</span>
                    <span className="text-[10px] text-slate-400 truncate w-full">{v.label.split('/')[0]}</span>
                    <span
                      className={`text-[9px] mt-1 px-1.5 py-0.2 rounded font-mono font-semibold ${
                        count >= 10
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : count >= 3
                          ? 'bg-amber-500/20 text-amber-300'
                          : 'bg-rose-500/20 text-rose-300'
                      }`}
                    >
                      {count} samples
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Custom sign option */}
            <div className="pt-2 border-t border-slate-800/80 flex gap-2">
              <input
                type="text"
                placeholder="Or custom sign (e.g. FOOD, PLEASE)..."
                value={customSignInput}
                onChange={(e) => setCustomSignInput(e.target.value.toUpperCase())}
                disabled={recordingState !== 'idle'}
                className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-amber-400"
              />
              <button
                type="button"
                onClick={() => {
                  if (customSignInput.trim()) {
                    setSelectedSign(customSignInput.trim());
                    setCustomSignInput('');
                  }
                }}
                disabled={!customSignInput.trim() || recordingState !== 'idle'}
                className="bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 px-3 py-1.5 rounded-lg text-xs font-semibold transition"
              >
                Set
              </button>
            </div>
          </div>

          {/* Authoritative ISLRTC Reference Card */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-amber-300 flex items-center gap-1.5 uppercase tracking-wider">
                <BookOpen className="w-3.5 h-3.5" /> ISLRTC Reference Standard
              </h3>
              <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                {currentSignDef.category}
              </span>
            </div>

            <div className="bg-slate-900/90 rounded-lg p-3 border border-slate-800 flex flex-col gap-2">
              <div className="flex items-baseline justify-between">
                <span className="text-base font-extrabold text-slate-100">{currentSignDef.sign}</span>
                <span className="text-xs text-slate-400">{currentSignDef.label}</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed italic">
                "{currentSignDef.islrtcDescription}"
              </p>

              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-[11px]">
                <div>
                  <span className="text-slate-500">Hands:</span>{' '}
                  <span className="font-semibold text-slate-200 uppercase">{currentSignDef.handsRequired}</span>
                </div>
                <div>
                  <span className="text-slate-500">Dominant:</span>{' '}
                  <span className="font-semibold text-slate-200 uppercase">{currentSignDef.dominantHand}</span>
                </div>
                <div>
                  <span className="text-slate-500">Location:</span>{' '}
                  <span className="font-semibold text-slate-200">{currentSignDef.location.replace('_', ' ')}</span>
                </div>
                <div>
                  <span className="text-slate-500">Motion:</span>{' '}
                  <span className="font-semibold text-slate-200 uppercase">{currentSignDef.movement}</span>
                </div>
              </div>
            </div>

            <div className="text-[11px] text-amber-200/80 bg-amber-950/30 border border-amber-900/40 p-2.5 rounded-lg flex items-start gap-2">
              <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-amber-400" />
              <span>{currentSignDef.tips}</span>
            </div>
          </div>
        </div>

        {/* Right Column: Recording HUD, Progress Bar & Review Drawer */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          {/* Active Recording Monitor Card */}
          <div className="bg-slate-950/60 p-5 rounded-xl border border-slate-800 flex flex-col gap-4 relative overflow-hidden">
            {/* Countdown Overlay */}
            {recordingState === 'countdown' && (
              <div className="absolute inset-0 z-20 bg-slate-950/90 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                <div className="w-20 h-20 rounded-full border-4 border-amber-400 flex items-center justify-center animate-ping text-3xl font-extrabold text-amber-300">
                  {countdown}
                </div>
                <div className="text-center">
                  <p className="text-sm font-bold text-amber-300">Get Ready to Sign "{selectedSign}"</p>
                  <p className="text-xs text-slate-400 mt-1">Recording starts in {countdown}s...</p>
                </div>
              </div>
            )}

            {/* Target Display Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Target Sign</span>
                <h3 className="text-2xl font-black text-amber-400">{selectedSign}</h3>
              </div>

              <div className="flex items-center gap-2">
                <div
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${
                    rawFrame?.rightHand || rawFrame?.leftHand
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                  }`}
                >
                  <Hand className="w-3.5 h-3.5" />
                  <span>
                    {rawFrame?.rightHand && rawFrame?.leftHand
                      ? 'Both Hands In Frame'
                      : rawFrame?.rightHand
                      ? 'Right Hand In Frame'
                      : rawFrame?.leftHand
                      ? 'Left Hand In Frame'
                      : 'No Hand In View'}
                  </span>
                </div>
              </div>
            </div>

            {/* Recording Controls */}
            {recordingState === 'idle' && (
              <div className="flex flex-col items-center justify-center py-6 gap-3">
                <button
                  onClick={startRecordingCountdown}
                  className="bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold px-6 py-3 rounded-xl shadow-lg shadow-amber-500/20 flex items-center gap-2 text-sm transition-all transform active:scale-95"
                >
                  <Play className="w-4 h-4 fill-current" />
                  Record 30-Frame Sample
                </button>
                <p className="text-[11px] text-slate-400 flex items-center gap-1.5">
                  Tip: Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700">Spacebar</kbd> anytime to record.
                </p>
              </div>
            )}

            {/* Progress Bar (During Recording) */}
            {recordingState === 'recording' && (
              <div className="flex flex-col gap-3 py-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-amber-400 font-bold flex items-center gap-2 animate-pulse">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping" />
                    RECORDING SEQUENCE...
                  </span>
                  <span className="font-mono text-slate-300 font-semibold">
                    {recordedFrames.length} / 30 Frames
                  </span>
                </div>

                <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden border border-slate-700">
                  <div
                    className="bg-amber-400 h-full rounded-full transition-all duration-75 ease-out"
                    style={{ width: `${Math.min(100, (recordedFrames.length / 30) * 100)}%` }}
                  />
                </div>

                <p className="text-[11px] text-slate-400 text-center">
                  Perform the gesture smoothly. Capturing normalized 3D landmarks...
                </p>
              </div>
            )}

            {/* Post-Capture Review Drawer */}
            {recordingState === 'review' && pendingSequence && (
              <div className="bg-slate-900 border border-amber-500/30 rounded-xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <h4 className="text-xs font-bold text-amber-300 uppercase tracking-wider flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Sample Captured (30 Frames)
                  </h4>
                  <span className="text-xs font-mono text-slate-400">
                    Duration: {pendingSequence.durationMs}ms
                  </span>
                </div>

                {validationReport && (
                  <div className="text-xs space-y-1.5 bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sign:</span>
                      <span className="font-bold text-slate-200">{pendingSequence.sign}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Frames Captured:</span>
                      <span className="font-mono text-slate-200">{validationReport.frameCount} / 30</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Hand Presence Coverage:</span>
                      <span
                        className={`font-mono font-bold ${
                          validationReport.handsPresentFraction >= 0.7
                            ? 'text-emerald-400'
                            : validationReport.handsPresentFraction >= 0.4
                            ? 'text-amber-400'
                            : 'text-rose-400'
                        }`}
                      >
                        {(validationReport.handsPresentFraction * 100).toFixed(0)}%
                      </span>
                    </div>

                    {validationReport.warnings.length > 0 && (
                      <div className="text-[11px] text-amber-300/90 pt-1 flex items-start gap-1">
                        <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" />
                        <span>{validationReport.warnings.join(' ')}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Review Action Buttons */}
                <div className="flex items-center justify-end gap-2 pt-2">
                  <button
                    onClick={handleDiscardSample}
                    disabled={isSaving}
                    className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-rose-900/40 hover:text-rose-200 hover:border-rose-700 text-slate-300 px-3.5 py-2 rounded-lg border border-slate-700 font-semibold transition"
                  >
                    <XCircle className="w-3.5 h-3.5" /> Discard / Re-record
                  </button>

                  <button
                    onClick={handleKeepSample}
                    disabled={isSaving}
                    className="flex items-center gap-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg font-bold shadow-md shadow-emerald-600/20 transition disabled:opacity-50"
                  >
                    <Check className="w-4 h-4" />
                    {isSaving ? 'Saving...' : 'Keep & Save Sample'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Session Sample Tally Dashboard */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-sky-400" /> Session & Disk Dataset Tally
              </h3>
              <span className="text-[11px] font-mono text-slate-400">
                Total Samples: <strong className="text-slate-100">{Object.values(sampleCounts).reduce((a, b) => a + b, 0)}</strong>
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {PROTOTYPE_VOCABULARY.map((v) => {
                const count = sampleCounts[v.sign] || 0;
                return (
                  <div
                    key={v.sign}
                    className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800/80 flex flex-col justify-between"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-200 truncate">{v.sign}</span>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold ${
                          count >= 10
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : count >= 3
                            ? 'bg-amber-500/20 text-amber-300'
                            : 'bg-rose-500/20 text-rose-300'
                        }`}
                      >
                        {count >= 10 ? 'Ready' : count >= 3 ? 'Min' : 'Need Data'}
                      </span>
                    </div>
                    <div className="text-lg font-mono font-black text-slate-100 mt-1">
                      {count} <span className="text-[10px] font-normal text-slate-500">seqs</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Export Actions */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800">
              <span className="text-[11px] text-slate-400">
                Session Recorded: <strong className="text-amber-300 font-mono">{sessionSequences.length}</strong> sequences
              </span>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleExportCSV}
                  disabled={sessionSequences.length === 0}
                  className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 font-medium transition"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" /> Export CSV
                </button>

                <button
                  onClick={handleExportJSON}
                  disabled={sessionSequences.length === 0}
                  className="flex items-center gap-1.5 text-xs bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-slate-950 px-3 py-1.5 rounded-lg font-bold shadow transition"
                >
                  <FileJson className="w-3.5 h-3.5 fill-current" /> Export JSON
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
