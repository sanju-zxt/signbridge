/**
 * recognitionState.ts
 * -----------------------------------------------------------------------------
 * State machine for the real-time ISL recognition pipeline.
 *
 *  IDLE  --? DETECTING --? CLASSIFYING --? RECOGNIZED
 *                ¦                ¦               ¦
 *                ¦                +--? LOW_CONFIDENCE
 *                ¦
 *                +--? LOST_TRACK --? IDLE
 */

import { RecognitionState, PipelineConfig, DEFAULT_PIPELINE_CONFIG } from '@signbridge/types';

export interface StateTransitionEvent {
  from: RecognitionState;
  to: RecognitionState;
  timestamp: number;
  reason?: string;
}

export class RecognitionStateMachine {
  private _state: RecognitionState = 'IDLE';
  private listeners: Array<(event: StateTransitionEvent) => void> = [];
  private readonly config: PipelineConfig;

  // Stable-prediction tracking
  private stablePrediction: string | null = null;
  private stableCount = 0;

  // Last stable result (prevent duplicate emissions for the same sign)
  private lastEmitted: string | null = null;
  private lastEmittedAt = 0;
  private readonly cooldownMs = 2000; // minimum ms between two identical sign emissions

  constructor(config: Partial<PipelineConfig> = {}) {
    this.config = { ...DEFAULT_PIPELINE_CONFIG, ...config };
  }

  get state(): RecognitionState {
    return this._state;
  }

  /** Subscribe to state transitions */
  on(listener: (event: StateTransitionEvent) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private transition(to: RecognitionState, reason?: string): void {
    if (this._state === to) return;
    const from = this._state;
    this._state = to;
    const event: StateTransitionEvent = { from, to, timestamp: Date.now(), reason };
    this.listeners.forEach(l => l(event));
  }

  /** Call when hands first detected in a frame */
  onHandDetected(): void {
    if (this._state === 'IDLE' || this._state === 'LOST_TRACK') {
      this.transition('DETECTING', 'Hand appeared in frame');
    }
  }

  /** Call when the temporal buffer is ready for inference */
  onBufferReady(): void {
    if (this._state === 'DETECTING') {
      this.transition('CLASSIFYING', 'Buffer sufficiently filled');
    }
  }

  /** Call on every inference result */
  onInferenceResult(
    gesture: string,
    confidence: number
  ): { shouldEmit: boolean; isStable: boolean } {
    if (confidence >= this.config.confidenceThreshold) {
      if (gesture === this.stablePrediction) {
        this.stableCount++;
      } else {
        this.stablePrediction = gesture;
        this.stableCount = 1;
      }

      if (this.stableCount >= this.config.stableFrames) {
        this.transition('RECOGNIZED', `Stable prediction: ${gesture} @ ${(confidence * 100).toFixed(1)}%`);

        // Check cooldown to prevent rapid duplicate emissions
        const now = Date.now();
        const isDuplicate =
          this.lastEmitted === gesture &&
          now - this.lastEmittedAt < this.cooldownMs;

        if (!isDuplicate) {
          this.lastEmitted = gesture;
          this.lastEmittedAt = now;
          return { shouldEmit: true, isStable: true };
        }
      }

      return { shouldEmit: false, isStable: this.stableCount >= this.config.stableFrames };
    } else {
      this.stableCount = 0;
      this.stablePrediction = null;
      this.transition('LOW_CONFIDENCE', `Confidence too low: ${(confidence * 100).toFixed(1)}%`);
      return { shouldEmit: false, isStable: false };
    }
  }

  /** Call when hands disappear from frame */
  onHandLost(): void {
    if (this._state !== 'IDLE') {
      this.transition('LOST_TRACK', 'No hands detected');
      this.stablePrediction = null;
      this.stableCount = 0;
    }
  }

  /** Call after a successful recognition to reset for the next sign */
  afterEmit(): void {
    this.stablePrediction = null;
    this.stableCount = 0;
    this.transition('DETECTING', 'Ready for next sign');
  }

  /** Hard reset */
  reset(): void {
    this._state = 'IDLE';
    this.stablePrediction = null;
    this.stableCount = 0;
    this.lastEmitted = null;
    this.lastEmittedAt = 0;
  }

  /** Human-readable description for UI */
  get statusLabel(): string {
    switch (this._state) {
      case 'IDLE':         return 'Position hands in frame';
      case 'DETECTING':    return 'Tracking — hold gesture';
      case 'CLASSIFYING':  return 'Classifying…';
      case 'RECOGNIZED':   return 'Sign recognised ?';
      case 'LOW_CONFIDENCE': return 'Gesture unclear — try again';
      case 'LOST_TRACK':   return 'Hands lost — reposition';
      case 'ERROR':        return 'Pipeline error';
    }
  }
}
