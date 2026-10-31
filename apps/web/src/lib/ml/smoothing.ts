/**
 * smoothing.ts
 * EMA smoothing and majority-vote buffer for temporal prediction stabilisation.
 */
import type { RecognitionResult } from '@signbridge/types';

export class ExponentialSmoother {
  private smoothed: Float32Array;
  private readonly alpha: number;
  constructor(numClasses: number, alpha: number = 0.4) {
    this.smoothed = new Float32Array(numClasses);
    this.alpha = alpha;
  }
  update(probs: number[]): number[] {
    const a = this.alpha, b = 1 - a;
    for (let i = 0; i < probs.length && i < this.smoothed.length; i++) {
      this.smoothed[i] = a * probs[i] + b * this.smoothed[i];
    }
    return Array.from(this.smoothed);
  }
  reset() { this.smoothed.fill(0); }
  get current(): Float32Array { return this.smoothed; }
}

export class VotingBuffer {
  private buf: string[] = [];
  constructor(private readonly windowSize: number = 8) {}
  push(sign: string): void {
    this.buf.push(sign);
    if (this.buf.length > this.windowSize) this.buf.shift();
  }
  getTopVote(): { sign: string; count: number; ratio: number } | null {
    if (!this.buf.length) return null;
    const counts = new Map<string, number>();
    for (const s of this.buf) counts.set(s, (counts.get(s) ?? 0) + 1);
    let top = '', topN = 0;
    for (const [s, n] of counts) { if (n > topN) { top = s; topN = n; } }
    return { sign: top, count: topN, ratio: topN / this.buf.length };
  }
  clear() { this.buf = []; }
}

export function applySmoothing(raw: number[], smoothed: Float32Array, alpha = 0.4): Float32Array {
  const b = 1 - alpha;
  for (let i = 0; i < Math.min(raw.length, smoothed.length); i++) {
    smoothed[i] = alpha * raw[i] + b * smoothed[i];
  }
  return smoothed;
}

// Satisfy import
export type { RecognitionResult };