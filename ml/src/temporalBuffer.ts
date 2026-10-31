/**
 * temporalBuffer.ts
 * -----------------------------------------------------------------------------
 * Rolling temporal frame buffer.  Maintains a sliding window of normalised
 * feature frames ready for GRU inference.
 */

import { NormalizedFeatureFrame, PipelineConfig, DEFAULT_PIPELINE_CONFIG } from '@signbridge/types';
import { FEATURE_DIM, extractFrameFeatures } from './features/landmarkFeatures';
import type { Landmark3D } from '@signbridge/types';

export class TemporalBuffer {
  private readonly windowSize: number;
  private readonly minSequenceLength: number;
  private frames: NormalizedFeatureFrame[] = [];
  private handMissingCount = 0;
  private readonly maxHandMissing = 10; // frames to allow before clearing

  constructor(config: Partial<PipelineConfig> = {}) {
    const cfg = { ...DEFAULT_PIPELINE_CONFIG, ...config };
    this.windowSize = cfg.windowSize;
    this.minSequenceLength = cfg.minSequenceLength;
  }

  /**
   * Add a new frame.  Returns true if the buffer is ready for inference.
   */
  push(
    timestamp: number,
    rightHand: Landmark3D[] | null,
    leftHand: Landmark3D[] | null
  ): boolean {
    const hasHand = rightHand !== null || leftHand !== null;

    if (!hasHand) {
      this.handMissingCount++;
      if (this.handMissingCount >= this.maxHandMissing) {
        this.clear();
      }
      return false;
    }

    this.handMissingCount = 0;
    const features = extractFrameFeatures(rightHand, leftHand);

    const frame: NormalizedFeatureFrame = {
      timestamp,
      features,
      hasHand,
      rightHandPresent: rightHand !== null,
      leftHandPresent: leftHand !== null,
    };

    this.frames.push(frame);

    // Keep only the latest windowSize frames
    if (this.frames.length > this.windowSize) {
      this.frames.shift();
    }

    return this.isReady();
  }

  /** Returns true when enough frames have accumulated for inference */
  isReady(): boolean {
    return this.frames.length >= this.minSequenceLength;
  }

  /** Returns true when the buffer is completely full */
  isFull(): boolean {
    return this.frames.length >= this.windowSize;
  }

  /** How many frames are currently buffered */
  get length(): number {
    return this.frames.length;
  }

  /** Get the window size */
  get size(): number {
    return this.windowSize;
  }

  /**
   * Build a flat Float32Array of shape [windowSize × FEATURE_DIM] (row-major)
   * for passing to the GRU model.  Zero-padded if buffer not yet full.
   */
  getFeatureMatrix(): Float32Array {
    const matrix = new Float32Array(this.windowSize * FEATURE_DIM);
    const paddingOffset = this.windowSize - this.frames.length;

    for (let i = 0; i < this.frames.length; i++) {
      matrix.set(this.frames[i].features, (paddingOffset + i) * FEATURE_DIM);
    }

    return matrix;
  }

  /**
   * Get all current frames (for data collection export).
   */
  getFrames(): NormalizedFeatureFrame[] {
    return [...this.frames];
  }

  /** Reset the buffer (e.g., after a recognised gesture or lost tracking) */
  clear(): void {
    this.frames = [];
    this.handMissingCount = 0;
  }

  /** How full the buffer is as a fraction [0..1] */
  get fillRatio(): number {
    return this.frames.length / this.windowSize;
  }
}
