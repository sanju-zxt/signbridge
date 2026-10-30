/**
 * mediapipe.ts
 * MediaPipe HandLandmarker singleton for real-time hand detection.
 */
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { Landmark3D } from '@signbridge2vision/types';

const WASM_PATH  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export interface DetectHandsResult {
  rightHand: Landmark3D[] | null;
  leftHand: Landmark3D[] | null;
  rightHandConfidence: number;
  leftHandConfidence: number;
}

class MediaPipeHandsService {
  private _landmarker: HandLandmarker | null = null;
  private _initialized = false;
  private _initPromise: Promise<void> | null = null;

  get isInitialized() { return this._initialized; }

  async init(): Promise<void> {
    if (this._initialized) return;
    if (this._initPromise) return this._initPromise;

    this._initPromise = (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
        this._landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        this._initialized = true;
        console.log('[MediaPipe] HandLandmarker ready (GPU)');
      } catch {
        console.warn('[MediaPipe] GPU failed, trying CPU');
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
        this._landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        this._initialized = true;
        console.log('[MediaPipe] HandLandmarker ready (CPU fallback)');
      }
    })();
    return this._initPromise;
  }

  detectHands(video: HTMLVideoElement, timestampMs: number): DetectHandsResult {
    const empty: DetectHandsResult = { rightHand: null, leftHand: null, rightHandConfidence: 0, leftHandConfidence: 0 };
    if (!this._landmarker || !this._initialized) return empty;
    try {
      const result = this._landmarker.detectForVideo(video, timestampMs);
      const out: DetectHandsResult = { rightHand: null, leftHand: null, rightHandConfidence: 0, leftHandConfidence: 0 };
      for (let i = 0; i < (result.handedness?.length ?? 0); i++) {
        // MediaPipe mirrors handedness in front-facing camera: 'Right' in API = left on screen
        const label = result.handedness[i]?.[0]?.categoryName ?? '';
        const conf  = result.handedness[i]?.[0]?.score ?? 0;
        const lms: Landmark3D[] = (result.landmarks[i] ?? []).map(lm => ({ x: lm.x, y: lm.y, z: lm.z ?? 0 }));
        if (label === 'Right') { out.leftHand  = lms; out.leftHandConfidence  = conf; }
        else                    { out.rightHand = lms; out.rightHandConfidence = conf; }
      }
      return out;
    } catch (err) {
      console.error('[MediaPipe] detectHands error:', err);
      return empty;
    }
  }
}

export const mediaPipeHands = new MediaPipeHandsService();
export async function initMediaPipe() { return mediaPipeHands.init(); }