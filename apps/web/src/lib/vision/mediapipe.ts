/**
 * mediapipe.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * MediaPipe Multimodal Vision Service for Real-Time ISL.
 *
 * Implements:
 * 1. HandLandmarker for 21-joint 3D hand tracking (dual hands, 126 dims)
 * 2. FaceLandmarker for 478-mesh facial & eye-gaze tracking (24 non-manual dims)
 * 3. Timestamp-synchronized extraction with graceful GPU -> CPU fallback
 * 4. Backward-compatible detectHands API + multimodal detectMultimodal API
 */

import { HandLandmarker, FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { Landmark3D } from '@signbridge/types';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const HAND_MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export interface DetectHandsResult {
  rightHand: Landmark3D[] | null;
  leftHand: Landmark3D[] | null;
  rightHandConfidence: number;
  leftHandConfidence: number;
}

export interface DetectMultimodalResult extends DetectHandsResult {
  faceLandmarks: Landmark3D[] | null;
  faceConfidence: number;
}

class MediaPipeVisionService {
  private _handLandmarker: HandLandmarker | null = null;
  private _faceLandmarker: FaceLandmarker | null = null;
  private _initialized = false;
  private _faceInitialized = false;
  private _initPromise: Promise<void> | null = null;

  get isInitialized() {
    return this._initialized;
  }

  get isFaceInitialized() {
    return this._faceInitialized;
  }

  async init(): Promise<void> {
    if (this._initialized) return;
    if (this._initPromise) return this._initPromise;

    this._initPromise = (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

        // 1. Initialize HandLandmarker (Primary)
        try {
          this._handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate: 'GPU' },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          console.log('[MediaPipe] HandLandmarker ready (GPU)');
        } catch {
          console.warn('[MediaPipe] GPU hand failed, trying CPU fallback');
          this._handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate: 'CPU' },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          console.log('[MediaPipe] HandLandmarker ready (CPU fallback)');
        }
        this._initialized = true;

        // 2. Initialize FaceLandmarker (Secondary feature branch)
        try {
          this._faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: FACE_MODEL_PATH, delegate: 'GPU' },
            runningMode: 'VIDEO',
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
          });
          this._faceInitialized = true;
          console.log('[MediaPipe] FaceLandmarker ready (GPU)');
        } catch {
          try {
            this._faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath: FACE_MODEL_PATH, delegate: 'CPU' },
              runningMode: 'VIDEO',
              numFaces: 1,
              minFaceDetectionConfidence: 0.5,
              minFacePresenceConfidence: 0.5,
              minTrackingConfidence: 0.5,
              outputFaceBlendshapes: false,
            });
            this._faceInitialized = true;
            console.log('[MediaPipe] FaceLandmarker ready (CPU fallback)');
          } catch (faceErr) {
            console.warn('[MediaPipe] FaceLandmarker initialization skipped; non-manual features will use neutral priors:', faceErr);
          }
        }
      } catch (err) {
        console.error('[MediaPipe] Critical vision tasks initialization failed:', err);
      }
    })();

    return this._initPromise;
  }

  detectHands(video: HTMLVideoElement, timestampMs: number): DetectHandsResult {
    const empty: DetectHandsResult = {
      rightHand: null,
      leftHand: null,
      rightHandConfidence: 0,
      leftHandConfidence: 0,
    };

    if (!this._handLandmarker || !this._initialized) return empty;

    try {
      const result = this._handLandmarker.detectForVideo(video, timestampMs);
      const out: DetectHandsResult = {
        rightHand: null,
        leftHand: null,
        rightHandConfidence: 0,
        leftHandConfidence: 0,
      };

      for (let i = 0; i < (result.handedness?.length ?? 0); i++) {
        const label = result.handedness[i]?.[0]?.categoryName ?? '';
        const conf = result.handedness[i]?.[0]?.score ?? 0;
        const lms: Landmark3D[] = (result.landmarks[i] ?? []).map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z ?? 0,
        }));
        if (label === 'Right') {
          out.leftHand = lms;
          out.leftHandConfidence = conf;
        } else {
          out.rightHand = lms;
          out.rightHandConfidence = conf;
        }
      }
      return out;
    } catch (err) {
      console.error('[MediaPipe] detectHands error:', err);
      return empty;
    }
  }

  detectMultimodal(video: HTMLVideoElement, timestampMs: number): DetectMultimodalResult {
    const handResult = this.detectHands(video, timestampMs);
    let faceLandmarks: Landmark3D[] | null = null;
    let faceConfidence = 0;

    if (this._faceLandmarker && this._faceInitialized) {
      try {
        const faceRes = this._faceLandmarker.detectForVideo(video, timestampMs);
        if (faceRes.faceLandmarks && faceRes.faceLandmarks.length > 0) {
          faceLandmarks = faceRes.faceLandmarks[0].map((lm) => ({
            x: lm.x,
            y: lm.y,
            z: lm.z ?? 0,
          }));
          faceConfidence = 0.95;
        }
      } catch (err) {
        // Safe degraded fallback to hand-only if face frame fails
        faceLandmarks = null;
      }
    }

    return {
      ...handResult,
      faceLandmarks,
      faceConfidence,
    };
  }
}

export const mediaPipeVision = new MediaPipeVisionService();
export const mediaPipeHands = mediaPipeVision; // Backward-compatible alias
export async function initMediaPipe() {
  return mediaPipeVision.init();
}
