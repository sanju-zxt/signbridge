/**
 * handLandmarker.ts
 * Robust, resilient loader for MediaPipe HandLandmarker.
 * Prioritizes local wasm and local task bundle, with graceful CDN and CPU fallbacks.
 */

export interface HandLandmarkerInstance {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => any;
  close?: () => void;
}

export async function createHandLandmarker(): Promise<HandLandmarkerInstance> {
  const { HandLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');

  // 1. Resolve WASM files (try local public /wasm, fallback to CDN)
  let vision: any = null;
  try {
    vision = await FilesetResolver.forVisionTasks('/wasm');
  } catch (err) {
    console.warn('[HandLandmarker] Local /wasm load failed, trying verified CDN:', err);
    try {
      vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm');
    } catch (cdnErr) {
      console.error('[HandLandmarker] CDN wasm load failed:', cdnErr);
      throw new Error('Failed to resolve MediaPipe Vision WASM binaries.');
    }
  }

  // 2. Load Model Asset (try local model, fallback to Google Storage)
  const modelPaths = [
    '/models/hand_landmarker.task',
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  ];

  const delegates: Array<'GPU' | 'CPU'> = ['GPU', 'CPU'];

  for (const modelPath of modelPaths) {
    for (const delegate of delegates) {
      try {
        const hl = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: modelPath,
            delegate,
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.45,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        console.log(`[HandLandmarker] Successfully initialised (${delegate} mode via ${modelPath})`);
        return hl;
      } catch (err) {
        console.warn(`[HandLandmarker] Initialisation attempt failed (${delegate} on ${modelPath}):`, (err as Error).message);
      }
    }
  }

  throw new Error('All HandLandmarker creation options failed.');
}
