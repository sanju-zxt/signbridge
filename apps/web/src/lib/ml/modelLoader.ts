/**
 * modelLoader.ts
 * TF.js model loader with WebGL backend, graceful bootstrap fallback.
 */
import type { ModelMetadata } from '@signbridge2vision/types';

export interface LoadedModel { model: any; metadata: ModelMetadata; }

let _tfPromise: Promise<any> | null = null;

async function getTf(): Promise<any> {
  if (_tfPromise) return _tfPromise;
  _tfPromise = (async () => {
    const [tf] = await Promise.all([
      import('@tensorflow/tfjs'),
      import('@tensorflow/tfjs-backend-webgl'),
    ]);
    await (tf as any).setBackend('webgl').catch(() => (tf as any).setBackend('cpu'));
    console.log('[ModelLoader] TF.js ready, backend:', (tf as any).getBackend());
    return tf;
  })();
  return _tfPromise;
}

export async function loadISLModel(modelUrl = '/models/sbv-gru-v1/model.json'): Promise<LoadedModel> {
  const tf = await getTf();

  let metadata: ModelMetadata = {
    id: 'sbv-gru-v1', version: '1.0.0-bootstrap',
    featureDim: 126, windowSize: 30, numClasses: 17,
    classes: [], isBootstrap: true,
    description: 'Bootstrap model (synthetic data). Collect real data for accuracy.',
  };

  try {
    const res = await fetch('/models/sbv-gru-v1/metadata.json');
    if (res.ok) metadata = await res.json();
  } catch { /* use defaults */ }

  try {
    console.log('[ModelLoader] Loading model from', modelUrl);
    const model = await (tf as any).loadLayersModel(modelUrl);
    console.log('[ModelLoader] Model loaded successfully');
    return { model, metadata };
  } catch (err) {
    console.warn('[ModelLoader] Model load failed, using bootstrap:', err);
    const tf2: any = await getTf();
    const model = tf2.sequential();
    model.add(tf2.layers.dense({ units: 17, activation: 'softmax', inputShape: [126] }));
    return { model, metadata: { ...metadata, isBootstrap: true } };
  }
}

export async function runInference(
  model: any,
  featureMatrix: Float32Array,
  windowSize: number,
  featureDim: number
): Promise<number[]> {
  const tf: any = await getTf();
  const input = tf.tensor3d(Array.from(featureMatrix), [1, windowSize, featureDim], 'float32');
  try {
    const output = model.predict(input);
    const data: Float32Array = await output.data();
    output.dispose();
    return Array.from(data);
  } finally {
    input.dispose();
  }
}