export type Modality = 'isl' | 'speech' | 'text';

export type SpeakerRole = 'isl_signer' | 'speech_speaker' | 'assistant' | 'observer';

export type MessageStatus = 'partial' | 'final' | 'low_confidence';

export type SystemState =
  | 'IDLE'
  | 'LISTENING'
  | 'SIGNING'
  | 'PROCESSING'
  | 'RECOGNIZED'
  | 'LOW_CONFIDENCE'
  | 'FINALIZING'
  | 'THINKING'
  | 'RESPONDING'
  | 'SPEAKING'
  | 'ERROR'
  | 'RECONNECTING';

export type SupportedLanguage = 'en' | 'hi' | 'kn' | 'te';

export interface AlternativeCandidate {
  sign: string;
  confidence: number;
  label: string;
}

export interface CanonicalMessage {
  id: string;
  sessionId: string;
  speakerId: string;
  speakerRole: SpeakerRole;
  speakerName: string;
  modality: Modality;
  text: string;
  normalizedText: string;
  confidence: number;
  alternatives?: AlternativeCandidate[];
  language: SupportedLanguage;
  timestamp: number;
  status: MessageStatus;
  latencyMs?: number;
  translations?: Partial<Record<SupportedLanguage, string>>;
}

export interface Landmark3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface NormalizedFrame {
  timestamp: number;
  leftHand: Landmark3D[] | null;
  rightHand: Landmark3D[] | null;
  pose: Landmark3D[] | null;
  features: number[];
}

export interface ISLSignDefinition {
  id: string;
  sign: string;
  label: string;
  category: 'greetings' | 'needs' | 'navigation' | 'clarification';
  gloss: string;
  canonicalSentence: string;
  translations: Record<SupportedLanguage, string>;
  description: string;
  keyFeatures: {
    hands: 'single' | 'both';
    dominantHand: 'left' | 'right' | 'either';
    primaryMotion: 'linear' | 'circular' | 'static' | 'repetitive';
    location: 'chest' | 'head' | 'neutral_space' | 'chin';
  };
}

export interface RecognitionEvent {
  id: string;
  sessionId: string;
  timestamp: number;
  sign: string;
  label: string;
  confidence: number;
  alternatives: AlternativeCandidate[];
  modelVersion: string;
  latencyMs: number;
  isDebounced: boolean;
  status: 'provisional' | 'final' | 'low_confidence';
}

export interface SpeechTranscriptEvent {
  sessionId: string;
  speakerId: string;
  transcript: string;
  isFinal: boolean;
  confidence: number;
  language: SupportedLanguage;
  timestamp: number;
}

export interface AgentResponseEvent {
  sessionId: string;
  replyToMessageId: string;
  text: string;
  translations: Record<SupportedLanguage, string>;
  suggestedQuickReplies: string[];
  latencyMs: number;
  model: string;
}

export interface LatencyMetricRecord {
  turnId: string;
  visionLatencyMs?: number;
  speechPartialLatencyMs?: number;
  speechFinalLatencyMs?: number;
  agentResponseLatencyMs?: number;
  ttsLatencyMs?: number;
  totalTurnLatencyMs: number;
  timestamp: number;
}

export interface LatencySummary {
  p50: number;
  p95: number;
  mean: number;
  min: number;
  max: number;
  sampleCount: number;
}

export interface ConfusionMatrixEntry {
  actual: string;
  predicted: string;
  count: number;
}

export interface MLClassMetric {
  sign: string;
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface MLBenchmarkMetrics {
  totalSamples: number;
  accuracy: number;
  macroF1: number;
  weightedF1: number;
  p50InferenceLatencyMs: number;
  p95InferenceLatencyMs: number;
  classMetrics: MLClassMetric[];
  confusionMatrix: ConfusionMatrixEntry[];
  evaluatedAt: string;
}

export interface ConversationSession {
  id: string;
  title: string;
  startedAt: number;
  endedAt?: number;
  participants: Array<{
    id: string;
    name: string;
    role: SpeakerRole;
    preferredLanguage: SupportedLanguage;
  }>;
  activeLanguage: SupportedLanguage;
  messages: CanonicalMessage[];
}

export interface DemoScenarioPreset {
  id: string;
  title: string;
  description: string;
  context: string;
  initialRole: SpeakerRole;
  suggestedSignSequence: string[];
  sampleDialogue: Array<{
    speaker: 'isl' | 'speech';
    text: string;
    sign?: string;
    response?: string;
  }>;
}

export * from './vocabulary';

// ──────────────────────────────────────────────────────────────────────────────
// Real-Time ML Recognition Pipeline Types
// ──────────────────────────────────────────────────────────────────────────────

/** Raw landmark frame as received from MediaPipe HandLandmarker */
export interface RawLandmarkFrame {
  timestamp: number;
  /** 21 landmarks for right hand, null if not detected */
  rightHand: Landmark3D[] | null;
  /** 21 landmarks for left hand, null if not detected */
  leftHand: Landmark3D[] | null;
  /** Handedness confidence scores */
  rightHandConfidence: number;
  leftHandConfidence: number;
}

/**
 * Normalized feature vector for one frame, ready for GRU input.
 * Shape: Float32Array of length FEATURE_DIM (126).
 * - [0..62]   right hand 21 landmarks normalized (x,y,z relative to wrist, scaled by palm diagonal)
 * - [63..125] left hand 21 landmarks normalized (zeros if absent)
 */
export interface NormalizedFeatureFrame {
  timestamp: number;
  features: Float32Array;
  /** true if at least one hand was detected in this frame */
  hasHand: boolean;
  rightHandPresent: boolean;
  leftHandPresent: boolean;
}

/** Output from the GRU inference engine for one temporal window */
export interface RecognitionResult {
  /** Sign vocabulary ID e.g. "HELLO", "FOOD" */
  gesture: string;
  /** Human-readable display name */
  displayName: string;
  /** Top-1 confidence score [0..1] */
  confidence: number;
  /** Full probability distribution over all 17 classes */
  probabilities: Array<{ sign: string; confidence: number; label: string }>;
  /** Timestamp of the last frame in the sequence */
  timestamp: number;
  /** How many frames were in the temporal window */
  sequenceLength: number;
  /** Inference time in milliseconds */
  latencyMs: number;
  /** Which model produced this result */
  modelId: string;
}

/** Recognition state machine states */
export type RecognitionState =
  | 'IDLE'          // No hands visible
  | 'DETECTING'     // Hand(s) first appeared, filling buffer
  | 'CLASSIFYING'   // Buffer full, inference running
  | 'RECOGNIZED'    // High-confidence stable prediction
  | 'LOW_CONFIDENCE'// Inference ran but confidence below threshold
  | 'LOST_TRACK'    // Hands were being tracked but disappeared
  | 'ERROR';        // Pipeline error

/** Metadata for a loaded TF.js model */
export interface ModelMetadata {
  id: string;
  version: string;
  featureDim: number;
  windowSize: number;
  numClasses: number;
  classes: string[];
  accuracy?: number;
  f1Score?: number;
  trainedAt?: string;
  isBootstrap: boolean;
  description: string;
}

/** Message sent to inference Web Worker */
export interface InferenceWorkerRequest {
  type: 'INFER' | 'LOAD_MODEL' | 'PING';
  /** Flattened feature matrix [windowSize * featureDim] */
  features?: Float32Array;
  windowSize?: number;
  featureDim?: number;
  /** Model URL to load */
  modelUrl?: string;
}

/** Message received from inference Web Worker */
export interface InferenceWorkerResponse {
  type: 'RESULT' | 'MODEL_LOADED' | 'ERROR' | 'PONG';
  result?: RecognitionResult;
  modelMetadata?: ModelMetadata;
  error?: string;
  latencyMs?: number;
}

/** A recorded training sequence (one sign performance) */
export interface CollectionSequence {
  sign: string;
  sessionId: string;
  sampleIndex: number;
  frames: RawLandmarkFrame[];
  recordedAt: string;
  durationMs: number;
}

/** Real-time pipeline configuration */
export interface PipelineConfig {
  /** Temporal window size in frames (default: 30) */
  windowSize: number;
  /** Minimum frames before inference runs (default: 20) */
  minSequenceLength: number;
  /** Confidence threshold to accept a prediction (default: 0.70) */
  confidenceThreshold: number;
  /** Number of consecutive stable frames required (default: 4) */
  stableFrames: number;
  /** Target camera FPS */
  targetFps: number;
  /** Feature vector dimensionality per frame */
  featureDim: number;
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  windowSize: 30,
  minSequenceLength: 20,
  confidenceThreshold: 0.70,
  stableFrames: 4,
  targetFps: 30,
  featureDim: 126,
};

export * from './vocabulary';

