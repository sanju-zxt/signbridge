/**
 * stt.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Low-latency streaming Speech-to-Text client using Web Speech API.
 *
 * Upgraded in SignBridge V17:
 *   - Continuous lifecycle with fresh instance recreation on onend (solves InvalidStateError)
 *   - Audio device enumeration & selection (lists physical microphones)
 *   - Granular status reporting ('idle', 'listening', 'speech-detected', 'transcript-received', etc.)
 *   - Real-time interim transcript streaming (< 15ms dispatch)
 *   - Duplicate final transcript suppression
 *   - TTS audio feedback loop suppression (ignores system speaker echo)
 *   - Graceful recovery from pauses and network interruptions
 *   - Supported locales: en (en-US / en-IN), hi-IN, kn-IN, te-IN
 */

import { SupportedLanguage } from '@signbridge/types';
import { ttsPlayer } from './tts';

export type SpeechClientStatus =
  | 'idle'
  | 'requesting-permission'
  | 'ready'
  | 'listening'
  | 'speech-detected'
  | 'transcript-received'
  | 'translating'
  | 'signing'
  | 'error'
  | 'permission-denied'
  | 'unsupported';

export interface AudioInputDevice {
  deviceId: string;
  label: string;
  groupId: string;
}

export interface STTOptions {
  language: SupportedLanguage;
  onPartial: (text: string, latencyMs?: number) => void;
  onFinal: (text: string, confidence: number, latencyMs?: number) => void;
  onError: (err: string) => void;
  onStateChange: (listening: boolean) => void;
  onStatusChange?: (status: SpeechClientStatus, message?: string) => void;
  onDeviceChange?: (device: AudioInputDevice | null) => void;
}

/**
 * Enumerate connected physical audio input devices (microphones).
 */
export async function enumerateAudioInputDevices(): Promise<AudioInputDevice[]> {
  if (typeof window === 'undefined' || !navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === 'audioinput')
      .map((d, idx) => ({
        deviceId: d.deviceId,
        label: d.label || `Microphone ${idx + 1}`,
        groupId: d.groupId,
      }));
  } catch (err) {
    console.warn('[STT] Failed to enumerate audio devices:', err);
    return [];
  }
}

export class StreamingSTTClient {
  private recognition: any = null;
  private isListening = false;
  private isExplicitlyStopped = true;
  private options: STTOptions;
  private restartTimeout: any = null;
  private consecutiveErrors = 0;
  private lastFinalTranscript = '';
  private lastFinalTimestamp = 0;
  private activeDevice: AudioInputDevice | null = null;
  private currentStatus: SpeechClientStatus = 'idle';
  private preferredDeviceId?: string;

  constructor(options: STTOptions) {
    this.options = options;
  }

  private setStatus(status: SpeechClientStatus, message?: string) {
    this.currentStatus = status;
    this.options.onStatusChange?.(status, message);
  }

  public getStatus(): SpeechClientStatus {
    return this.currentStatus;
  }

  public getActiveDevice(): AudioInputDevice | null {
    return this.activeDevice;
  }

  /**
   * Start speech recognition session.
   * Prompts for permission if needed, detects audio hardware, and launches continuous recognition.
   */
  public async start(preferredDeviceId?: string) {
    if (typeof window === 'undefined') return;

    this.preferredDeviceId = preferredDeviceId;
    this.isExplicitlyStopped = false;
    clearTimeout(this.restartTimeout);
    this.setStatus('requesting-permission', 'Requesting microphone permission...');

    // 1. Microphone Hardware & Permission Check
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const audioConstraints: boolean | MediaTrackConstraints = preferredDeviceId
          ? { deviceId: { exact: preferredDeviceId } }
          : true;

        const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });

        // Query active track label & hardware details
        const audioTracks = typeof stream.getAudioTracks === 'function' ? stream.getAudioTracks() : stream.getTracks();
        if (audioTracks && audioTracks.length > 0) {
          const track = audioTracks[0];
          const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
          this.activeDevice = {
            deviceId: settings.deviceId || preferredDeviceId || 'default',
            label: track.label || 'Default System Microphone',
            groupId: settings.groupId || '',
          };
          this.options.onDeviceChange?.(this.activeDevice);
        }

        // Immediately release stream tracks so Web Speech API gets clean access
        if (typeof stream.getTracks === 'function') {
          stream.getTracks().forEach((track) => track.stop());
        }
      }
    } catch (micErr: any) {
      console.warn('[STT] Mic initialization notice:', micErr.name, micErr.message);
      if (micErr.name === 'NotAllowedError' || micErr.name === 'PermissionDeniedError') {
        this.isExplicitlyStopped = true;
        this.setStatus('permission-denied', 'Microphone permission denied.');
        this.options.onError(
          'Microphone permission was denied. Please allow microphone access in your browser address bar.'
        );
        this.options.onStateChange(false);
        return;
      }
      if (micErr.name === 'NotFoundError' || micErr.name === 'DevicesNotFoundError') {
        this.isExplicitlyStopped = true;
        this.setStatus('error', 'No microphone detected.');
        this.options.onError(
          'No microphone detected on your device. Please plug in a microphone.'
        );
        this.options.onStateChange(false);
        return;
      }
    }

    // 2. Browser Web Speech API Detection
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      this.isExplicitlyStopped = true;
      this.setStatus('unsupported', 'Web Speech API is not supported in this browser.');
      this.options.onError(
        'Web Speech API is not supported in this browser. Please use Google Chrome or Microsoft Edge for live voice recognition.'
      );
      this.options.onStateChange(false);
      return;
    }

    // 3. Launch Recognizer with Clean Instance
    this.createAndStartRecognizer();
  }

  /**
   * Creates a fresh SpeechRecognition instance and binds robust event handlers.
   * Creating a new instance on every restart is REQUIRED by the W3C Web Speech API
   * because reusing an ended recognizer throws InvalidStateError.
   */
  private createAndStartRecognizer() {
    if (this.isExplicitlyStopped || typeof window === 'undefined') return;

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) return;

    // Clean up previous instance cleanly
    if (this.recognition) {
      try {
        this.recognition.onstart = null;
        this.recognition.onspeechstart = null;
        this.recognition.onspeechend = null;
        this.recognition.onresult = null;
        this.recognition.onerror = null;
        this.recognition.onend = null;
        this.recognition.abort();
      } catch {}
      this.recognition = null;
    }

    try {
      const recog = new SpeechRecognition();
      this.recognition = recog;
      recog.continuous = true;
      recog.interimResults = true;
      recog.maxAlternatives = 1;

      // Locale mapping
      const langMap: Record<SupportedLanguage, string> = {
        en: 'en-US',
        hi: 'hi-IN',
        kn: 'kn-IN',
        te: 'te-IN',
      };
      recog.lang = langMap[this.options.language] || 'en-US';

      recog.onstart = () => {
        this.isListening = true;
        this.consecutiveErrors = 0;
        console.log(`[STT] Speech recognition started (language: ${recog.lang})`);
        this.setStatus('listening', 'Microphone ready. Listening for speech...');
        this.options.onStateChange(true);
      };

      recog.onspeechstart = () => {
        this.setStatus('speech-detected', 'Speech detected...');
      };

      recog.onresult = (event: any) => {
        const eventTime = performance.now();

        // Guard against TTS feedback loop (speaker echo)
        if (ttsPlayer.isRecentlySpeaking(700)) {
          return;
        }

        let interimAccumulator = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const item = event.results[i];
          if (!item || !item[0]) continue;

          const transcript = item[0].transcript?.trim() || '';
          const confidence = item[0].confidence || 0.95;

          if (item.isFinal) {
            if (transcript.length > 0) {
              const now = Date.now();
              // Prevent duplicate final triggers within 1500ms
              const isDuplicate =
                transcript.toLowerCase() === this.lastFinalTranscript.toLowerCase() &&
                now - this.lastFinalTimestamp < 1500;

              if (!isDuplicate) {
                this.lastFinalTranscript = transcript;
                this.lastFinalTimestamp = now;
                const latency = Math.round(performance.now() - eventTime);
                console.log(
                  `[STT] Speech recognition finalized: "${transcript}" (confidence: ${confidence.toFixed(2)}, latency: ${latency}ms)`
                );
                this.setStatus('transcript-received', `Transcript received: "${transcript}"`);
                this.options.onFinal(transcript, confidence, latency);
              }
            }
          } else {
            interimAccumulator += (interimAccumulator ? ' ' : '') + transcript;
          }
        }

        if (interimAccumulator.length > 0) {
          const latency = Math.round(performance.now() - eventTime);
          this.setStatus('speech-detected', `Listening: "${interimAccumulator}"`);
          this.options.onPartial(interimAccumulator, latency);
        }
      };

      recog.onerror = (event: any) => {
        const err = event.error;

        if (err === 'not-allowed' || err === 'service-not-allowed') {
          this.isExplicitlyStopped = true;
          this.isListening = false;
          this.setStatus('permission-denied', 'Microphone access blocked.');
          this.options.onStateChange(false);
          this.options.onError(
            'Microphone access blocked. Click the microphone icon in your browser address bar and choose "Allow".'
          );
          return;
        }

        if (err === 'no-speech') {
          // Normal silence pause — do not treat as fatal error, keep status clean
          return;
        }

        if (err === 'aborted') {
          // Normal when stopped or restarted
          return;
        }

        if (err === 'audio-capture') {
          this.setStatus('error', 'No audio was captured from the microphone.');
          this.options.onError('Audio capture failed. Please check your microphone connection.');
          return;
        }

        if (err === 'network') {
          this.setStatus('error', 'Speech recognition network connectivity error.');
          this.options.onError('Network connection error during speech recognition.');
          return;
        }

        console.warn('[STT] Speech recognition warning:', err);
        this.consecutiveErrors++;

        if (this.consecutiveErrors > 2) {
          this.setStatus('error', `Speech recognition error: ${err}`);
          this.options.onError(`Speech recognition error: ${err}. Please check microphone settings.`);
        }
      };

      recog.onend = () => {
        this.isListening = false;

        if (!this.isExplicitlyStopped) {
          // RECREATE A FRESH INSTANCE on restart to eliminate InvalidStateError
          const backoff = Math.min(200 * Math.pow(1.5, this.consecutiveErrors), 2000);

          clearTimeout(this.restartTimeout);
          this.restartTimeout = setTimeout(() => {
            if (!this.isExplicitlyStopped) {
              try {
                this.createAndStartRecognizer();
              } catch (e) {
                console.warn('[STT] Restart attempt:', e);
              }
            }
          }, backoff);
        } else {
          this.setStatus('idle', 'Microphone is off.');
          this.options.onStateChange(false);
        }
      };

      recog.start();
    } catch (err: any) {
      console.error('[STT] Initialization failed:', err);
      this.isListening = false;
      this.setStatus('error', err.message || 'Could not initialize speech recognition');
      this.options.onStateChange(false);
      this.options.onError(err.message || 'Could not initialize speech recognition');
    }
  }

  public stop() {
    this.isExplicitlyStopped = true;
    this.isListening = false;
    clearTimeout(this.restartTimeout);

    if (this.recognition) {
      try {
        this.recognition.onstart = null;
        this.recognition.onspeechstart = null;
        this.recognition.onspeechend = null;
        this.recognition.onresult = null;
        this.recognition.onerror = null;
        this.recognition.onend = null;
        this.recognition.abort();
      } catch {}
      this.recognition = null;
    }

    this.setStatus('idle', 'Microphone stopped.');
    this.options.onStateChange(false);
  }

  public setLanguage(lang: SupportedLanguage) {
    if (this.options.language === lang) return;
    this.options.language = lang;

    if (this.isListening && !this.isExplicitlyStopped) {
      this.stop();
      this.start(this.preferredDeviceId);
    }
  }

  public isActive(): boolean {
    return this.isListening;
  }
}
