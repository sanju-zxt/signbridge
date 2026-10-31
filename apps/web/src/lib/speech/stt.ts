/**
 * stt.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Low-latency streaming Speech-to-Text client using Web Speech API.
 *
 * Features:
 *   - Clean single-instance recognizer lifecycle (no multi-instance leaks)
 *   - Device-lock-free permission probe (releases audio stream immediately)
 *   - Real-time interim transcript streaming (< 15ms dispatch)
 *   - Duplicate final transcript suppression
 *   - TTS audio feedback loop suppression (ignores system speaker echo)
 *   - Exponential backoff on transient errors (no rapid crash-restart loop)
 *   - Supported locales: en-IN, hi-IN, kn-IN, te-IN
 */

import { SupportedLanguage } from '@signbridge/types';
import { ttsPlayer } from './tts';

export interface STTOptions {
  language: SupportedLanguage;
  onPartial: (text: string, latencyMs?: number) => void;
  onFinal: (text: string, confidence: number, latencyMs?: number) => void;
  onError: (err: string) => void;
  onStateChange: (listening: boolean) => void;
}

export class StreamingSTTClient {
  private recognition: any = null;
  private isListening = false;
  private isExplicitlyStopped = false;
  private options: STTOptions;
  private restartTimeout: any = null;
  private consecutiveErrors = 0;
  private lastFinalTranscript = '';
  private lastFinalTimestamp = 0;

  constructor(options: STTOptions) {
    this.options = options;
  }

  public async start() {
    if (typeof window === 'undefined') return;

    this.isExplicitlyStopped = false;
    clearTimeout(this.restartTimeout);

    // 1. Pre-flight permission check without locking the audio hardware
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        // Immediately release tracks so Web Speech API gets exclusive, unconflicted access
        stream.getTracks().forEach((track) => track.stop());
      }
    } catch (micErr: any) {
      console.warn('[STT] Mic pre-flight notice:', micErr.name);
      if (micErr.name === 'NotAllowedError' || micErr.name === 'PermissionDeniedError') {
        this.options.onError(
          'Microphone permission was denied. Please allow microphone access in your browser address bar.'
        );
        this.isExplicitlyStopped = true;
        this.options.onStateChange(false);
        return;
      }
    }

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      this.options.onError(
        'Web Speech API is not supported in this browser. Please use Google Chrome or Microsoft Edge for live voice recognition.'
      );
      this.options.onStateChange(false);
      return;
    }

    // Abort previous instance cleanly if one exists
    if (this.recognition) {
      try {
        this.recognition.onstart = null;
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
        en: 'en-IN',
        hi: 'hi-IN',
        kn: 'kn-IN',
        te: 'te-IN',
      };
      recog.lang = langMap[this.options.language] || 'en-IN';

      recog.onstart = () => {
        this.isListening = true;
        this.consecutiveErrors = 0;
        console.log(`[STT] Speech recognition started (language: ${recog.lang})`);
        this.options.onStateChange(true);
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
                this.options.onFinal(transcript, confidence, latency);
              }
            }
          } else {
            interimAccumulator += (interimAccumulator ? ' ' : '') + transcript;
          }
        }

        if (interimAccumulator.length > 0) {
          const latency = Math.round(performance.now() - eventTime);
          this.options.onPartial(interimAccumulator, latency);
        }
      };

      recog.onerror = (event: any) => {
        const err = event.error;

        if (err === 'not-allowed' || err === 'service-not-allowed') {
          this.isExplicitlyStopped = true;
          this.isListening = false;
          this.options.onStateChange(false);
          this.options.onError(
            'Microphone access blocked. Click the microphone icon in your browser address bar and choose "Allow".'
          );
          return;
        }

        if (err === 'no-speech') {
          // Normal silence pause — do not treat as fatal
          return;
        }

        if (err === 'aborted') {
          // Normal when stopped or restarted
          return;
        }

        console.warn('[STT] Speech recognition warning:', err);
        this.consecutiveErrors++;

        if (this.consecutiveErrors > 3) {
          this.options.onError(`Speech recognition error: ${err}. Please check microphone settings.`);
        }
      };

      recog.onend = () => {
        this.isListening = false;

        if (!this.isExplicitlyStopped) {
          // Calculate backoff: 300ms base, escalating if errors occurred
          const backoff = Math.min(300 * Math.pow(2, this.consecutiveErrors), 2500);

          clearTimeout(this.restartTimeout);
          this.restartTimeout = setTimeout(() => {
            if (!this.isExplicitlyStopped && this.recognition) {
              try {
                this.recognition.start();
              } catch (e) {
                console.warn('[STT] Restart attempt:', e);
              }
            }
          }, backoff);
        } else {
          this.options.onStateChange(false);
        }
      };

      recog.start();
    } catch (err: any) {
      console.error('[STT] Initialization failed:', err);
      this.isListening = false;
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
        this.recognition.onresult = null;
        this.recognition.onerror = null;
        this.recognition.onend = null;
        this.recognition.abort();
      } catch {}
      this.recognition = null;
    }

    this.options.onStateChange(false);
  }

  public setLanguage(lang: SupportedLanguage) {
    if (this.options.language === lang) return;
    this.options.language = lang;

    if (this.isListening && !this.isExplicitlyStopped) {
      this.stop();
      this.start();
    }
  }

  public isActive(): boolean {
    return this.isListening;
  }
}
