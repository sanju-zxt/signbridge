import { SupportedLanguage } from '@signbridge/types';

export class TTSPlayer {
  private synth: SpeechSynthesis | null = null;
  private isSpeakingState: boolean = false;
  private lastSpokenEndTime: number = 0;

  constructor() {
    if (typeof window !== 'undefined') {
      this.synth = window.speechSynthesis;
    }
  }

  public speak(text: string, language: SupportedLanguage = 'en', onEnd?: () => void) {
    if (!this.synth) return;

    // Interrupt any ongoing utterance to keep perceived latency low
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    const langMap: Record<SupportedLanguage, string> = {
      en: 'en-IN',
      hi: 'hi-IN',
      kn: 'kn-IN',
      te: 'te-IN'
    };
    utterance.lang = langMap[language] || 'en-IN';
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      this.isSpeakingState = true;
    };

    utterance.onend = () => {
      this.isSpeakingState = false;
      this.lastSpokenEndTime = Date.now();
      if (onEnd) onEnd();
    };

    utterance.onerror = () => {
      this.isSpeakingState = false;
      this.lastSpokenEndTime = Date.now();
      if (onEnd) onEnd();
    };

    this.synth.speak(utterance);
  }

  public stop() {
    if (this.synth) {
      this.synth.cancel();
      this.isSpeakingState = false;
      this.lastSpokenEndTime = Date.now();
    }
  }

  public isSpeaking(): boolean {
    return this.isSpeakingState;
  }

  /**
   * Returns true if TTS is currently active OR ended within the last windowMs (default: 600ms),
   * preventing audio feedback echo into the speech recognizer.
   */
  public isRecentlySpeaking(windowMs = 600): boolean {
    return this.isSpeakingState || (Date.now() - this.lastSpokenEndTime < windowMs);
  }
}

export const ttsPlayer = new TTSPlayer();
