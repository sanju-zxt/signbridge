import { SupportedLanguage } from '@signbridge2vision/types';

export interface SpeechProcessingResult {
  transcript: string;
  normalizedText: string;
  language: SupportedLanguage;
  confidence: number;
  isFinal: boolean;
}

export class SpeechService {
  normalizeTranscript(rawText: string): string {
    if (!rawText) return '';
    let text = rawText.trim();
    // Capitalize first letter
    text = text.charAt(0).toUpperCase() + text.slice(1);
    // Add period if missing trailing punctuation
    if (!/[.!?]$/.test(text)) {
      text += '.';
    }
    return text;
  }
}

export const speechService = new SpeechService();
