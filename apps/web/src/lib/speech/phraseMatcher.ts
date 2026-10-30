/**
 * phraseMatcher.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Multi-lingual Speech-to-Sign phrase matcher with word boundaries,
 * longer-phrase priority, and sequential sentence extraction.
 *
 * Covers the full 25-concept Indian Sign Language (ISL) vocabulary:
 *   HELLO, THANK_YOU, PLEASE, SORRY, YES, NO, GOOD, HELP, WATER, FOOD,
 *   HOME, SCHOOL, HOSPITAL, NEED, EMERGENCY, POLICE, STOP, GO, WAIT, COME,
 *   UNDERSTAND, DONT_UNDERSTAND, NAME, WHERE, BATHROOM
 * Plus:
 *   CRYING, REPEAT, LEFT, RIGHT, HERE, GOING
 */

export interface MatchedPhrase {
  canonicalAction: string;
  matchedText: string;
  index: number;
  confidence: number;
  priority: number;
  isEmotional: boolean;
  category: 'greeting' | 'politeness' | 'emotion' | 'direction' | 'need' | 'clarification' | 'location' | 'urgent';
}

interface PhraseRule {
  canonicalAction: string;
  pattern: RegExp;
  priority: number; // Higher number = evaluated first
  isEmotional: boolean;
  category: 'greeting' | 'politeness' | 'emotion' | 'direction' | 'need' | 'clarification' | 'location' | 'urgent';
}

// Multi-word and phrase priority rules
const PHRASE_RULES: PhraseRule[] = [
  // ── High Priority Multi-Word Idioms (Priority 95 - 100) ──────────────────
  {
    canonicalAction: 'THANK_YOU',
    pattern: /\b(thank\s+you\s+(so\s+much|very\s+much|a\s+lot)|many\s+thanks|bahut\s+dhanyavaad|chala\s+dhanyavadamulu|tumba\s+dhanyavadagalu)\b/i,
    priority: 100,
    isEmotional: false,
    category: 'politeness',
  },
  {
    canonicalAction: 'DONT_UNDERSTAND',
    pattern: /\b(i\s+do(n't| not)\s+understand|did(n't| not)\s+understand|not\s+understanding|samajh\s+nahi\s+aaya|artha\s+agilla|ardham\s+kaledu)\b/i,
    priority: 100,
    isEmotional: false,
    category: 'clarification',
  },
  {
    canonicalAction: 'UNDERSTAND',
    pattern: /\b(i\s+understand|i\s+got\s+it|makes\s+sense|samajh\s+aaya|samajh\s+gaya|artha\s+ayithu|ardhamayindi)\b/i,
    priority: 95,
    isEmotional: false,
    category: 'clarification',
  },
  {
    canonicalAction: 'SORRY',
    pattern: /\b(i('m| am)?\s+very\s+sorry|i('m| am)?\s+so\s+sorry|i\s+sincerely\s+apologize|please\s+forgive\s+me|bahut\s+maaf\s+karna)\b/i,
    priority: 95,
    isEmotional: true,
    category: 'emotion',
  },
  {
    canonicalAction: 'EMERGENCY',
    pattern: /\b(medical\s+emergency|urgent\s+help|emergency\s+situation|aapaatkaal)\b/i,
    priority: 95,
    isEmotional: true,
    category: 'urgent',
  },
  {
    canonicalAction: 'GOING',
    pattern: /\b(i('m| am)?\s+(going|leaving|heading\s+out|taking\s+off)|see\s+you\s+(later|soon)|good\s+bye|have\s+to\s+go|gotta\s+go)\b/i,
    priority: 95,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'CRYING',
    pattern: /\b(i('m| am)?\s+crying|bursting\s+into\s+tears|feeling\s+like\s+crying|tears\s+in\s+my\s+eyes|crying\s+so\s+hard)\b/i,
    priority: 95,
    isEmotional: true,
    category: 'emotion',
  },
  {
    canonicalAction: 'HI',
    pattern: /\b(good\s+(morning|afternoon|evening|day)|hi\s+there|hey\s+there|nice\s+to\s+meet\s+you)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'greeting',
  },
  {
    canonicalAction: 'HELP',
    pattern: /\b(i\s+need\s+help|please\s+help\s+me|emergency\s+help|help\s+me|need\s+assistance)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'WATER',
    pattern: /\b(drinking\s+water|glass\s+of\s+water|peene\s+ka\s+paani|kudiyuva\s+neeru)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'FOOD',
    pattern: /\b(something\s+to\s+eat|having\s+lunch|having\s+dinner|khana\s+chahiye)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'REPEAT',
    pattern: /\b(please\s+repeat|repeat\s+that|say\s+again|once\s+more|phir\s+se\s+bolein)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'clarification',
  },

  // ── Standard 25-Concept Words & Multilingual Variations (Priority 70 - 85)
  {
    canonicalAction: 'THANK_YOU',
    pattern: /\b(thank\s+you|thanks|dhanyavaad|dhanyavada|shukriya|dhanyavaada|nandri)\b/i,
    priority: 85,
    isEmotional: false,
    category: 'politeness',
  },
  {
    canonicalAction: 'SORRY',
    pattern: /\b(sorry|pardon(\s+me)?|apologies|excuse\s+me|maaf\s+kijiye|kshemisi|mannikavum)\b/i,
    priority: 85,
    isEmotional: true,
    category: 'emotion',
  },
  {
    canonicalAction: 'HELLO',
    pattern: /\b(hello|namaste|namaskara|namaskar|vanakkam|namaskaram)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'greeting',
  },
  {
    canonicalAction: 'HI',
    pattern: /\b(hi|hey)\b/i,
    priority: 78,
    isEmotional: false,
    category: 'greeting',
  },
  {
    canonicalAction: 'PLEASE',
    pattern: /\b(please|kindly|kripya|dayavittu|dayachesi|tayavuceytu)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'politeness',
  },
  {
    canonicalAction: 'HELP',
    pattern: /\b(help|assist|madad|sahaya|sahayam|otavi)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'WATER',
    pattern: /\b(water|paani|neeru|neellu|thanni|thirsty)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'FOOD',
    pattern: /\b(food|eat|eating|meal|hungry|lunch|dinner|khana|oota|bhojana|thindi|saapadu)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'HOSPITAL',
    pattern: /\b(hospital|clinic|doctor|aspitre|vaidhyashala|maruthuvamanai)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'location',
  },
  {
    canonicalAction: 'BATHROOM',
    pattern: /\b(bathroom|washroom|restroom|toilet|shauchalay|lavatory)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'location',
  },
  {
    canonicalAction: 'POLICE',
    pattern: /\b(police|cop|police\s+station|thana|rakshakudu)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'urgent',
  },
  {
    canonicalAction: 'EMERGENCY',
    pattern: /\b(emergency|urgent)\b/i,
    priority: 80,
    isEmotional: true,
    category: 'urgent',
  },
  {
    canonicalAction: 'SCHOOL',
    pattern: /\b(school|college|classroom|vidyalaya|shale|patashala|pallikkutam)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'location',
  },
  {
    canonicalAction: 'HOME',
    pattern: /\b(home|house|residence|ghar|mane|illu|veedu)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'location',
  },
  {
    canonicalAction: 'GOOD',
    pattern: /\b(good|great|fine|nice|well|accha|badhiya|chennagide|manchidi|nalladhu)\b/i,
    priority: 78,
    isEmotional: false,
    category: 'politeness',
  },
  {
    canonicalAction: 'STOP',
    pattern: /\b(stop|halt|pause|cease|ruko|nillu|aagu|nillunga)\b/i,
    priority: 78,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'GO',
    pattern: /\b(go|proceed|move|depart|jao|chalo|hogu|vellu|ponga)\b/i,
    priority: 75,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'WAIT',
    pattern: /\b(wait|hold\s+on|stay|one\s+second|intezar|tadi|kaathiru)\b/i,
    priority: 75,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'COME',
    pattern: /\b(come|approach|aao|idhar\s+aao|baa|vaa|vachindi)\b/i,
    priority: 75,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'NEED',
    pattern: /\b(need|require|want|zaroorat|beku|kavali|thevai)\b/i,
    priority: 75,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'NAME',
    pattern: /\b(name|called|naam|hesaru|peru|peyar)\b/i,
    priority: 75,
    isEmotional: false,
    category: 'clarification',
  },
  {
    canonicalAction: 'WHERE',
    pattern: /\b(where\s+is|where|location|which\s+way|kahan|yelli|ekkada|enge)\b/i,
    priority: 75,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'YES',
    pattern: /\b(yes|yeah|yep|correct|agreed|ha|haan|houdu|avunu|aam)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'clarification',
  },
  {
    canonicalAction: 'NO',
    pattern: /\b(no|nope|nah|not\s+really|incorrect|wrong|nahi|illa|kaadhu|illai)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'clarification',
  },
  {
    canonicalAction: 'CRYING',
    pattern: /\b(crying|cried|shedding\s+tears|weeping|sobbing|rona|alu)\b/i,
    priority: 70,
    isEmotional: true,
    category: 'emotion',
  },
  {
    canonicalAction: 'GOING',
    pattern: /\b(going|leaving|departing|goodbye|bye(\s+bye)?|alvida|ja\s+raha\s+hoon)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'HERE',
    pattern: /\b(here|yahan|illi|ikkada|inge)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'LEFT',
    pattern: /\b(left|turn\s+left|baayein|edagade|edama)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'RIGHT',
    pattern: /\b(right|turn\s+right|daayein|balagade|kudi)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'direction',
  },
];

/**
 * Matches recognized speech text against known conversational and sign phrases.
 * Returns matches sorted by priority (higher priority first).
 */
export function matchSpeechPhrases(text: string): MatchedPhrase[] {
  if (!text || typeof text !== 'string') return [];
  const normalized = text.trim();
  if (normalized.length === 0) return [];

  const matches: MatchedPhrase[] = [];
  const seenActions = new Set<string>();

  // Sort rules by priority descending
  const sortedRules = [...PHRASE_RULES].sort((a, b) => b.priority - a.priority);

  for (const rule of sortedRules) {
    const match = normalized.match(rule.pattern);
    if (match && !seenActions.has(rule.canonicalAction)) {
      seenActions.add(rule.canonicalAction);
      matches.push({
        canonicalAction: rule.canonicalAction,
        matchedText: match[0],
        index: match.index ?? 0,
        confidence: 0.95,
        priority: rule.priority,
        isEmotional: rule.isEmotional,
        category: rule.category,
      });
    }
  }

  return matches;
}

/**
 * Returns the highest-priority single action for a spoken phrase.
 */
export function getPrimaryActionFromSpeech(text: string): string | null {
  const matches = matchSpeechPhrases(text);
  return matches.length > 0 ? matches[0].canonicalAction : null;
}

/**
 * Extracts all matched ISL sign gestures in chronological sentence sequence order.
 * E.g.: "Hello, thank you for your help" -> ["HELLO", "THANK_YOU", "HELP"]
 */
export function matchSpeechSequence(text: string): string[] {
  const matches = matchSpeechPhrases(text);
  // Sort chronologically by their appearance in the spoken utterance
  const ordered = [...matches].sort((a, b) => a.index - b.index);
  return ordered.map((m) => m.canonicalAction);
}
