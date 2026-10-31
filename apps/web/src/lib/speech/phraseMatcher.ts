/**
 * phraseMatcher.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Multi-lingual Speech-to-Sign phrase matcher and English-to-ISL grammar translator.
 *
 * Adheres to Indian Sign Language (ISL) grammatical structure:
 *   - Subject-Object-Verb (SOV) order (e.g. ME WATER WANT, POLICE CALL)
 *   - Topic-Comment & Wh-questions at sentence end (e.g. YOUR NAME WHAT, BATHROOM WHERE)
 *   - Time-Topic-Comment structure
 *   - Word boundaries, longer-phrase priority, and sequential sentence extraction
 *
 * Covers all 10 ISL communication categories:
 *   1. Greetings & Introductions (HELLO, NAME, YOU, I_ME, MY)
 *   2. Politeness & Conversation (PLEASE, THANK_YOU, YES, NO, HOW_ARE_YOU, GOOD, BAD)
 *   3. Medical & Emergency (HOSPITAL, DOCTOR, SICK, PAIN, MEDICINE, AMBULANCE, HELP, EMERGENCY, POLICE, DANGER, STOP)
 *   4. Food & Sustenance (WATER, FOOD, WANT, BUY, HOW_MUCH)
 *   5. Navigation & Movement (WHERE, BATHROOM, GO, COME, HERE, THERE, HOME)
 *   6. Education & Workplace (WORK, LEARN, SCHOOL, UNDERSTAND, DONT_UNDERSTAND)
 *   7. Clarification & Actions (WAIT, REPEAT, AGAIN, SORRY, FINE, GOODBYE)
 */

import {
  translateEnglishToISLGloss,
  translateEnglishToASLGloss,
  ASL_GRAMMAR_RULES,
  parseEnglishToISLSentenceIR,
  ISLSentenceIR,
} from '@signbridge/types';

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
    pattern: /\b(medical\s+emergency|urgent\s+help|emergency\s+situation|call\s+911|aapaatkaal)\b/i,
    priority: 95,
    isEmotional: true,
    category: 'urgent',
  },
  {
    canonicalAction: 'HOW_ARE_YOU',
    pattern: /\b(how\s+are\s+you(\s+doing)?|how('s| is)\s+it\s+going|how\s+do\s+you\s+feel|aap\s+kaise\s+hain|neevu\s+hegiddiri)\b/i,
    priority: 95,
    isEmotional: false,
    category: 'greeting',
  },
  {
    canonicalAction: 'HELLO',
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
    pattern: /\b(drinking\s+water|glass\s+of\s+water|bottle\s+of\s+water|peene\s+ka\s+paani|kudiyuva\s+neeru)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'FOOD',
    pattern: /\b(something\s+to\s+eat|having\s+lunch|having\s+dinner|meal\s+time|khana\s+chahiye)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'AGAIN',
    pattern: /\b(please\s+repeat|repeat\s+that|say\s+again|once\s+more|phir\s+se\s+bolein)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'clarification',
  },
  {
    canonicalAction: 'HOW_MUCH',
    pattern: /\b(how\s+much\s+(is\s+it|does\s+it\s+cost)|what\s+is\s+the\s+price|kitna\s+hua|eshtu\s+bele)\b/i,
    priority: 90,
    isEmotional: false,
    category: 'need',
  },

  // ── Standard Core ASL Words (Priority 70 - 85) ───────────────────────────
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
    pattern: /\b(hello|namaste|namaskara|namaskar|vanakkam|namaskaram|hi|hey)\b/i,
    priority: 80,
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
    canonicalAction: 'AMBULANCE',
    pattern: /\b(ambulance|emergency\s+van|rogi\s+vahan)\b/i,
    priority: 85,
    isEmotional: true,
    category: 'urgent',
  },
  {
    canonicalAction: 'CALL',
    pattern: /\b(call|phone|ring|dial|bulao|kareyeri)\b/i,
    priority: 82,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'DOCTOR',
    pattern: /\b(doctor|physician|dr|vaidya|daktar)\b/i,
    priority: 82,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'MEDICINE',
    pattern: /\b(medicine|medication|pill|tablet|drug|dawai|aushadhi|mandhu)\b/i,
    priority: 82,
    isEmotional: false,
    category: 'need',
  },
  {
    canonicalAction: 'PAIN',
    pattern: /\b(pain|hurt|hurting|ache|injury|injured|dard|novvu|noppi)\b/i,
    priority: 82,
    isEmotional: true,
    category: 'emotion',
  },
  {
    canonicalAction: 'HOSPITAL',
    pattern: /\b(hospital|clinic|emergency\s+room|aspitre|vaidhyashala|maruthuvamanai)\b/i,
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
    priority: 85,
    isEmotional: false,
    category: 'urgent',
  },
  {
    canonicalAction: 'EMERGENCY',
    pattern: /\b(emergency|urgent|aapaatkaal)\b/i,
    priority: 80,
    isEmotional: true,
    category: 'urgent',
  },
  {
    canonicalAction: 'SCHOOL',
    pattern: /\b(school|college|classroom|university|vidyalaya|shale|patashala)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'location',
  },
  {
    canonicalAction: 'WORK',
    pattern: /\b(work|working|job|office|kaam|kelasa|pani)\b/i,
    priority: 80,
    isEmotional: false,
    category: 'location',
  },
  {
    canonicalAction: 'GOOD',
    pattern: /\b(good|great|fine|nice|well|accha|badhiya|chennagide|manchidi)\b/i,
    priority: 78,
    isEmotional: false,
    category: 'politeness',
  },
  {
    canonicalAction: 'BAD',
    pattern: /\b(bad|poor|awful|unwell|bura|chettadi|kettadhu)\b/i,
    priority: 78,
    isEmotional: false,
    category: 'emotion',
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
    canonicalAction: 'WANT',
    pattern: /\b(want|desire|need|require|zaroorat|beku|kavali|thevai)\b/i,
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
    canonicalAction: 'HERE',
    pattern: /\b(here|yahan|illi|ikkada|inge)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'THERE',
    pattern: /\b(there|vahan|alli|akkada|ange)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'direction',
  },
  {
    canonicalAction: 'HAPPY',
    pattern: /\b(happy|glad|joyful|cheerful|khush|santhosha|anandam)\b/i,
    priority: 70,
    isEmotional: true,
    category: 'emotion',
  },
  {
    canonicalAction: 'FINE',
    pattern: /\b(fine|okay|all\s+right|theek|sari)\b/i,
    priority: 70,
    isEmotional: false,
    category: 'politeness',
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
 * Extracts and returns an ordered sequence of ISL actions for 3D avatar animation.
 * Evaluates ISL grammatical structure (SOV / Wh-questions at sentence end) first,
 * followed by ASL fallback, and chronological appearance.
 * E.g.: "Hello, how are you?" -> ["HELLO", "YOU", "HOW"]
 *       "Where is the bathroom?" -> ["BATHROOM", "WHERE"]
 *       "Can you help me?" -> ["YOU", "HELP", "NEED"]
 *       "Please call an ambulance" -> ["PLEASE", "AMBULANCE", "CALL"]
 */
export function matchSpeechSequence(text: string): string[] {
  if (!text || !text.trim()) return [];

  // 1. Full ISL Linguistic Transformation Engine (ISL-IR)
  const islIR = parseEnglishToISLSentenceIR(text);
  if (islIR.glossSequence.length > 0) {
    return islIR.glossSequence;
  }

  // 2. Legacy / rule-based fallback
  const islTranslation = translateEnglishToISLGloss(text);
  if (islTranslation.glosses.length > 0) {
    return islTranslation.glosses;
  }

  // 3. Fallback to chronological appearance of matched concepts
  const matches = matchSpeechPhrases(text);
  const ordered = [...matches].sort((a, b) => a.index - b.index);
  return ordered.map((m) => m.canonicalAction);
}

export function matchSpeechISLIR(text: string): ISLSentenceIR {
  return parseEnglishToISLSentenceIR(text);
}
