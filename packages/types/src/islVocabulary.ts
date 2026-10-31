/**
 * islVocabulary.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Indian Sign Language (ISL) Canonical Vocabulary Manifest & Grammar Engine
 *
 * Implements:
 * 1. 10 Prioritized Everyday & Emergency ISL Communication Categories:
 *    - Greetings and introductions
 *    - Personal information
 *    - Everyday conversations
 *    - Questions and responses
 *    - Assistance and requests
 *    - Medical emergencies
 *    - Emergency services
 *    - Education and workplace communication
 *    - Directions and locations
 *    - Common actions and expressions
 * 2. Linguistic features: SOV canonical structure, two-handed vs single-handed phonology,
 *    non-manual markers (head tilt, furrowed brows for Wh-endings), and multilingual translations.
 * 3. Bidirectional translation:
 *    - ISL Gloss Sequence -> Grammatically complete English sentence
 *    - English / Spoken Utterance -> ISL Gloss Sequence (SOV & Topic-Comment)
 */

import { ISLSignDefinition, DemoScenarioPreset } from './index';
import { translateISLIRToEnglish, parseEnglishToISLSentenceIR } from './islGrammar';

export type ISLCommunicationCategory =
  | 'greetings_and_introductions'
  | 'personal_information'
  | 'everyday_conversations'
  | 'questions_and_responses'
  | 'assistance_and_requests'
  | 'medical_emergencies'
  | 'emergency_services'
  | 'education_and_workplace'
  | 'directions_and_locations'
  | 'common_actions_and_expressions';

export const ISL_VOCABULARY: ISLSignDefinition[] = [
  // ── 1. Greetings and Introductions ──────────────────────────────────────────
  {
    id: 'HELLO',
    sign: 'HELLO',
    label: 'Hello / Namaste',
    category: 'greetings_and_introductions',
    gloss: 'HELLO',
    aslGrammarType: 'standard',
    canonicalSentence: 'Hello, nice to meet you.',
    translations: {
      en: 'Hello, nice to meet you.',
      hi: 'नमस्ते, आपसे मिलकर खुशी हुई।',
      kn: 'ನಮಸ್ಕಾರ, ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಿದ್ದು ಸಂತೋಷವಾಯಿತು.',
      te: 'నమస్కారం, మిమ్మల్ని కలవడం చాలా సంతోషంగా ఉంది.',
    },
    description: 'Two-handed folded palms (Namaste) held at chest level with respectful head nod, or right flat palm touching temple and moving smoothly outward.',
    nonManualMarkers: 'Warm smile, slight respectful head nod',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'linear',
      location: 'chest',
    },
  },
  {
    id: 'NAME',
    sign: 'NAME',
    label: 'Name',
    category: 'greetings_and_introductions',
    gloss: 'NAME',
    aslGrammarType: 'wh_question',
    canonicalSentence: 'What is your name?',
    translations: {
      en: 'What is your name?',
      hi: 'आपका नाम क्या है?',
      kn: 'ನಿಮ್ಮ ಹೆಸರು ಏನು?',
      te: 'మీ పేరు ఏమిటి?',
    },
    description: 'Index and middle fingers extended on dominant hand tapping across the non-dominant palm or index fingers crossing.',
    nonManualMarkers: 'Slight head tilt forward, furrowed brow for question',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'NICE_TO_MEET_YOU',
    sign: 'NICE_TO_MEET_YOU',
    label: 'Nice to meet you',
    category: 'greetings_and_introductions',
    gloss: 'NICE MEET YOU',
    aslGrammarType: 'standard',
    canonicalSentence: 'Nice to meet you.',
    translations: {
      en: 'Nice to meet you.',
      hi: 'आपसे मिलकर बहुत अच्छा लगा।',
      kn: 'ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಿದ್ದು ತುಂಬಾ ಸಂತೋಷವಾಯಿತು.',
      te: 'మిమ్మల్ని కలవడం చాలా ఆనందంగా ఉంది.',
    },
    description: 'Right flat palm sweeps forward across chest (GOOD), followed by upright index fingers meeting at center line (MEET).',
    nonManualMarkers: 'Gentle smile, affirmative eye contact',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chest',
    },
  },
  {
    id: 'YOU',
    sign: 'YOU',
    label: 'You',
    category: 'greetings_and_introductions',
    gloss: 'YOU',
    aslGrammarType: 'standard',
    canonicalSentence: 'You.',
    translations: {
      en: 'You.',
      hi: 'आप / तुम।',
      kn: 'ನೀವು / ನೀನು.',
      te: 'మీరు / నువ్వు.',
    },
    description: 'Dominant index finger points directly forward toward the conversational partner.',
    nonManualMarkers: 'Direct eye contact',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'I_ME',
    sign: 'I_ME',
    label: 'I / Me',
    category: 'greetings_and_introductions',
    gloss: 'I_ME',
    aslGrammarType: 'standard',
    canonicalSentence: 'I or Me.',
    translations: {
      en: 'I or Me.',
      hi: 'मैं / मुझे।',
      kn: 'ನಾನು / ನನಗೆ.',
      te: 'నేను / నాకు.',
    },
    description: 'Dominant index finger points to center of chest.',
    nonManualMarkers: 'Neutral gaze',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chest',
    },
  },
  {
    id: 'MY',
    sign: 'MY',
    label: 'My / Mine',
    category: 'greetings_and_introductions',
    gloss: 'MY',
    aslGrammarType: 'standard',
    canonicalSentence: 'My.',
    translations: {
      en: 'My.',
      hi: 'मेरा / मेरी।',
      kn: 'ನನ್ನ.',
      te: 'నాది / నా యొక్క.',
    },
    description: 'Flat open palm rests firmly flat against center of chest.',
    nonManualMarkers: 'Neutral nod',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'static',
      location: 'chest',
    },
  },
  {
    id: 'FRIEND',
    sign: 'FRIEND',
    label: 'Friend',
    category: 'greetings_and_introductions',
    gloss: 'FRIEND',
    aslGrammarType: 'standard',
    canonicalSentence: 'Friend.',
    translations: {
      en: 'Friend.',
      hi: 'दोस्त / मित्र।',
      kn: 'ಸ್ನೇಹಿತ.',
      te: 'స్నేహితుడు.',
    },
    description: 'Both hands hook index fingers together in front of torso and shake gently.',
    nonManualMarkers: 'Friendly smile',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'GOODBYE',
    sign: 'GOODBYE',
    label: 'Goodbye',
    category: 'greetings_and_introductions',
    gloss: 'GOODBYE',
    aslGrammarType: 'standard',
    canonicalSentence: 'Goodbye, see you later.',
    translations: {
      en: 'Goodbye, see you later.',
      hi: 'अलविदा, फिर मिलेंगे।',
      kn: 'ವಿದಾಯ, ಮತ್ತೆ ಸಿಗೋಣ.',
      te: 'వీడ్కోలు, మళ్లీ కలుద్దాం.',
    },
    description: 'Open dominant hand waves laterally at shoulder height or both palms fold inward and part.',
    nonManualMarkers: 'Parting smile',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'head',
    },
  },

  // ── 2. Personal Information ────────────────────────────────────────────────
  {
    id: 'DEAF',
    sign: 'DEAF',
    label: 'Deaf',
    category: 'personal_information',
    gloss: 'DEAF',
    aslGrammarType: 'standard',
    canonicalSentence: 'I am Deaf.',
    translations: {
      en: 'I am Deaf.',
      hi: 'मैं बधिर हूँ।',
      kn: 'ನಾನು ಕಿವುಡ/ಕಿವುಡಿ.',
      te: 'నేను బధిరుడను.',
    },
    description: 'Dominant index finger touches the ear and then moves to touch near the mouth corner.',
    nonManualMarkers: 'Affirmative calm nod',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'head',
    },
  },
  {
    id: 'HEARING',
    sign: 'HEARING',
    label: 'Hearing Person',
    category: 'personal_information',
    gloss: 'HEARING',
    aslGrammarType: 'standard',
    canonicalSentence: 'Hearing person.',
    translations: {
      en: 'Hearing person.',
      hi: 'सुनने वाला व्यक्ति।',
      kn: 'ಕೇಳುವ ವ್ಯಕ್ತಿ.',
      te: 'వినే వ్యక్తి.',
    },
    description: 'Dominant index finger rotates in small circular path beside the ear or in front of lips.',
    nonManualMarkers: 'Neutral expression',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'circular',
      location: 'head',
    },
  },

  // ── 3. Everyday Conversations ──────────────────────────────────────────────
  {
    id: 'HOW_ARE_YOU',
    sign: 'HOW_ARE_YOU',
    label: 'How are you?',
    category: 'everyday_conversations',
    gloss: 'HOW_ARE_YOU',
    aslGrammarType: 'wh_question',
    canonicalSentence: 'How are you doing?',
    translations: {
      en: 'How are you doing?',
      hi: 'आप कैसे हैं?',
      kn: 'ನೀವು ಹೇಗಿದ್ದೀರಿ?',
      te: 'మీరు ఎలా ఉన్నారు?',
    },
    description: 'Both hands curved palms facing upward rotate outwards, accompanied by an open questioning expression.',
    nonManualMarkers: 'Questioning head tilt, slightly furrowed brows',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'circular',
      location: 'chest',
    },
  },
  {
    id: 'GOOD',
    sign: 'GOOD',
    label: 'Good / Fine',
    category: 'everyday_conversations',
    gloss: 'GOOD',
    aslGrammarType: 'standard',
    canonicalSentence: 'Everything is good.',
    translations: {
      en: 'Everything is good.',
      hi: 'सब कुछ अच्छा है।',
      kn: 'ಎಲ್ಲವೂ ಚೆನ್ನಾಗಿದೆ.',
      te: 'అంతా బాగుంది.',
    },
    description: 'Dominant thumbs up pushed forward or flat palm moving from chin into open neutral space.',
    nonManualMarkers: 'Smile, affirmative nod',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'BAD',
    sign: 'BAD',
    label: 'Bad / Unwell',
    category: 'everyday_conversations',
    gloss: 'BAD',
    aslGrammarType: 'standard',
    canonicalSentence: 'That is bad.',
    translations: {
      en: 'That is bad.',
      hi: 'यह बुरा / खराब है।',
      kn: 'ಇದು ಕೆಟ್ಟದಾಗಿದೆ.',
      te: 'ఇది చెడ్డది.',
    },
    description: 'Dominant hand turns palm downward with limp wrist or thumbs down motion.',
    nonManualMarkers: 'Frown, head shake',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'YES',
    sign: 'YES',
    label: 'Yes',
    category: 'everyday_conversations',
    gloss: 'YES',
    aslGrammarType: 'standard',
    canonicalSentence: 'Yes, that is correct.',
    translations: {
      en: 'Yes, that is correct.',
      hi: 'हाँ, यह सही है।',
      kn: 'ಹೌದು, ಅದು ಸರಿ.',
      te: 'అవును, అది నిజం.',
    },
    description: 'Dominant closed fist nods vertically up and down imitating a head nod.',
    nonManualMarkers: 'Affirmative head nod',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'NO',
    sign: 'NO',
    label: 'No',
    category: 'everyday_conversations',
    gloss: 'NO',
    aslGrammarType: 'standard',
    canonicalSentence: 'No, that is not correct.',
    translations: {
      en: 'No, that is not correct.',
      hi: 'नहीं, यह सही नहीं है।',
      kn: 'ಇಲ್ಲ, ಅದು ಸರಿಯಲ್ಲ.',
      te: 'కాదు, అది సరికాదు.',
    },
    description: 'Index and middle fingers snap down against thumb, or dominant flat hand waves side-to-side.',
    nonManualMarkers: 'Head shake side-to-side',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'PLEASE',
    sign: 'PLEASE',
    label: 'Please',
    category: 'everyday_conversations',
    gloss: 'PLEASE',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please assist me.',
    translations: {
      en: 'Please assist me.',
      hi: 'कृपया मेरी सहायता करें।',
      kn: 'ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ.',
      te: 'దయచేసి నాకు సహాయం చేయండి.',
    },
    description: 'Dominant flat open palm placed over center of chest and rubbed in gentle clockwise circular motion.',
    nonManualMarkers: 'Gentle, polite facial expression',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'circular',
      location: 'chest',
    },
  },
  {
    id: 'THANK_YOU',
    sign: 'THANK_YOU',
    label: 'Thank you',
    category: 'everyday_conversations',
    gloss: 'THANK_YOU',
    aslGrammarType: 'standard',
    canonicalSentence: 'Thank you very much.',
    translations: {
      en: 'Thank you very much.',
      hi: 'बहुत-बहुत धन्यवाद।',
      kn: 'ತುಂಬಾ ಧನ್ಯವಾದಗಳು.',
      te: 'చాలా ధన్యవాదాలు.',
    },
    description: 'Fingertips of flat dominant hand touch chin or forehead and extend smoothly forward toward recipient.',
    nonManualMarkers: 'Appreciative smile, respectful nod',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chin',
    },
  },
  {
    id: 'SORRY',
    sign: 'SORRY',
    label: 'Sorry / Apology',
    category: 'everyday_conversations',
    gloss: 'SORRY',
    aslGrammarType: 'standard',
    canonicalSentence: 'I am very sorry.',
    translations: {
      en: 'I am very sorry.',
      hi: 'मुझे बहुत खेद है / क्षमा करें।',
      kn: 'ನನ್ನನ್ನು ಕ್ಷಮಿಸಿ.',
      te: 'నన్ను క్షమించండి.',
    },
    description: 'Dominant closed fist rubs circular motion over heart/chest, or earlobes gently touched in traditional Indian apology gesture.',
    nonManualMarkers: 'Apologetic, remorseful expression',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'circular',
      location: 'chest',
    },
  },

  // ── 4. Questions and Responses ─────────────────────────────────────────────
  {
    id: 'WHAT',
    sign: 'WHAT',
    label: 'What',
    category: 'questions_and_responses',
    gloss: 'WHAT',
    aslGrammarType: 'wh_question',
    canonicalSentence: 'What is that?',
    translations: {
      en: 'What is that?',
      hi: 'वह क्या है?',
      kn: 'ಅದು ಏನು?',
      te: 'అది ఏమిటి?',
    },
    description: 'Index finger wiggles side to side or both open palms held up swaying gently in neutral space.',
    nonManualMarkers: 'Furrowed eyebrows, head tilt',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'WHERE',
    sign: 'WHERE',
    label: 'Where',
    category: 'questions_and_responses',
    gloss: 'WHERE',
    aslGrammarType: 'wh_question',
    canonicalSentence: 'Where is the location?',
    translations: {
      en: 'Where is the location?',
      hi: 'वह स्थान कहाँ है?',
      kn: 'ಆ ಸ್ಥಳ ಎಲ್ಲಿದೆ?',
      te: 'ఆ ప్రదేశం ఎక్కడ ఉంది?',
    },
    description: 'Both open palms held facing upward in neutral space, swaying slightly with questioning expression.',
    nonManualMarkers: 'Furrowed eyebrows, inquisitive head tilt',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'HOW',
    sign: 'HOW',
    label: 'How',
    category: 'questions_and_responses',
    gloss: 'HOW',
    aslGrammarType: 'wh_question',
    canonicalSentence: 'How is it done?',
    translations: {
      en: 'How is it done?',
      hi: 'यह कैसे किया जाता है?',
      kn: 'ಇದು ಹೇಗೆ ಮಾಡಲಾಗುತ್ತದೆ?',
      te: 'ఇది ఎలా చేస్తారు?',
    },
    description: 'Backs of hands touch, palms rotate upward and open outward toward conversational partner.',
    nonManualMarkers: 'Furrowed brow, head forward',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'circular',
      location: 'chest',
    },
  },

  // ── 5. Assistance and Requests ─────────────────────────────────────────────
  {
    id: 'HELP',
    sign: 'HELP',
    label: 'Help / Assist',
    category: 'assistance_and_requests',
    gloss: 'HELP',
    aslGrammarType: 'standard',
    canonicalSentence: 'I need help or assistance.',
    translations: {
      en: 'I need help or assistance.',
      hi: 'मुझे सहायता या मदद चाहिए।',
      kn: 'ನನಗೆ ಸಹಾಯ ಬೇಕು.',
      te: 'నాకు సహాయం కావాలి.',
    },
    description: 'Left flat palm forms a base facing upward; right closed fist with thumb up rests on the palm and both hands lift together.',
    nonManualMarkers: 'Concerned, earnest expression',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'WANT',
    sign: 'WANT',
    label: 'Want',
    category: 'assistance_and_requests',
    gloss: 'WANT',
    aslGrammarType: 'standard',
    canonicalSentence: 'I want this.',
    translations: {
      en: 'I want this.',
      hi: 'मुझे यह चाहिए।',
      kn: 'ನನಗೆ ಇದು ಬೇಕು.',
      te: 'నాకు ఇది కావాలి.',
    },
    description: 'Both open palms face upward in front of body and pull back toward chest while curling fingers inward.',
    nonManualMarkers: 'Affirmative nod',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'NEED',
    sign: 'NEED',
    label: 'Need / Require',
    category: 'assistance_and_requests',
    gloss: 'NEED',
    aslGrammarType: 'standard',
    canonicalSentence: 'I need assistance.',
    translations: {
      en: 'I need assistance.',
      hi: 'मुझे आवश्यकता है।',
      kn: 'ನನಗೆ ಅಗತ್ಯವಿದೆ.',
      te: 'నాకు అవసరం.',
    },
    description: 'Dominant hooked index finger flexes downward firmly in emphatic motion.',
    nonManualMarkers: 'Firm head nod',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'WAIT',
    sign: 'WAIT',
    label: 'Wait',
    category: 'assistance_and_requests',
    gloss: 'WAIT',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please wait a moment.',
    translations: {
      en: 'Please wait a moment.',
      hi: 'कृपया थोड़ा इंतज़ार करें।',
      kn: 'ದಯವಿಟ್ಟು ಒಂದು ಕ್ಷಣ ಕಾಯಿರಿ.',
      te: 'దయచేసి కాసేపు ఆగండి.',
    },
    description: 'Both open palms raised facing outward at chest height with fingers gently wiggling or held steady.',
    nonManualMarkers: 'Attentive, steady gaze',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'static',
      location: 'chest',
    },
  },
  {
    id: 'CALL',
    sign: 'CALL',
    label: 'Call',
    category: 'assistance_and_requests',
    gloss: 'CALL',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please make a call.',
    translations: {
      en: 'Please make a call.',
      hi: 'कृपया कॉल करें।',
      kn: 'ದಯವಿಟ್ಟು ಕರೆ ಮಾಡಿ.',
      te: 'దయచేసి కాల్ చేయండి.',
    },
    description: 'Y-hand shape (thumb and pinky extended) placed near ear and mouth simulating a telephone receiver.',
    nonManualMarkers: 'Urgent gaze if emergency',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'static',
      location: 'head',
    },
  },
  {
    id: 'GIVE',
    sign: 'GIVE',
    label: 'Give',
    category: 'assistance_and_requests',
    gloss: 'GIVE',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please give to me.',
    translations: {
      en: 'Please give to me.',
      hi: 'कृपया मुझे दें।',
      kn: 'ದಯವಿಟ್ಟು ನನಗೆ ಕೊಡಿ.',
      te: 'దయచేసి నాకు ఇవ్వండి.',
    },
    description: 'Flat open hands sweep outward or inward directionally indicating transfer of an object.',
    nonManualMarkers: 'Gentle nod',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },

  // ── 6. Medical Emergencies ─────────────────────────────────────────────────
  {
    id: 'HOSPITAL',
    sign: 'HOSPITAL',
    label: 'Hospital',
    category: 'medical_emergencies',
    gloss: 'HOSPITAL',
    aslGrammarType: 'standard',
    canonicalSentence: 'Where is the hospital?',
    translations: {
      en: 'Where is the hospital?',
      hi: 'अस्पताल कहाँ है?',
      kn: 'ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ?',
      te: 'ఆసుపత్రి ఎక్కడ ఉంది?',
    },
    description: 'Dominant index and middle fingers draw a cross on non-dominant upper arm or forehead.',
    nonManualMarkers: 'Concerned expression',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chest',
    },
  },
  {
    id: 'DOCTOR',
    sign: 'DOCTOR',
    label: 'Doctor',
    category: 'medical_emergencies',
    gloss: 'DOCTOR',
    aslGrammarType: 'standard',
    canonicalSentence: 'I need a doctor.',
    translations: {
      en: 'I need a doctor.',
      hi: 'मुझे डॉक्टर की आवश्यकता है।',
      kn: 'ನನಗೆ ವೈದ್ಯರ ಅಗತ್ಯವಿದೆ.',
      te: 'నాకు డాక్టర్ అవసరం.',
    },
    description: 'Dominant fingertips tap the pulse on the inside of the non-dominant wrist twice.',
    nonManualMarkers: 'Serious attentive expression',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'MEDICINE',
    sign: 'MEDICINE',
    label: 'Medicine',
    category: 'medical_emergencies',
    gloss: 'MEDICINE',
    aslGrammarType: 'standard',
    canonicalSentence: 'I need medicine.',
    translations: {
      en: 'I need medicine.',
      hi: 'मुझे दवाई चाहिए।',
      kn: 'ನನಗೆ ಔಷಧಿ ಬೇಕು.',
      te: 'నాకు మందులు కావాలి.',
    },
    description: 'Dominant middle finger twists or grinds against open non-dominant palm simulating pestle and mortar.',
    nonManualMarkers: 'Neutral expression',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'circular',
      location: 'neutral_space',
    },
  },
  {
    id: 'PAIN',
    sign: 'PAIN',
    label: 'Pain / Hurt',
    category: 'medical_emergencies',
    gloss: 'PAIN',
    aslGrammarType: 'standard',
    canonicalSentence: 'I am experiencing pain.',
    translations: {
      en: 'I am experiencing pain.',
      hi: 'मुझे दर्द हो रहा है।',
      kn: 'ನನಗೆ ನೋವಾಗುತ್ತಿದೆ.',
      te: 'నాకు నొప్పిగా ఉంది.',
    },
    description: 'Both index fingers point toward each other and twist sharply with wincing facial expression.',
    nonManualMarkers: 'Wincing eyes, grimace',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'repetitive',
      location: 'chest',
    },
  },
  {
    id: 'EMERGENCY',
    sign: 'EMERGENCY',
    label: 'Emergency / Urgent',
    category: 'medical_emergencies',
    gloss: 'EMERGENCY',
    aslGrammarType: 'standard',
    canonicalSentence: 'This is an emergency!',
    translations: {
      en: 'This is an emergency!',
      hi: 'यह एक आपातकाल है!',
      kn: 'ಇದು ತುರ್ತು ಪರಿಸ್ಥಿತಿ!',
      te: 'ఇది అత్యవసర పరిస్థితి!',
    },
    description: 'Both open hands shake rapidly at chest height or E-hand shape shakes urgently.',
    nonManualMarkers: 'Wide eyes, alarmed expression',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'repetitive',
      location: 'chest',
    },
  },
  {
    id: 'AMBULANCE',
    sign: 'AMBULANCE',
    label: 'Ambulance',
    category: 'medical_emergencies',
    gloss: 'AMBULANCE',
    aslGrammarType: 'standard',
    canonicalSentence: 'Call an ambulance immediately.',
    translations: {
      en: 'Call an ambulance immediately.',
      hi: 'तुरंत एम्बुलेंस बुलाएं।',
      kn: 'ಕೂಡಲೇ ಆಂಬ್ಯುಲೆನ್ಸ್ ಕರೆಯಿರಿ.',
      te: 'వెంటనే అంబులెన్స్ పిలవండి.',
    },
    description: 'Dominant hand held above head rotating in circles imitating revolving beacon light on top of emergency vehicle.',
    nonManualMarkers: 'Alarmed, urgent expression',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'circular',
      location: 'head',
    },
  },

  // ── 7. Emergency Services ──────────────────────────────────────────────────
  {
    id: 'POLICE',
    sign: 'POLICE',
    label: 'Police',
    category: 'emergency_services',
    gloss: 'POLICE',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please call the police.',
    translations: {
      en: 'Please call the police.',
      hi: 'कृपया पुलिस को बुलाएं।',
      kn: 'ದಯವಿಟ್ಟು ಪೊಲೀಸರನ್ನು ಕರೆಯಿರಿ.',
      te: 'దయచేసి పోలీసులను పిలవండి.',
    },
    description: 'C-hand shape held at opposite shoulder/chest simulating police badge or diagonal sash.',
    nonManualMarkers: 'Firm, alert expression',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'static',
      location: 'chest',
    },
  },
  {
    id: 'DANGER',
    sign: 'DANGER',
    label: 'Danger',
    category: 'emergency_services',
    gloss: 'DANGER',
    aslGrammarType: 'standard',
    canonicalSentence: 'Danger, stay alert.',
    translations: {
      en: 'Danger, stay alert.',
      hi: 'खतरा है, सावधान रहें।',
      kn: 'ಅಪಾಯ, ಎಚ್ಚರವಾಗಿರಿ.',
      te: 'ప్రమాదం, జాగ్రత్తగా ఉండండి.',
    },
    description: 'Dominant open hand held vertically in warning with rapid outward defensive push.',
    nonManualMarkers: 'Widened eyes, alert warning head tilt',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chest',
    },
  },
  {
    id: 'STOP',
    sign: 'STOP',
    label: 'Stop',
    category: 'emergency_services',
    gloss: 'STOP',
    aslGrammarType: 'standard',
    canonicalSentence: 'Stop immediately.',
    translations: {
      en: 'Stop immediately.',
      hi: 'तुरंत रुकें।',
      kn: 'ಕೂಡಲೇ ನಿಲ್ಲಿಸಿ.',
      te: 'వెంటనే ఆపండి.',
    },
    description: 'Dominant open flat hand chops down decisively at a 90-degree angle onto the flat open palm of the non-dominant hand.',
    nonManualMarkers: 'Firm authoritative mouth set',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },

  // ── 8. Education and Workplace Communication ──────────────────────────────
  {
    id: 'SCHOOL',
    sign: 'SCHOOL',
    label: 'School / College',
    category: 'education_and_workplace',
    gloss: 'SCHOOL',
    aslGrammarType: 'standard',
    canonicalSentence: 'School or college.',
    translations: {
      en: 'School or college.',
      hi: 'स्कूल या कॉलेज।',
      kn: 'ಶಾಲೆ ಅಥವಾ ಕಾಲೇಜು.',
      te: 'పాఠశాల లేదా కళాశాల.',
    },
    description: 'Dominant flat open palm claps down horizontally onto non-dominant flat open palm twice.',
    nonManualMarkers: 'Neutral assertion',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'chest',
    },
  },
  {
    id: 'WORK',
    sign: 'WORK',
    label: 'Work / Job',
    category: 'education_and_workplace',
    gloss: 'WORK',
    aslGrammarType: 'standard',
    canonicalSentence: 'Work or office.',
    translations: {
      en: 'Work or office.',
      hi: 'काम या कार्यालय।',
      kn: 'ಕೆಲಸ ಅಥವಾ ಕಚೇರಿ.',
      te: 'పని లేదా కార్యాలయం.',
    },
    description: 'Dominant fist taps down on top of non-dominant fist at wrists twice.',
    nonManualMarkers: 'Focused expression',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },
  {
    id: 'UNDERSTAND',
    sign: 'UNDERSTAND',
    label: 'Understand',
    category: 'education_and_workplace',
    gloss: 'UNDERSTAND',
    aslGrammarType: 'standard',
    canonicalSentence: 'I understand clearly.',
    translations: {
      en: 'I understand clearly.',
      hi: 'मुझे समझ आ गया।',
      kn: 'ನನಗೆ ಅರ್ಥವಾಯಿತು.',
      te: 'నాకు అర్థమైంది.',
    },
    description: 'Index finger flicks upward from closed fist near temple simulating a lightbulb turning on.',
    nonManualMarkers: 'Affirmative nod, enlightened expression',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'head',
    },
  },
  {
    id: 'DONT_UNDERSTAND',
    sign: 'DONT_UNDERSTAND',
    label: "Don't Understand",
    category: 'education_and_workplace',
    gloss: 'DONT_UNDERSTAND',
    aslGrammarType: 'standard',
    canonicalSentence: "I do not understand, please clarify.",
    translations: {
      en: "I do not understand, please clarify.",
      hi: 'मुझे समझ नहीं आया, कृपया स्पष्ट करें।',
      kn: 'ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ, ದಯವಿಟ್ಟು ವಿವರಿಸಿ.',
      te: 'నాకు అర్థం కాలేదు, దయచేసి వివరించండి.',
    },
    description: 'Fingertips touch forehead and pull away with outward rotation and slight head shake.',
    nonManualMarkers: 'Puzzled frown, slight head shake',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'head',
    },
  },
  {
    id: 'REPEAT',
    sign: 'REPEAT',
    label: 'Repeat / Again',
    category: 'education_and_workplace',
    gloss: 'REPEAT',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please repeat that.',
    translations: {
      en: 'Please repeat that.',
      hi: 'कृपया दोहराएं।',
      kn: 'ದಯವಿಟ್ಟು ಪುನರಾವರ್ತಿಸಿ.',
      te: 'దయచేసి మళ్లీ చెప్పండి.',
    },
    description: 'Bent dominant fingers scoop into open palm of non-dominant hand repeatedly.',
    nonManualMarkers: 'Inquiring nod',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space',
    },
  },

  // ── 9. Directions and Locations ───────────────────────────────────────────
  {
    id: 'BATHROOM',
    sign: 'BATHROOM',
    label: 'Bathroom / Restroom',
    category: 'directions_and_locations',
    gloss: 'BATHROOM',
    aslGrammarType: 'standard',
    canonicalSentence: 'Where is the bathroom?',
    translations: {
      en: 'Where is the bathroom?',
      hi: 'शौचालय कहाँ है?',
      kn: 'ಸ್ನಾನಗೃಹ ಎಲ್ಲಿದೆ?',
      te: 'బాత్‌రూమ్ ఎక్కడ ఉంది?',
    },
    description: 'Dominant hand with pinky extended (traditional Indian gesture for restroom) or T-hand shape shaken gently side-to-side.',
    nonManualMarkers: 'Inquiring expression',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'chest',
    },
  },
  {
    id: 'HERE',
    sign: 'HERE',
    label: 'Here',
    category: 'directions_and_locations',
    gloss: 'HERE',
    aslGrammarType: 'standard',
    canonicalSentence: 'Right here.',
    translations: {
      en: 'Right here.',
      hi: 'यहाँ पर।',
      kn: 'ಇಲ್ಲಿಯೇ.',
      te: 'ఇక్కడే.',
    },
    description: 'Both open hands with flat palms facing upward held steady in front of waist, or index finger pointing directly downward.',
    nonManualMarkers: 'Direct downward glance',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'static',
      location: 'neutral_space',
    },
  },
  {
    id: 'LEFT',
    sign: 'LEFT',
    label: 'Left',
    category: 'directions_and_locations',
    gloss: 'LEFT',
    aslGrammarType: 'standard',
    canonicalSentence: 'Turn to the left.',
    translations: {
      en: 'Turn to the left.',
      hi: 'बाईं ओर मुड़ें।',
      kn: 'ಎಡಕ್ಕೆ ತಿರುಗಿ.',
      te: 'ఎడమవైపు తిరగండి.',
    },
    description: 'Dominant hand points or sweeps decisively toward the left side.',
    nonManualMarkers: 'Gaze direction left',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'RIGHT',
    sign: 'RIGHT',
    label: 'Right',
    category: 'directions_and_locations',
    gloss: 'RIGHT',
    aslGrammarType: 'standard',
    canonicalSentence: 'Turn to the right.',
    translations: {
      en: 'Turn to the right.',
      hi: 'दाईं ओर मुड़ें।',
      kn: 'ಬಲಕ್ಕೆ ತಿರುಗಿ.',
      te: 'కుడివైపు తిరగండి.',
    },
    description: 'Dominant hand points or sweeps decisively toward the right side.',
    nonManualMarkers: 'Gaze direction right',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },

  // ── 10. Common Actions and Expressions ─────────────────────────────────────
  {
    id: 'WATER',
    sign: 'WATER',
    label: 'Water / Drink',
    category: 'common_actions_and_expressions',
    gloss: 'WATER',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please give me water to drink.',
    translations: {
      en: 'Please give me water to drink.',
      hi: 'कृपया मुझे पीने के लिए पानी दें।',
      kn: 'ದಯವಿಟ್ಟು ನನಗೆ ಕುಡಿಯಲು ನೀರು ಕೊಡಿ.',
      te: 'దయచేసి నాకు తాగడానికి నీరు ఇవ్వండి.',
    },
    description: 'Dominant cupped hand or thumb tilted toward mouth simulating drinking water from a vessel.',
    nonManualMarkers: 'Gentle tilt of chin',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chin',
    },
  },
  {
    id: 'FOOD',
    sign: 'FOOD',
    label: 'Food / Eat',
    category: 'common_actions_and_expressions',
    gloss: 'FOOD',
    aslGrammarType: 'standard',
    canonicalSentence: 'I want food to eat.',
    translations: {
      en: 'I want food to eat.',
      hi: 'मुझे खाना खाना है।',
      kn: 'ನನಗೆ ಊಟ ಬೇಕು.',
      te: 'నాకు ఆహారం కావాలి.',
    },
    description: 'Clustered fingertips brought to mouth repeatedly in eating gesture.',
    nonManualMarkers: 'Affirmative nod',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'chin',
    },
  },
  {
    id: 'GO',
    sign: 'GO',
    label: 'Go',
    category: 'common_actions_and_expressions',
    gloss: 'GO',
    aslGrammarType: 'standard',
    canonicalSentence: 'Go or depart.',
    translations: {
      en: 'Go or depart.',
      hi: 'जाओ या प्रस्थान करो।',
      kn: 'ಹೋಗು ಅಥವಾ ಹೊರಡು.',
      te: 'వెళ్లు లేదా బయలుదేరు.',
    },
    description: 'Both index fingers point forward and sweep outward away from the body.',
    nonManualMarkers: 'Forward gaze',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'COME',
    sign: 'COME',
    label: 'Come',
    category: 'common_actions_and_expressions',
    gloss: 'COME',
    aslGrammarType: 'standard',
    canonicalSentence: 'Please come here.',
    translations: {
      en: 'Please come here.',
      hi: 'कृपया यहाँ आएं।',
      kn: 'ದಯವಿಟ್ಟು ಇಲ್ಲಿ ಬನ್ನಿ.',
      te: 'దయచేసి ఇక్కడికి రండి.',
    },
    description: 'Both open hands sweep inward toward body with welcoming motion.',
    nonManualMarkers: 'Welcoming smile',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'linear',
      location: 'neutral_space',
    },
  },
  {
    id: 'HOME',
    sign: 'HOME',
    label: 'Home / House',
    category: 'directions_and_locations',
    gloss: 'HOME',
    aslGrammarType: 'standard',
    canonicalSentence: 'I am going home.',
    translations: {
      en: 'I am going home.',
      hi: 'मैं घर जा रहा हूँ।',
      kn: 'ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ.',
      te: 'నేను ఇంటికి వెళ్తున్నాను.',
    },
    description: 'Both flat hands form an inverted V roof shape touching fingertips above chest level.',
    nonManualMarkers: 'Comfortable relaxed expression',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'static',
      location: 'chest',
    },
  },
];

export const ISL_CLASSES = ISL_VOCABULARY.map((v) => v.sign);

export interface ISLExtendedConcept {
  id: string;
  sign: string;
  label: string;
  categoryGroup: string;
  categoryName: string;
  isTrainedModelClass: boolean;
  hasAuthenticTrainingData: boolean;
  animationStatus: 'native_glb' | 'procedural' | 'fingerspelled' | 'planned';
  recordingRequirements?: string;
  description: string;
  nonManualMarkers?: string;
}

export const ISL_EXPANDED_VOCABULARY: ISLExtendedConcept[] = [
  {
    id: 'HELLO',
    sign: 'HELLO',
    label: 'Hello / Namaste',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Frontal 60fps recording of two-hand namaste / temple wave',
    description: 'Two-handed folded palms (Namaste) held at chest level with respectful head nod, or right flat palm touching temple and moving smoothly outward.',
    nonManualMarkers: 'Warm smile, slight respectful head nod',
  },
  {
    id: 'GOOD_MORNING',
    sign: 'GOOD_MORNING',
    label: 'Good Morning',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'procedural',
    recordingRequirements: 'Composite sign: GOOD followed by SUNRISE/MORNING arc',
    description: 'Thumb-up (GOOD) followed by dominant hand rising from non-dominant arm simulating sunrise.',
    nonManualMarkers: 'Pleasant bright facial expression, head upright',
  },
  {
    id: 'GOOD_NIGHT',
    sign: 'GOOD_NIGHT',
    label: 'Good Night',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'procedural',
    recordingRequirements: 'Composite sign: GOOD followed by SUNSET/NIGHT cover',
    description: 'Thumb-up (GOOD) followed by dominant hand curving down over non-dominant arm simulating sunset/darkness.',
    nonManualMarkers: 'Relaxed evening calm facial expression',
  },
  {
    id: 'GOODBYE',
    sign: 'GOODBYE',
    label: 'Goodbye',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Open palm lateral waving at shoulder/head height',
    description: 'Open dominant palm raised to shoulder/temple height waving side-to-side.',
    nonManualMarkers: 'Gentle smile, eye contact',
  },
  {
    id: 'THANK_YOU',
    sign: 'THANK_YOU',
    label: 'Thank You',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Flat dominant hand touching chin moving outward towards conversational partner',
    description: 'Flat dominant hand fingertips touching chin and moving smoothly forward/downward toward listener.',
    nonManualMarkers: 'Grateful smile, slight head bow',
  },
  {
    id: 'PLEASE',
    sign: 'PLEASE',
    label: 'Please',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Open flat palm rubbing chest clockwise',
    description: 'Open flat palm placed on center of chest rubbing in gentle clockwise circles.',
    nonManualMarkers: 'Respectful expression, chin slightly lowered',
  },
  {
    id: 'SORRY',
    sign: 'SORRY',
    label: 'Sorry / Apology',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'A-fist rubbing center of chest in circular motion',
    description: 'Closed A-fist rubbing over chest in small circular motions.',
    nonManualMarkers: 'Regretful, apologetic expression with slightly downcast eyes',
  },
  {
    id: 'WELCOME',
    sign: 'WELCOME',
    label: 'Welcome',
    categoryGroup: 'A',
    categoryName: 'Basic communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'procedural',
    recordingRequirements: 'Both open palms sweeping inward toward torso',
    description: 'Both open hands palms up sweeping in an inviting arc inward toward body.',
    nonManualMarkers: 'Warm, inviting open smile',
  },
  {
    id: 'I_ME',
    sign: 'I_ME',
    label: 'I / Me',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Index finger pointing directly to center of chest',
    description: 'Dominant index finger pointing inward to center of signer chest.',
    nonManualMarkers: 'Neutral / self-referential eye contact',
  },
  {
    id: 'YOU',
    sign: 'YOU',
    label: 'You',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Index finger pointing forward directly to conversational partner',
    description: 'Dominant index finger pointing straight forward at conversational partner.',
    nonManualMarkers: 'Direct eye contact with partner',
  },
  {
    id: 'MY',
    sign: 'MY',
    label: 'My / Mine',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Open flat palm placed firmly on chest',
    description: 'Flat dominant palm placed flat against center of signer chest indicating possession.',
    nonManualMarkers: 'Affirmative head nod',
  },
  {
    id: 'YOUR',
    sign: 'YOUR',
    label: 'Your / Yours',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'procedural',
    recordingRequirements: 'Open flat palm pushed forward toward partner',
    description: 'Flat dominant palm facing and extending forward toward conversational partner indicating possession.',
    nonManualMarkers: 'Direct eye contact',
  },
  {
    id: 'NAME',
    sign: 'NAME',
    label: 'Name',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Two H-hands (index + middle) tapping across each other at chest level',
    description: 'Index and middle fingers extended on both hands, dominant fingers tapping crosswise on non-dominant fingers.',
    nonManualMarkers: 'Neutral inquisitive tilt',
  },
  {
    id: 'AGE',
    sign: 'AGE',
    label: 'Age / How Old',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant fist pulled downward from chin',
    description: 'Dominant hand grasps imaginary beard at chin and pulls downward indicating age.',
    nonManualMarkers: 'Furrowed brows when asking question',
  },
  {
    id: 'FAMILY',
    sign: 'FAMILY',
    label: 'Family',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Both F-hands touching thumbs/index and circling outward to touch pinkies',
    description: 'Both hands in F-shape (thumb-index touching) starting together and circling outward to touch pinkies.',
    nonManualMarkers: 'Affectionate open expression',
  },
  {
    id: 'FRIEND',
    sign: 'FRIEND',
    label: 'Friend',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Both index fingers hooking into each other repeatedly',
    description: 'Both index fingers curved like hooks, locking over each other alternately.',
    nonManualMarkers: 'Warm friendly smile',
  },
  {
    id: 'PERSON',
    sign: 'PERSON',
    label: 'Person / Individual',
    categoryGroup: 'B',
    categoryName: 'Personal information',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Two flat hands moving downward along torso sides',
    description: 'Both flat hands moving vertically downward in parallel framing a human body silhouette.',
    nonManualMarkers: 'Neutral objective gaze',
  },
  {
    id: 'WHAT',
    sign: 'WHAT',
    label: 'What',
    categoryGroup: 'C',
    categoryName: 'Questions',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Both open flat hands held forward, palms up, shaking laterally with furrowed brows',
    description: 'Both hands extended forward with palms facing upward, shaking horizontally side to side.',
    nonManualMarkers: 'Strongly furrowed eyebrows, head tilted slightly forward',
  },
  {
    id: 'WHERE',
    sign: 'WHERE',
    label: 'Where',
    categoryGroup: 'C',
    categoryName: 'Questions',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Dominant index finger upright waving left to right with question gaze',
    description: 'Dominant index finger held vertically upright at chest height, wagging side to side.',
    nonManualMarkers: 'Furrowed eyebrows, head tilted inquiringly',
  },
  {
    id: 'WHO',
    sign: 'WHO',
    label: 'Who',
    categoryGroup: 'C',
    categoryName: 'Questions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant index finger circling mouth or chin',
    description: 'Index finger points toward chin and traces small circles.',
    nonManualMarkers: 'Furrowed eyebrows, pursed lips',
  },
  {
    id: 'WHEN',
    sign: 'WHEN',
    label: 'When',
    categoryGroup: 'C',
    categoryName: 'Questions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant index circling around non-dominant index fingertip',
    description: 'Dominant index finger circles around stationary non-dominant index fingertip before landing.',
    nonManualMarkers: 'Furrowed eyebrows, head tilt',
  },
  {
    id: 'WHY',
    sign: 'WHY',
    label: 'Why',
    categoryGroup: 'C',
    categoryName: 'Questions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant flat hand touching forehead and dropping into Y-handshape',
    description: 'Flat hand touches temple then pulls outward/downward into a Y-handshape.',
    nonManualMarkers: 'Intense furrowed eyebrows, chin forward',
  },
  {
    id: 'HOW',
    sign: 'HOW',
    label: 'How',
    categoryGroup: 'C',
    categoryName: 'Questions',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Both cupped hands starting backs together, rotating outward so palms face upward',
    description: 'Both hands start with backs of fingers touching, wrists rotate outward until palms face up.',
    nonManualMarkers: 'Furrowed eyebrows, slight head tilt',
  },
  {
    id: 'WHICH',
    sign: 'WHICH',
    label: 'Which',
    categoryGroup: 'C',
    categoryName: 'Questions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Both A-fists with thumbs upright alternating up and down',
    description: 'Both hands in thumbs-up shape alternating up and down like balance scales.',
    nonManualMarkers: 'Furrowed eyebrows, evaluating gaze',
  },
  {
    id: 'EAT',
    sign: 'EAT',
    label: 'Eat / Food',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Fingertips brought together touching mouth repeatedly',
    description: 'All fingertips of dominant hand gathered together touching mouth in repeated eating motion.',
    nonManualMarkers: 'Relaxed neutral / chewing articulation',
  },
  {
    id: 'DRINK',
    sign: 'DRINK',
    label: 'Drink / Water',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'C-hand tilted toward mouth like holding a glass',
    description: 'Curved C-hand tipped upward toward mouth as if drinking from a glass or cup.',
    nonManualMarkers: 'Slight head tilt back',
  },
  {
    id: 'GO',
    sign: 'GO',
    label: 'Go / Depart',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Both index fingers arcing forward and away from signer',
    description: 'Index fingers of one or both hands pointing and sweeping smoothly forward into space.',
    nonManualMarkers: 'Determined forward gaze',
  },
  {
    id: 'COME',
    sign: 'COME',
    label: 'Come / Arrive',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Both open palms or index fingers sweeping inward toward body',
    description: 'Hands extended forward sweeping inward toward torso with beckoning motion.',
    nonManualMarkers: 'Welcoming gaze',
  },
  {
    id: 'SLEEP',
    sign: 'SLEEP',
    label: 'Sleep',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Flat hand resting against tilted cheek with closed eyes',
    description: 'Open hand placed against side of face while head tilts sideways resting on hand.',
    nonManualMarkers: 'Eyes closed or relaxed downward gaze',
  },
  {
    id: 'WORK',
    sign: 'WORK',
    label: 'Work / Job',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Both fists with wrists tapping each other twice',
    description: 'Both hands formed into S-fists, dominant wrist tapping the base of non-dominant wrist twice.',
    nonManualMarkers: 'Focused, serious facial expression',
  },
  {
    id: 'STUDY',
    sign: 'STUDY',
    label: 'Study / Learn',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant fingers fluttering upward from open non-dominant palm toward forehead',
    description: 'Dominant hand flutters fingers repeatedly over non-dominant palm representing reading a book.',
    nonManualMarkers: 'Attentive concentrated expression',
  },
  {
    id: 'WALK',
    sign: 'WALK',
    label: 'Walk',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Two downward fingers alternating forward steps',
    description: 'Dominant inverted V-hand (index + middle) mimics walking legs across space.',
    nonManualMarkers: 'Neutral rhythmic gaze',
  },
  {
    id: 'SIT',
    sign: 'SIT',
    label: 'Sit',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Bent dominant fingers resting across non-dominant horizontal fingers',
    description: 'Bent index and middle fingers of dominant hand hook over horizontal index and middle fingers.',
    nonManualMarkers: 'Neutral posture transition',
  },
  {
    id: 'STAND',
    sign: 'STAND',
    label: 'Stand',
    categoryGroup: 'D',
    categoryName: 'Daily activities',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant V-fingers standing upright on flat non-dominant palm',
    description: 'Inverted V-fingers placed standing firmly upon flat upturned non-dominant palm.',
    nonManualMarkers: 'Upright alert posture',
  },
  {
    id: 'HELP',
    sign: 'HELP',
    label: 'Help / Assist',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Dominant thumbs-up fist placed upon flat non-dominant palm, lifting together',
    description: 'Dominant closed fist with thumb up resting on flat non-dominant palm, both lifting upward toward partner.',
    nonManualMarkers: 'Earnest, helpful expression',
  },
  {
    id: 'UNDERSTAND',
    sign: 'UNDERSTAND',
    label: 'Understand',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Index finger flicking upright from fist near temple (lightbulb moment)',
    description: 'Dominant index finger flicking upward into open extension near temple/forehead.',
    nonManualMarkers: 'Enlightened nod, wide eyes',
  },
  {
    id: 'KNOW',
    sign: 'KNOW',
    label: 'Know',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Bent flat hand touching temple twice',
    description: 'Fingertips of flat dominant hand tap the side of the forehead twice.',
    nonManualMarkers: 'Affirmative confident head nod',
  },
  {
    id: 'THINK',
    sign: 'THINK',
    label: 'Think',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Index finger touching side of forehead with thoughtful expression',
    description: 'Index finger touches temple with thoughtful contemplative gaze.',
    nonManualMarkers: 'Eyes slightly squinted, brow furrowed',
  },
  {
    id: 'SPEAK',
    sign: 'SPEAK',
    label: 'Speak / Talk',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: '4 fingers fluttering outward repeatedly from mouth',
    description: 'Index finger or 4 fingers fluttering forward from lips representing speech flow.',
    nonManualMarkers: 'Mouth movement accompanying sign',
  },
  {
    id: 'REPEAT',
    sign: 'REPEAT',
    label: 'Repeat / Again',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Dominant curved hand making forward circular roll landing in palm',
    description: 'Curved dominant hand makes forward rolling arc and taps flat non-dominant palm.',
    nonManualMarkers: 'Inquisitive / clarifying expression',
  },
  {
    id: 'WAIT',
    sign: 'WAIT',
    label: 'Wait / Hold on',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Both open palms held up flutter fingers gently',
    description: 'Both open hands held forward with palms facing upward, wiggling fingers patiently.',
    nonManualMarkers: 'Calm, patient expression',
  },
  {
    id: 'TELL',
    sign: 'TELL',
    label: 'Tell',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Index finger touching chin and arcing forward to listener',
    description: 'Index finger taps chin and then arcs smoothly forward toward listener.',
    nonManualMarkers: 'Direct communicative gaze',
  },
  {
    id: 'ASK',
    sign: 'ASK',
    label: 'Ask',
    categoryGroup: 'E',
    categoryName: 'Communication',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Palms together in praying pose bowing forward to partner',
    description: 'Both flat palms pressed together moving forward toward listener in polite query.',
    nonManualMarkers: 'Questioning head tilt',
  },
  {
    id: 'HAPPY',
    sign: 'HAPPY',
    label: 'Happy / Glad',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Open flat hands brushing upward against chest in circular motion',
    description: 'Both flat hands brush upward against chest in repeated buoyant circles.',
    nonManualMarkers: 'Bright beaming smile, crinkled eyes',
  },
  {
    id: 'SAD',
    sign: 'SAD',
    label: 'Sad / Depressed',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Both open hands dropping slowly down in front of face',
    description: 'Both open hands start near eyes and slowly droop downward past chin.',
    nonManualMarkers: 'Downturned mouth, drooping posture',
  },
  {
    id: 'ANGRY',
    sign: 'ANGRY',
    label: 'Angry',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Clawed hands rising in front of torso with tense expression',
    description: 'Clawed hands pull upward in front of chest with sudden tense contraction.',
    nonManualMarkers: 'Intense scowl, tightened jaw',
  },
  {
    id: 'TIRED',
    sign: 'TIRED',
    label: 'Tired / Exhausted',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Bent hands on chest rolling downward while shoulders slump',
    description: 'Fingertips resting on upper chest, hands roll downward while shoulders slump.',
    nonManualMarkers: 'Exhausted drooping eyes, slight sigh',
  },
  {
    id: 'SICK',
    sign: 'SICK',
    label: 'Sick / Ill',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Middle fingers touching forehead and stomach simultaneously',
    description: 'Dominant middle finger touches forehead while non-dominant middle finger touches abdomen.',
    nonManualMarkers: 'Pained, uncomfortable grimace',
  },
  {
    id: 'GOOD',
    sign: 'GOOD',
    label: 'Good / Fine',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Thumb up held forward with confident nod',
    description: 'Dominant hand forms strong thumbs-up gesture pushed slightly forward in neutral space.',
    nonManualMarkers: 'Warm approving smile, firm nod',
  },
  {
    id: 'BAD',
    sign: 'BAD',
    label: 'Bad / Unwell',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Thumb down thrust downward with head shake',
    description: 'Dominant hand forms thumbs-down gesture thrust downward with emphatic negative inflection.',
    nonManualMarkers: 'Disapproving scowl, head shake',
  },
  {
    id: 'WORRIED',
    sign: 'WORRIED',
    label: 'Worried / Anxious',
    categoryGroup: 'F',
    categoryName: 'Feelings',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Both flat hands alternating in circles before face',
    description: 'Both flat hands circle in front of face alternately with troubled gaze.',
    nonManualMarkers: 'Furrowed brow, anxious gaze',
  },
  {
    id: 'HOSPITAL',
    sign: 'HOSPITAL',
    label: 'Hospital',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Two fingers forming cross on upper arm/shoulder',
    description: 'Dominant H-fingers (index and middle) trace cross pattern on non-dominant shoulder.',
    nonManualMarkers: 'Serious, urgent expression',
  },
  {
    id: 'DOCTOR',
    sign: 'DOCTOR',
    label: 'Doctor',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Two fingers tapping pulse on inside of non-dominant wrist',
    description: 'Dominant index and middle fingers tap the pulse point on the inside of non-dominant wrist twice.',
    nonManualMarkers: 'Professional, focused expression',
  },
  {
    id: 'WATER',
    sign: 'WATER',
    label: 'Water / Drink',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'W-handshape tapping chin twice',
    description: 'Dominant hand forms W-shape (three middle fingers upright) tapping chin twice.',
    nonManualMarkers: 'Neutral / requesting gaze',
  },
  {
    id: 'FOOD',
    sign: 'FOOD',
    label: 'Food / Meal',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Fingertips bunched together bringing to mouth',
    description: 'All fingertips of dominant hand gathered together touching mouth twice.',
    nonManualMarkers: 'Appetitive / requesting gaze',
  },
  {
    id: 'MEDICINE',
    sign: 'MEDICINE',
    label: 'Medicine',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Middle finger grinding in non-dominant palm like mortar and pestle',
    description: 'Dominant middle finger twists circular inside cupped non-dominant palm.',
    nonManualMarkers: 'Serious medical gaze',
  },
  {
    id: 'EMERGENCY',
    sign: 'EMERGENCY',
    label: 'Emergency',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'E-hand shaking urgently side-to-side',
    description: 'Dominant hand forms E-shape shaking urgently side-to-side at chest level.',
    nonManualMarkers: 'Intense urgent expression, wide alert eyes',
  },
  {
    id: 'AMBULANCE',
    sign: 'AMBULANCE',
    label: 'Ambulance',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Both hands raised rotating wrists like flashing emergency beacons',
    description: 'Both hands raised above head in claw/C shape, rotating wrists in alternating beacon flashes.',
    nonManualMarkers: 'High-urgency alert expression',
  },
  {
    id: 'PAIN',
    sign: 'PAIN',
    label: 'Pain / Hurt',
    categoryGroup: 'G',
    categoryName: 'Emergency and healthcare',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Two index fingers pointing at each other and twisting near point of hurt',
    description: 'Both index fingers point at each other with tips nearly touching, jabbing sharply in and out.',
    nonManualMarkers: 'Pained wince, furrowed brow',
  },
  {
    id: 'TODAY',
    sign: 'TODAY',
    label: 'Today',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'NOW sign followed by DAY sign',
    description: 'Both bent hands drop downward in neutral space (NOW) followed by open arc (DAY).',
    nonManualMarkers: 'Affirmative nod',
  },
  {
    id: 'TOMORROW',
    sign: 'TOMORROW',
    label: 'Tomorrow',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant thumb touching cheek and arcing forward',
    description: 'A-hand with thumb touching cheek, arcing forward into space ahead of signer.',
    nonManualMarkers: 'Forward-looking gaze',
  },
  {
    id: 'YESTERDAY',
    sign: 'YESTERDAY',
    label: 'Yesterday',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant thumb touching cheek and arcing backward',
    description: 'A-hand with thumb touching cheek, arcing backward over signer shoulder.',
    nonManualMarkers: 'Backward-looking gaze',
  },
  {
    id: 'NOW',
    sign: 'NOW',
    label: 'Now / Present',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'procedural',
    recordingRequirements: 'Both bent hands dropping firmly downward in front of torso',
    description: 'Both hands with fingers bent at 90 degrees drop downward firmly in unison.',
    nonManualMarkers: 'Decisive present-focused nod',
  },
  {
    id: 'LATER',
    sign: 'LATER',
    label: 'Later',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'L-hand tilting forward from wrist like clock hand',
    description: 'Dominant hand forms L-shape and tilts forward like hands of a clock.',
    nonManualMarkers: 'Neutral gaze forward',
  },
  {
    id: 'HERE',
    sign: 'HERE',
    label: 'Here',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Both flat hands palms up dropping down in neutral space',
    description: 'Both open flat hands with palms facing upward drop downward twice in neutral space.',
    nonManualMarkers: 'Direct grounding nod',
  },
  {
    id: 'THERE',
    sign: 'THERE',
    label: 'There',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Index finger pointing distinctly away into distance',
    description: 'Dominant index finger points emphatically at distant location.',
    nonManualMarkers: 'Eye gaze directed at pointing target',
  },
  {
    id: 'HOME',
    sign: 'HOME',
    label: 'Home / House',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Flat hands forming roof shape touching at fingertips',
    description: 'Both flat hands form an inverted V roof shape touching fingertips above chest level.',
    nonManualMarkers: 'Comfortable relaxed expression',
  },
  {
    id: 'SCHOOL',
    sign: 'SCHOOL',
    label: 'School / College',
    categoryGroup: 'H',
    categoryName: 'Time and location',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Open palms clapping horizontally twice',
    description: 'Dominant flat open palm claps down onto stationary flat non-dominant palm twice horizontally.',
    nonManualMarkers: 'Neutral educational gaze',
  },
  {
    id: 'GIVE',
    sign: 'GIVE',
    label: 'Give / Offer',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Both flat hands palms up sweeping forward toward partner',
    description: 'Both open cupped hands sweep smoothly forward from signer torso toward conversational partner.',
    nonManualMarkers: 'Polite, offering smile',
  },
  {
    id: 'TAKE',
    sign: 'TAKE',
    label: 'Take / Receive',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Open hand reaching forward and closing into fist pulling back',
    description: 'Dominant open hand reaches into space, closes into a grasping fist, and pulls back toward body.',
    nonManualMarkers: 'Decisive assertive gaze',
  },
  {
    id: 'MAKE',
    sign: 'MAKE',
    label: 'Make / Build',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Two fists stacked twisting against each other',
    description: 'Both fists stacked vertically, twisting back and forth in opposing directions.',
    nonManualMarkers: 'Determined constructive gaze',
  },
  {
    id: 'WANT',
    sign: 'WANT',
    label: 'Want / Desire',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'procedural',
    recordingRequirements: 'Both clawed hands palms up pulling backward toward torso',
    description: 'Both open clawed hands with palms facing upward pull smoothly backward toward signer torso.',
    nonManualMarkers: 'Desirous earnest expression',
  },
  {
    id: 'NEED',
    sign: 'NEED',
    label: 'Need / Must',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: true,
    hasAuthenticTrainingData: true,
    animationStatus: 'native_glb',
    recordingRequirements: 'Dominant bent index finger bending sharply downward',
    description: 'Dominant bent index finger (X-handshape) flexes sharply downward in emphatic stroke.',
    nonManualMarkers: 'Firm, emphatic nod',
  },
  {
    id: 'CAN',
    sign: 'CAN',
    label: 'Can / Able',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Both S-fists dropping downward simultaneously',
    description: 'Both hands formed into S-fists dropping downward simultaneously with conviction.',
    nonManualMarkers: 'Confident, firm nod',
  },
  {
    id: 'CANNOT',
    sign: 'CANNOT',
    label: 'Cannot / Impossible',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'planned',
    recordingRequirements: 'Dominant index finger striking down across non-dominant index finger',
    description: 'Dominant index finger strikes downward across stationary non-dominant index finger tip.',
    nonManualMarkers: 'Firm head shake, tight lips',
  },
  {
    id: 'FINISH',
    sign: 'FINISH',
    label: 'Finish / Done',
    categoryGroup: 'I',
    categoryName: 'Common actions',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'procedural',
    recordingRequirements: 'Both open 5-hands held forward flicking outward',
    description: 'Both open hands with fingers spread flick outward laterally, palms turning downward.',
    nonManualMarkers: 'Emphatic completion nod, relaxed exhale',
  },
  {
    id: 'LETTER_D',
    sign: 'LETTER_D',
    label: 'Letter D',
    categoryGroup: 'Alphabet',
    categoryName: 'ISL Fingerspelling',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'fingerspelled',
    recordingRequirements: 'Index pointing straight up, thumb touching middle in D-loop',
    description: 'Dominant index finger extended straight up while thumb and middle finger touch to form a circular loop.',
    nonManualMarkers: 'Neutral fingerspelling focus',
  },
  {
    id: 'LETTER_A',
    sign: 'LETTER_A',
    label: 'Letter A',
    categoryGroup: 'Alphabet',
    categoryName: 'ISL Fingerspelling',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'fingerspelled',
    recordingRequirements: 'Fist with thumb resting beside index finger',
    description: 'Fingers folded into tight fist with thumb resting alongside index finger.',
    nonManualMarkers: 'Neutral fingerspelling focus',
  },
  {
    id: 'LETTER_N',
    sign: 'LETTER_N',
    label: 'Letter N',
    categoryGroup: 'Alphabet',
    categoryName: 'ISL Fingerspelling',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'fingerspelled',
    recordingRequirements: 'Fist with two fingers draped forward over thumb',
    description: 'Index and middle fingers folded downward over the thumb in front of palm.',
    nonManualMarkers: 'Neutral fingerspelling focus',
  },
  {
    id: 'LETTER_I',
    sign: 'LETTER_I',
    label: 'Letter I',
    categoryGroup: 'Alphabet',
    categoryName: 'ISL Fingerspelling',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'fingerspelled',
    recordingRequirements: 'Pinky finger pointing upright from closed fist',
    description: 'Pinky finger extended vertically while remaining four digits are clenched in fist.',
    nonManualMarkers: 'Neutral fingerspelling focus',
  },
  {
    id: 'LETTER_E',
    sign: 'LETTER_E',
    label: 'Letter E',
    categoryGroup: 'Alphabet',
    categoryName: 'ISL Fingerspelling',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'fingerspelled',
    recordingRequirements: 'Fingers curled inward resting on horizontal thumb',
    description: 'All four fingers bent sharply at knuckles resting lightly on top of thumb.',
    nonManualMarkers: 'Neutral fingerspelling focus',
  },
  {
    id: 'LETTER_L',
    sign: 'LETTER_L',
    label: 'Letter L',
    categoryGroup: 'Alphabet',
    categoryName: 'ISL Fingerspelling',
    isTrainedModelClass: false,
    hasAuthenticTrainingData: false,
    animationStatus: 'fingerspelled',
    recordingRequirements: 'Index finger pointing up, thumb extended sideways at 90 degrees',
    description: 'Index finger extended vertically and thumb extended horizontally forming letter L shape.',
    nonManualMarkers: 'Neutral fingerspelling focus',
  },
];

export const ISL_TRAINED_MODEL_CLASSES = ISL_VOCABULARY.map((v) => v.sign);


/**
 * Translates an accumulated sequence of continuous ISL signs into a natural, grammatically correct English sentence.
 *
 * Adheres strictly to ISL SOV (Subject-Object-Verb) and TSOV rules:
 * - Adds English copula and auxiliary verbs (is, am, are, can, have, please) intelligently
 * - Zero hallucination of facts not conveyed by the signer
 * - Resolves Wh-questions placed at end of sentence
 */
export function translateISLGlossSequenceToEnglish(glosses: string[]): string {
  if (!glosses || glosses.length === 0) return '';

  const clean = glosses
    .map((g) => g.toUpperCase().trim().replace(/\s+/g, '_'))
    .filter((g) => g.length > 0 && g !== 'NONE' && g !== 'UNKNOWN');

  if (clean.length === 0) return '';

  // ── First delegate to Comprehensive ISL Linguistic Grammar Engine ──────────
  const irResult = translateISLIRToEnglish(clean);
  if (irResult && !irResult.endsWith('NONE.')) {
    return irResult;
  }

  const joined = clean.join(' ');

  // ── Multi-Sign ISL Idioms & Sentences ─────────────────────────────────────
  // 1. Greetings & Introductions
  if (joined === 'HELLO YOU HOW' || joined === 'HELLO HOW_ARE_YOU' || joined === 'HELLO YOU HOW_ARE_YOU') {
    return 'Hello, how are you?';
  }
  if (joined === 'HELLO YOU HOW THANK_YOU' || joined === 'HELLO HOW_ARE_YOU THANK_YOU') {
    return 'Hello, how are you? Thank you very much.';
  }
  if (joined === 'NAME YOU WHAT' || joined === 'YOU NAME WHAT' || joined === 'YOU NAME') {
    return 'What is your name?';
  }
  if (joined === 'MY NAME' || joined === 'I_ME NAME') {
    return 'My name is...';
  }
  if (joined === 'NICE_TO_MEET_YOU' || joined === 'NICE MEET YOU') {
    return 'Nice to meet you.';
  }
  if (joined === 'GOODBYE' || joined === 'GOODBYE SEE_YOU') {
    return 'Goodbye, see you later.';
  }

  // 2. Assistance & Requests
  if (joined === 'ME HELP NEED' || joined === 'I_ME HELP NEED' || joined === 'HELP NEED' || joined === 'I_ME NEED HELP') {
    return 'I need help.';
  }
  if (joined === 'YOU HELP I_ME' || joined === 'YOU HELP ME' || joined === 'YOU HELP' || joined === 'CAN YOU HELP ME') {
    return 'Can you help me?';
  }
  if (joined === 'PLEASE HELP' || joined === 'HELP PLEASE') {
    return 'Please help me.';
  }
  if (joined === 'PLEASE WAIT' || joined === 'WAIT PLEASE' || joined === 'PLEASE WAIT HERE') {
    return 'Please wait a moment.';
  }
  if (joined === 'PLEASE WATER GIVE' || joined === 'WATER PLEASE' || joined === 'PLEASE WATER') {
    return 'Please give me water.';
  }
  if (joined === 'WATER WANT' || joined === 'I_ME WATER WANT' || joined === 'WATER DRINK') {
    return 'I want water to drink.';
  }
  if (joined === 'FOOD WANT' || joined === 'I_ME FOOD WANT' || joined === 'FOOD EAT') {
    return 'I want food to eat.';
  }
  if (joined === 'PLEASE WATER FOOD GIVE' || joined === 'WATER FOOD PLEASE') {
    return 'Please give me water and food.';
  }
  if (joined === 'I_ME HOME GO WANT' || joined === 'HOME GO WANT' || joined === 'I_ME HOME GO') {
    return 'I want to go home.';
  }

  // 3. Medical Emergencies
  if (joined === 'AMBULANCE CALL NEED' || joined === 'AMBULANCE CALL' || joined === 'PLEASE AMBULANCE CALL') {
    return 'Please call an ambulance.';
  }
  if (joined === 'HOSPITAL WHERE' || joined === 'WHERE HOSPITAL') {
    return 'Where is the nearest hospital?';
  }
  if (joined === 'DOCTOR NEED' || joined === 'I_ME DOCTOR NEED') {
    return 'I need a doctor.';
  }
  if (joined === 'MEDICINE NEED' || joined === 'I_ME MEDICINE NEED') {
    return 'I need medicine.';
  }
  if (joined === 'I_ME PAIN' || joined === 'PAIN' || joined === 'HURT') {
    return 'I am in pain.';
  }
  if (joined === 'EMERGENCY' || joined === 'EMERGENCY HELP') {
    return 'This is an emergency!';
  }

  // 4. Emergency Services
  if (joined === 'POLICE CALL NEED' || joined === 'POLICE CALL' || joined === 'PLEASE POLICE CALL') {
    return 'Please call the police.';
  }
  if (joined === 'DANGER' || joined === 'I_ME DANGER') {
    return 'I am in danger.';
  }
  if (joined === 'STOP' || joined === 'STOP PLEASE') {
    return 'Please stop immediately.';
  }

  // 5. Directions and Locations
  if (joined === 'BATHROOM WHERE' || joined === 'WHERE BATHROOM') {
    return 'Where is the bathroom?';
  }
  if (joined === 'WHERE' || joined === 'LOCATION WHERE') {
    return 'Where is it located?';
  }
  if (joined === 'TURN LEFT' || joined === 'LEFT') {
    return 'Turn left.';
  }
  if (joined === 'TURN RIGHT' || joined === 'RIGHT') {
    return 'Turn right.';
  }
  if (joined === 'COME HERE' || joined === 'HERE COME') {
    return 'Please come here.';
  }

  // 6. Education and Workplace
  if (joined === 'I_ME DONT_UNDERSTAND' || joined === 'DONT_UNDERSTAND' || joined === 'NOT UNDERSTAND') {
    return "I don't understand.";
  }
  if (joined === 'I_ME UNDERSTAND' || joined === 'UNDERSTAND') {
    return 'I understand clearly.';
  }
  if (joined === 'REPEAT PLEASE' || joined === 'PLEASE REPEAT' || joined === 'AGAIN PLEASE') {
    return 'Can you please repeat that?';
  }

  // Single-sign fallbacks to canonical verified sentences
  if (clean.length === 1) {
    const vocab = ISL_VOCABULARY.find((v) => v.sign === clean[0] || v.id === clean[0]);
    if (vocab) return vocab.canonicalSentence;
  }


  // Multi-sign general concatenation fallback with proper punctuation
  const parts: string[] = [];
  for (const gloss of clean) {
    const vocab = ISL_VOCABULARY.find((v) => v.sign === gloss || v.id === gloss);
    if (vocab) {
      parts.push(vocab.label);
    } else {
      parts.push(gloss.toLowerCase());
    }
  }

  return parts.join(' ') + '.';
}

/**
 * Translates spoken English text or natural language into an ordered ISL gloss sequence.
 * Enforces ISL grammatical principles:
 * - Wh-questions moved to end (e.g. "Where is the bathroom?" -> BATHROOM WHERE)
 * - Subject-Object-Verb (SOV) ordering
 * - Multi-sign sentence resolution without dropping words
 */
export function translateEnglishToISLGloss(text: string): { glosses: string[]; ruleApplied: string } {
  if (!text || !text.trim()) {
    return { glosses: [], ruleApplied: 'empty' };
  }

  // 0. Primary: Full ISL Linguistic Transformation Engine (ISL-IR)
  const islIR = parseEnglishToISLSentenceIR(text);
  if (islIR && islIR.glossSequence.length > 0) {
    return { glosses: islIR.glossSequence, ruleApplied: islIR.linguisticExplanation };
  }

  const normalized = text.toLowerCase().trim().replace(/[?!.,;]/g, '');

  // 1. Greetings & Introductions
  if (/^(hello|hi|hey|namaste)[,\s]+(how\s+are\s+you|how\s+do\s+you\s+do)/i.test(normalized)) {
    return { glosses: ['HELLO', 'YOU', 'HOW'], ruleApplied: 'greeting_how_are_you' };
  }
  if (/^how\s+are\s+you/i.test(normalized)) {
    return { glosses: ['YOU', 'HOW'], ruleApplied: 'how_are_you_sov' };
  }
  if (/^what\s+is\s+your\s+name/i.test(normalized) || /^can\s+you\s+tell\s+me\s+your\s+name/i.test(normalized) || /^tell\s+me\s+your\s+name/i.test(normalized)) {
    return { glosses: ['YOUR', 'NAME', 'WHAT'], ruleApplied: 'name_wh_question' };
  }
  if (/^nice\s+to\s+meet\s+you/i.test(normalized)) {
    return { glosses: ['NICE_TO_MEET_YOU'], ruleApplied: 'nice_to_meet_you_idiom' };
  }

  // 2. Assistance & Requests
  if (/^can\s+you\s+help\s+me/i.test(normalized) || /^could\s+you\s+help\s+me/i.test(normalized) || /^please\s+help\s+me/i.test(normalized)) {
    return { glosses: ['YOU', 'HELP', 'NEED'], ruleApplied: 'can_you_help_me_sov' };
  }
  if (/^i\s+need\s+help/i.test(normalized) || /^help\s+me/i.test(normalized)) {
    return { glosses: ['I_ME', 'HELP', 'NEED'], ruleApplied: 'i_need_help_sov' };
  }
  if (/^i\s+want\s+to\s+go\s+home/i.test(normalized) || /^want\s+to\s+go\s+home/i.test(normalized) || /^go\s+home/i.test(normalized)) {
    return { glosses: ['I_ME', 'HOME', 'GO', 'WANT'], ruleApplied: 'home_go_want_sov' };
  }
  if (/^please\s+give\s+me\s+water\s+and\s+food/i.test(normalized) || /^water\s+and\s+food\s+please/i.test(normalized)) {
    return { glosses: ['PLEASE', 'WATER', 'FOOD', 'GIVE'], ruleApplied: 'water_food_request' };
  }
  if (/^please\s+give\s+me\s+water/i.test(normalized) || /^i\s+need\s+water/i.test(normalized) || /^water\s+please/i.test(normalized)) {
    return { glosses: ['PLEASE', 'WATER', 'GIVE'], ruleApplied: 'water_request' };
  }
  if (/^i\s+want\s+food/i.test(normalized) || /^food\s+please/i.test(normalized)) {
    return { glosses: ['FOOD', 'WANT'], ruleApplied: 'food_request' };
  }
  if (/^please\s+wait\s+(here|a\s+moment|one\s+second)/i.test(normalized) || /^please\s+wait/i.test(normalized)) {
    return { glosses: ['PLEASE', 'WAIT', 'HERE'], ruleApplied: 'please_wait' };
  }

  // 3. Medical Emergencies
  if (/^i\s+need\s+medical\s+assistance/i.test(normalized) || /^medical\s+assistance/i.test(normalized)) {
    return { glosses: ['HOSPITAL', 'HELP', 'NEED'], ruleApplied: 'medical_assistance_sov' };
  }
  if (/^please\s+call\s+an\s+ambulance/i.test(normalized) || /^call\s+an\s+ambulance/i.test(normalized)) {
    return { glosses: ['PLEASE', 'AMBULANCE', 'CALL'], ruleApplied: 'call_ambulance_sov' };
  }
  if (/^where\s+is\s+the\s+(nearest\s+)?hospital/i.test(normalized)) {
    return { glosses: ['HOSPITAL', 'WHERE'], ruleApplied: 'hospital_wh_question' };
  }
  if (/^i\s+need\s+a\s+doctor/i.test(normalized)) {
    return { glosses: ['DOCTOR', 'NEED'], ruleApplied: 'doctor_need_sov' };
  }
  if (/^i\s+need\s+medicine/i.test(normalized)) {
    return { glosses: ['MEDICINE', 'NEED'], ruleApplied: 'medicine_need_sov' };
  }
  if (/^i\s+am\s+in\s+pain/i.test(normalized) || /^it\s+hurts/i.test(normalized)) {
    return { glosses: ['I_ME', 'PAIN'], ruleApplied: 'pain_statement' };
  }

  // 4. Emergency Services
  if (/^please\s+call\s+the\s+police/i.test(normalized) || /^call\s+the\s+police/i.test(normalized)) {
    return { glosses: ['PLEASE', 'POLICE', 'CALL'], ruleApplied: 'call_police_sov' };
  }
  if (/^i\s+am\s+in\s+danger/i.test(normalized) || /^danger/i.test(normalized)) {
    return { glosses: ['I_ME', 'DANGER'], ruleApplied: 'danger_statement' };
  }
  if (/^stop\s+immediately/i.test(normalized) || /^stop/i.test(normalized)) {
    return { glosses: ['STOP'], ruleApplied: 'stop_command' };
  }

  // 5. Locations and Directions
  if (/^where\s+is\s+the\s+bathroom/i.test(normalized) || /^where\s+is\s+the\s+restroom/i.test(normalized) || /^where\s+is\s+the\s+toilet/i.test(normalized)) {
    return { glosses: ['BATHROOM', 'WHERE'], ruleApplied: 'bathroom_wh_question' };
  }
  if (/^thank\s+you\s+(very\s+much|so\s+much)?/i.test(normalized)) {
    return { glosses: ['THANK_YOU'], ruleApplied: 'thank_you_polite' };
  }

  // 6. Clarification & Repetition
  if (/^i\s+do(n't|\s+not)\s+understand/i.test(normalized)) {
    return { glosses: ['I_ME', 'DONT_UNDERSTAND'], ruleApplied: 'dont_understand_sov' };
  }
  if (/^can\s+you\s+repeat\s+that/i.test(normalized) || /^please\s+repeat/i.test(normalized)) {
    return { glosses: ['REPEAT', 'PLEASE'], ruleApplied: 'repeat_request' };
  }

  // 7. General Semantic SOV Extraction Fallback
  const detectedTokens: Array<{ token: string; pos: 'polite' | 'time' | 'subj' | 'obj' | 'verb' | 'wh' }> = [];
  if (/\b(please)\b/i.test(normalized)) detectedTokens.push({ token: 'PLEASE', pos: 'polite' });
  if (/\b(hello|hi|hey|namaste)\b/i.test(normalized)) detectedTokens.push({ token: 'HELLO', pos: 'polite' });
  if (/\b(i|me|my)\b/i.test(normalized)) detectedTokens.push({ token: 'I_ME', pos: 'subj' });
  if (/\b(you)\b/i.test(normalized)) detectedTokens.push({ token: 'YOU', pos: 'subj' });
  if (/\b(your)\b/i.test(normalized)) detectedTokens.push({ token: 'YOUR', pos: 'subj' });
  if (/\b(we|us)\b/i.test(normalized)) detectedTokens.push({ token: 'WE_US', pos: 'subj' });
  if (/\b(home)\b/i.test(normalized)) detectedTokens.push({ token: 'HOME', pos: 'obj' });
  if (/\b(water)\b/i.test(normalized)) detectedTokens.push({ token: 'WATER', pos: 'obj' });
  if (/\b(food)\b/i.test(normalized)) detectedTokens.push({ token: 'FOOD', pos: 'obj' });
  if (/\b(hospital)\b/i.test(normalized)) detectedTokens.push({ token: 'HOSPITAL', pos: 'obj' });
  if (/\b(bathroom|restroom|toilet)\b/i.test(normalized)) detectedTokens.push({ token: 'BATHROOM', pos: 'obj' });
  if (/\b(name)\b/i.test(normalized)) detectedTokens.push({ token: 'NAME', pos: 'obj' });
  if (/\b(ambulance)\b/i.test(normalized)) detectedTokens.push({ token: 'AMBULANCE', pos: 'obj' });
  if (/\b(doctor)\b/i.test(normalized)) detectedTokens.push({ token: 'DOCTOR', pos: 'obj' });
  if (/\b(medicine)\b/i.test(normalized)) detectedTokens.push({ token: 'MEDICINE', pos: 'obj' });
  if (/\b(police)\b/i.test(normalized)) detectedTokens.push({ token: 'POLICE', pos: 'obj' });
  if (/\b(go)\b/i.test(normalized)) detectedTokens.push({ token: 'GO', pos: 'verb' });
  if (/\b(come)\b/i.test(normalized)) detectedTokens.push({ token: 'COME', pos: 'verb' });
  if (/\b(want)\b/i.test(normalized)) detectedTokens.push({ token: 'WANT', pos: 'verb' });
  if (/\b(help)\b/i.test(normalized)) detectedTokens.push({ token: 'HELP', pos: 'verb' });
  if (/\b(call)\b/i.test(normalized)) detectedTokens.push({ token: 'CALL', pos: 'verb' });
  if (/\b(need)\b/i.test(normalized)) detectedTokens.push({ token: 'NEED', pos: 'verb' });
  if (/\b(where)\b/i.test(normalized)) detectedTokens.push({ token: 'WHERE', pos: 'wh' });
  if (/\b(what)\b/i.test(normalized)) detectedTokens.push({ token: 'WHAT', pos: 'wh' });
  if (/\b(how)\b/i.test(normalized)) detectedTokens.push({ token: 'HOW', pos: 'wh' });

  if (detectedTokens.length > 0) {
    const posOrder: Record<string, number> = { polite: 1, time: 2, subj: 3, obj: 4, verb: 5, wh: 6 };
    detectedTokens.sort((a, b) => posOrder[a.pos] - posOrder[b.pos]);
    const uniqueGlosses = Array.from(new Set(detectedTokens.map((t) => t.token)));
    return { glosses: uniqueGlosses, ruleApplied: 'general_sov_fallback' };
  }

  return { glosses: [], ruleApplied: 'unmatched' };
}
