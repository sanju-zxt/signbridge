import { ISLSignDefinition, DemoScenarioPreset } from './index';

export const ISL_VOCABULARY: ISLSignDefinition[] = [
  {
    id: 'HELLO',
    sign: 'HELLO',
    label: 'Hello / Namaste',
    category: 'greetings',
    gloss: 'HELLO',
    canonicalSentence: 'Hello, nice to meet you.',
    translations: {
      en: 'Hello, nice to meet you.',
      hi: 'नमस्ते, आपसे मिलकर अच्छा लगा।',
      kn: 'ನಮಸ್ಕಾರ, ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಿದ್ದು ಸಂತೋಷವಾಯಿತು.',
      te: 'నమస్కారం, మిమ్మల్ని కలవడం ఆనందంగా ఉంది.'
    },
    description: 'Open dominant hand raised to forehead/temple and extended outward in a waving salute or folded palms in Namaste.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'head'
    }
  },
  {
    id: 'THANK_YOU',
    sign: 'THANK_YOU',
    label: 'Thank You / Dhanyavad',
    category: 'greetings',
    gloss: 'THANK-YOU',
    canonicalSentence: 'Thank you very much.',
    translations: {
      en: 'Thank you very much.',
      hi: 'आपका बहुत-बहुत धन्यवाद।',
      kn: 'ತುಂಬಾ ಧನ್ಯವಾದಗಳು.',
      te: 'చాలా ధన్యవాదాలు.'
    },
    description: 'Flat open hand touching chin/lips and gently extending forward towards the conversational partner.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chin'
    }
  },
  {
    id: 'PLEASE',
    sign: 'PLEASE',
    label: 'Please / Kripya',
    category: 'greetings',
    gloss: 'PLEASE',
    canonicalSentence: 'Please help me with this.',
    translations: {
      en: 'Please help me with this.',
      hi: 'कृपया मेरी इसमें मदद करें।',
      kn: 'ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ.',
      te: 'దయచేసి నాకు సహాయం చేయండి.'
    },
    description: 'Flat palm placed on chest rubbing in a gentle clockwise circular motion.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'circular',
      location: 'chest'
    }
  },
  {
    id: 'GOODBYE',
    sign: 'GOODBYE',
    label: 'Goodbye / Alvida',
    category: 'greetings',
    gloss: 'GOODBYE',
    canonicalSentence: 'Goodbye, have a good day.',
    translations: {
      en: 'Goodbye, have a good day.',
      hi: 'अलविदा, आपका दिन शुभ हो।',
      kn: 'ವಿದಾಯ, ಶುಭ ದಿನ.',
      te: 'వీడ్కోలు, మంచి రోజు.'
    },
    description: 'Open hand raised above chest, fingers flexing or hand waving side-to-side.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space'
    }
  },
  {
    id: 'HELP',
    sign: 'HELP',
    label: 'Help / I Need Help',
    category: 'needs',
    gloss: 'HELP',
    canonicalSentence: 'I need help, please.',
    translations: {
      en: 'I need help, please.',
      hi: 'मुझे मदद की ज़रूरत है, कृपया सहायता करें।',
      kn: 'ನನಗೆ ಸಹಾಯ ಬೇಕು, ದಯವಿಟ್ಟು ಸಹಾಯ ಮಾಡಿ.',
      te: 'నాకు సహాయం కావాలి, దయచేసి సహాయం చేయండి.'
    },
    description: 'Closed fist with thumb up placed on the flat palm of non-dominant hand, lifting upward together.',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chest'
    }
  },
  {
    id: 'WATER',
    sign: 'WATER',
    label: 'Water / Paani',
    category: 'needs',
    gloss: 'WATER',
    canonicalSentence: 'I need drinking water.',
    translations: {
      en: 'I need drinking water.',
      hi: 'मुझे पीने का पानी चाहिए।',
      kn: 'ನನಗೆ ಕುಡಿಯಲು ನೀರು ಬೇಕು.',
      te: 'నాకు త్రాగే నీరు కావాలి.'
    },
    description: 'W-shaped three fingers or cupped hand brought near mouth/chin with tapping or tilting motion.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'chin'
    }
  },
  {
    id: 'FOOD',
    sign: 'FOOD',
    label: 'Food / Khana',
    category: 'needs',
    gloss: 'FOOD',
    canonicalSentence: 'I am hungry, where can I get food?',
    translations: {
      en: 'I am hungry, where can I get food?',
      hi: 'मुझे भूख लगी है, खाना कहाँ मिलेगा?',
      kn: 'ನನಗೆ ಹಸಿವಾಗಿದೆ, ಊಟ ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ?',
      te: 'నాకు ఆకలిగా ఉంది, ఆహారం ఎక్కడ దొరుకుతుంది?'
    },
    description: 'Fingertips clustered together brought to mouth repeatedly in an eating gesture.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'chin'
    }
  },
  {
    id: 'HOSPITAL',
    sign: 'HOSPITAL',
    label: 'Hospital / Doctor',
    category: 'needs',
    gloss: 'HOSPITAL',
    canonicalSentence: 'Where is the nearest hospital or clinic?',
    translations: {
      en: 'Where is the nearest hospital or clinic?',
      hi: 'निकटतम अस्पताल या डॉक्टर कहाँ है?',
      kn: 'ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ ಅಥವಾ ಕ್ಲಿನಿಕ್ ಎಲ್ಲಿದೆ?',
      te: 'సమీపంలోని ఆసుపತ್ರಿ లేదా క్లినిక్ ఎక్కడ ఉంది?'
    },
    description: 'Index finger traces a medical cross sign on upper arm or two fingers tap pulse at wrist.',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'chest'
    }
  },
  {
    id: 'BATHROOM',
    sign: 'BATHROOM',
    label: 'Bathroom / Washroom',
    category: 'needs',
    gloss: 'BATHROOM',
    canonicalSentence: 'Where is the restroom / bathroom?',
    translations: {
      en: 'Where is the restroom / bathroom?',
      hi: 'शौचालय / बाथरूम कहाँ है?',
      kn: 'ವಾಶ್‌ರೂಮ್ / ಶೌಚಾಲಯ ಎಲ್ಲಿದೆ?',
      te: 'వాష్‌రూమ్ / బాత్‌రూమ్ ఎక్కడ ఉంది?'
    },
    description: 'T-handshape (thumb between index and middle fingers) shaken side to side, or washing hands motion.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space'
    }
  },
  {
    id: 'WHERE',
    sign: 'WHERE',
    label: 'Where / Kahan',
    category: 'navigation',
    gloss: 'WHERE',
    canonicalSentence: 'Where is this located?',
    translations: {
      en: 'Where is this located?',
      hi: 'यह कहाँ स्थित है?',
      kn: 'ಇದು ಎಲ್ಲಿ ಇದೆ?',
      te: 'ఇది ఎక్కడ ఉంది?'
    },
    description: 'Both open palms face upward in neutral space, swaying gently side to side with questioning posture.',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'repetitive',
      location: 'neutral_space'
    }
  },
  {
    id: 'HERE',
    sign: 'HERE',
    label: 'Here / Yahan',
    category: 'navigation',
    gloss: 'HERE',
    canonicalSentence: 'Right here at this spot.',
    translations: {
      en: 'Right here at this spot.',
      hi: 'ठीक यहाँ इस स्थान पर।',
      kn: 'ಇಲ್ಲಿಯೇ ಈ ಸ್ಥಳದಲ್ಲಿ.',
      te: 'ఇక్కడే ఈ ప్రదేశంలో.'
    },
    description: 'Both index fingers or open palms pointing downward into immediate front space.',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'either',
      primaryMotion: 'static',
      location: 'neutral_space'
    }
  },
  {
    id: 'LEFT',
    sign: 'LEFT',
    label: 'Turn Left / Baayein',
    category: 'navigation',
    gloss: 'LEFT',
    canonicalSentence: 'Turn to the left side.',
    translations: {
      en: 'Turn to the left side.',
      hi: 'बाईं ओर मुड़ें।',
      kn: 'ಎಡಭಾಗಕ್ಕೆ ತಿರುಗಿ.',
      te: 'ఎడమ వైపు తిరగండి.'
    },
    description: 'Index finger pointing firmly towards the left direction.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space'
    }
  },
  {
    id: 'RIGHT',
    sign: 'RIGHT',
    label: 'Turn Right / Daayein',
    category: 'navigation',
    gloss: 'RIGHT',
    canonicalSentence: 'Turn to the right side.',
    translations: {
      en: 'Turn to the right side.',
      hi: 'दाईं ओर मुड़ें।',
      kn: 'ಬಲಭಾಗಕ್ಕೆ ತಿರುಗಿ.',
      te: 'కుడి వైపు తిరగండి.'
    },
    description: 'Index finger pointing firmly towards the right direction.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'neutral_space'
    }
  },
  {
    id: 'YES',
    sign: 'YES',
    label: 'Yes / Haan',
    category: 'clarification',
    gloss: 'YES',
    canonicalSentence: 'Yes, that is correct.',
    translations: {
      en: 'Yes, that is correct.',
      hi: 'हाँ, यह बिल्कुल सही है।',
      kn: 'ಹೌದು, ಅದು ಸರಿ.',
      te: 'అవును, అది సరైనది.'
    },
    description: 'Closed fist nodding up and down like a head nod.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space'
    }
  },
  {
    id: 'NO',
    sign: 'NO',
    label: 'No / Nahi',
    category: 'clarification',
    gloss: 'NO',
    canonicalSentence: 'No, that is not what I meant.',
    translations: {
      en: 'No, that is not what I meant.',
      hi: 'नहीं, मेरा यह मतलब नहीं था।',
      kn: 'ಇಲ್ಲ, ನನ್ನ ಉದ್ದೇಶ ಅದಲ್ಲ.',
      te: 'కాదు, నా ఉద్దేశం అది కాదు.'
    },
    description: 'Index and middle fingers snapping down against thumb repeatedly in a firm negation.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'repetitive',
      location: 'neutral_space'
    }
  },
  {
    id: 'DONT_UNDERSTAND',
    sign: 'DONT_UNDERSTAND',
    label: "Don't Understand / Samajh Nahi Aaya",
    category: 'clarification',
    gloss: 'DONT-UNDERSTAND',
    canonicalSentence: 'I do not understand, could you clarify?',
    translations: {
      en: 'I do not understand, could you clarify?',
      hi: 'मुझे समझ नहीं आया, क्या आप स्पष्ट कर सकते हैं?',
      kn: 'ನನಗೆ ಅರ್ಥವಾಗುತ್ತಿಲ್ಲ, ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟಪಡಿಸಬಹುದೇ?',
      te: 'నాకు అర్థం కాలేదు, మీరు స్పష్టం చేయగలరా?'
    },
    description: 'Index finger touches forehead with slight frown and flicks outward, expressing incomprehension.',
    keyFeatures: {
      hands: 'single',
      dominantHand: 'right',
      primaryMotion: 'linear',
      location: 'head'
    }
  },
  {
    id: 'REPEAT',
    sign: 'REPEAT',
    label: 'Please Repeat / Phir Se Bolein',
    category: 'clarification',
    gloss: 'REPEAT',
    canonicalSentence: 'Could you please repeat that again?',
    translations: {
      en: 'Could you please repeat that again?',
      hi: 'क्या आप कृपया इसे दोबारा दोहरा सकते हैं?',
      kn: 'ದಯವಿಟ್ಟು ಅದನ್ನು ಇನ್ನೊಮ್ಮೆ ಪುನರಾವರ್ತಿಸಬಹುದೇ?',
      te: 'దయచేసి దాన్ని మళ్ళీ పునరావృతం చేయగలరా?'
    },
    description: 'Curved right hand turning in a forward rolling circular loop over the left hand.',
    keyFeatures: {
      hands: 'both',
      dominantHand: 'right',
      primaryMotion: 'circular',
      location: 'neutral_space'
    }
  }
];

export const DEMO_SCENARIO_PRESETS: DemoScenarioPreset[] = [
  {
    id: 'hospital_helpdesk',
    title: 'Hospital Help Desk & Emergency',
    description: 'An ISL user needs urgent navigation to the clinic emergency department.',
    context: 'At the front reception desk of a multi-specialty hospital.',
    initialRole: 'isl_signer',
    suggestedSignSequence: ['HELLO', 'HELP', 'HOSPITAL', 'WHERE', 'THANK_YOU'],
    sampleDialogue: [
      {
        speaker: 'isl',
        sign: 'HELLO',
        text: 'Hello, nice to meet you.'
      },
      {
        speaker: 'speech',
        text: 'Hello! Welcome to Apollo Hospital. How can I help you today?'
      },
      {
        speaker: 'isl',
        sign: 'HOSPITAL',
        text: 'Where is the nearest hospital or clinic?'
      },
      {
        speaker: 'speech',
        text: 'The emergency wing is immediately on the first floor to your right. Follow the red corridor line.'
      },
      {
        speaker: 'isl',
        sign: 'THANK_YOU',
        text: 'Thank you very much.'
      }
    ]
  },
  {
    id: 'college_registration',
    title: 'College Admissions & Student Services',
    description: 'A deaf student inquires about the admission counter and documents submission.',
    context: 'At the student services administrative center.',
    initialRole: 'isl_signer',
    suggestedSignSequence: ['HELLO', 'PLEASE', 'HELP', 'WHERE', 'BATHROOM'],
    sampleDialogue: [
      {
        speaker: 'isl',
        sign: 'HELLO',
        text: 'Hello, nice to meet you.'
      },
      {
        speaker: 'speech',
        text: 'Good morning! Welcome to the University Admissions Office.'
      },
      {
        speaker: 'isl',
        sign: 'HELP',
        text: 'I need help, please.'
      },
      {
        speaker: 'speech',
        text: 'Sure! For verification, please take Counter 4 on the left side.'
      },
      {
        speaker: 'isl',
        sign: 'LEFT',
        text: 'Turn to the left side.'
      },
      {
        speaker: 'speech',
        text: 'Yes, exactly on the left right past the main library.'
      }
    ]
  },
  {
    id: 'workplace_collaboration',
    title: 'Workplace Team Sync & Daily Standup',
    description: 'Collaborative discussion between hearing team members and an ISL engineer.',
    context: 'In an agile team meeting room.',
    initialRole: 'isl_signer',
    suggestedSignSequence: ['HELLO', 'YES', 'DONT_UNDERSTAND', 'REPEAT', 'THANK_YOU'],
    sampleDialogue: [
      {
        speaker: 'speech',
        text: 'Did we finish the API gateway deployment for the release today?'
      },
      {
        speaker: 'isl',
        sign: 'YES',
        text: 'Yes, that is correct.'
      },
      {
        speaker: 'speech',
        text: 'Great. Let us review the latency benchmarks next.'
      },
      {
        speaker: 'isl',
        sign: 'DONT_UNDERSTAND',
        text: 'I do not understand, could you clarify?'
      },
      {
        speaker: 'speech',
        text: 'I will put the benchmark dashboard on the shared screen right now.'
      }
    ]
  }
];
