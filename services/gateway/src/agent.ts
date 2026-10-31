import {
  CanonicalMessage,
  AgentResponseEvent,
  SupportedLanguage,
  ISL_VOCABULARY
} from '@signbridge/types';

export interface AgentInput {
  sessionId: string;
  replyToMessage: CanonicalMessage;
  conversationHistory: CanonicalMessage[];
  targetLanguage?: SupportedLanguage;
}

export class ConversationalAgent {
  private apiKey?: string;

  constructor() {
    this.apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  }

  async processMessage(input: AgentInput): Promise<AgentResponseEvent> {
    const startTime = Date.now();
    const { replyToMessage, conversationHistory } = input;

    if (replyToMessage.confidence < 0.65 && replyToMessage.modality === 'isl') {
      const altNames = replyToMessage.alternatives?.map((a: any) => a.label).join(' or ') || '';
      const clarificationText = `I noticed you may have signed "${replyToMessage.text}". Did you mean ${replyToMessage.text}${altNames ? ` or perhaps ${altNames}` : ''}?`;

      return {
        sessionId: input.sessionId,
        replyToMessageId: replyToMessage.id,
        text: clarificationText,
        translations: {
          en: clarificationText,
          hi: `मैंने देखा कि आपने शायद "${replyToMessage.text}" का संकेत किया। क्या आपका मतलब ${replyToMessage.text} था?`,
          kn: `ನೀವು "${replyToMessage.text}" ಎಂದು ಸಂಜ್ಞೆ ಮಾಡಿದ್ದೀರಿ ಎಂದು ತೋರುತ್ತದೆ. ನಿಮ್ಮ ಅರ್ಥ ${replyToMessage.text} ಎಂದೇ?`,
          te: `మీరు "${replyToMessage.text}" అని సంజ్ఞ చేసినట్లుంది. మీ ఉద్దేశ్యం ${replyToMessage.text} ఆ?`
        },
        suggestedQuickReplies: ['Yes, that is correct', 'No, let me repeat', 'Show alternatives'],
        latencyMs: Date.now() - startTime,
        model: 'SignBridge-ContextGuard-v1'
      };
    }

    if (this.apiKey) {
      try {
        const response = await this.callGeminiAPI(input);
        return {
          ...response,
          latencyMs: Date.now() - startTime
        };
      } catch (err) {
        console.warn('Gemini API call error, falling back to contextual engine:', err);
      }
    }

    return this.generateContextualLocalResponse(input, startTime);
  }

  private async callGeminiAPI(input: AgentInput): Promise<Omit<AgentResponseEvent, 'latencyMs'>> {
    const { replyToMessage, conversationHistory } = input;

    const historyStr = conversationHistory.slice(-5).map(m => `[${m.speakerRole} via ${m.modality}]: ${m.text}`).join('\n');
    const prompt = `You are SignBridge2Vision, an intelligent real-time conversational assistant bridging Indian Sign Language (ISL) users and hearing/speech users.
Previous conversation turns:
${historyStr}

Latest incoming message:
[Speaker: ${replyToMessage.speakerRole} | Modality: ${replyToMessage.modality} | Confidence: ${(replyToMessage.confidence * 100).toFixed(0)}%]: "${replyToMessage.text}"

Task:
1. Provide a direct, helpful, concise conversational response (1-2 sentences).
2. Generate accurate translations in English (en), Hindi (hi), Kannada (kn), and Telugu (te).
3. Suggest 2-3 quick action replies for the ISL user.

Format your output strictly as a JSON object:
{
  "text": "...",
  "translations": {
    "en": "...",
    "hi": "...",
    "kn": "...",
    "te": "..."
  },
  "suggestedQuickReplies": ["...", "..."]
}`;

    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${this.apiKey}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { responseMimeType: 'application/json' }
      })
    });

    if (!res.ok) {
      throw new Error(`Gemini API returned status ${res.status}`);
    }

    const data = await res.json();
    const rawText = data.candidates?.[0]?.content?.parts?.[0]?.text;
    const parsed = JSON.parse(rawText);

    return {
      sessionId: input.sessionId,
      replyToMessageId: replyToMessage.id,
      text: parsed.text || 'Understood.',
      translations: parsed.translations || { en: parsed.text },
      suggestedQuickReplies: parsed.suggestedQuickReplies || ['Thank you', 'Understood'],
      model: 'gemini-2.0-flash'
    };
  }

  private generateContextualLocalResponse(input: AgentInput, startTime: number): AgentResponseEvent {
    const { replyToMessage } = input;
    const textUpper = replyToMessage.text.toUpperCase();

    let responseText = 'Understood. How else may I assist you?';
    let translations: Record<SupportedLanguage, string> = {
      en: 'Understood. How else may I assist you?',
      hi: 'समझ गया। मैं आपकी और क्या मदद कर सकता हूँ?',
      kn: 'ಅರ್ಥವಾಯಿತು. ನಾನು ನಿಮಗೆ ಇನ್ನೇನು ಸಹಾಯ ಮಾಡಬಹುದು?',
      te: 'అర్థమైంది. నేను మీకు ఇంకా ఎలా సహాయపడగలను?'
    };
    let quickReplies: string[] = ['Thank you', 'I need help', 'Where is it?'];

    if (textUpper.includes('HELLO') || textUpper.includes('NAMASTE')) {
      responseText = 'Hello! Welcome to SignBridge. How can I assist you today?';
      translations = {
        en: 'Hello! Welcome to SignBridge. How can I assist you today?',
        hi: 'नमस्ते! साइनब्रिज में आपका स्वागत है। आज मैं आपकी क्या मदद कर सकता हूँ?',
        kn: 'ನಮಸ್ಕಾರ! ಸೈನ್‌ಬ್ರಿಡ್ಜ್‌ಗೆ ಸುಸ್ವಾಗತ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?',
        te: 'నమస్కారం! సైన్‌బ్రిడ్జ్‌కు స్వాగతం. ఈరోజు నేను మీకు ఎలా సహాయపడగలను?'
      };
      quickReplies = ['I need help', 'Where is the hospital?', 'Bathroom'];
    } else if (textUpper.includes('HELP')) {
      responseText = 'I am here to help you immediately. Please let me know what assistance you need.';
      translations = {
        en: 'I am here to help you immediately. Please let me know what assistance you need.',
        hi: 'मैं तुरंत आपकी सहायता के लिए उपस्थित हूँ। कृपया बताएं कि आपको क्या मदद चाहिए।',
        kn: 'ನಾನು ನಿಮಗೆ ತಕ್ಷಣ ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ನಿಮಗೆ ಏನು ಸಹಾಯ ಬೇಕು ಎಂದು ತಿಳಿಸಿ.',
        te: 'నేను మీకు వెంటనే సహాయం చేయడానికి ఇక్కడ ఉన్నాను. దయచేసి మీకు ఏ సహాయం కావాలో తెలియజేయండి.'
      };
      quickReplies = ['Hospital', 'Water', 'Bathroom', 'Where is the counter?'];
    } else if (textUpper.includes('HOSPITAL') || textUpper.includes('DOCTOR')) {
      responseText = 'The medical help desk and emergency clinic is on Floor 1, straight down the main corridor on your right.';
      translations = {
        en: 'The medical help desk and emergency clinic is on Floor 1, straight down the main corridor on your right.',
        hi: 'चिकित्सा सहायता केंद्र और आपातकालीन क्लिनिक पहली मंजिल पर, मुख्य गलियारे में सीधे दाईं ओर है।',
        kn: 'ವೈದ್ಯಕೀಯ ಸಹಾಯ ಕೇಂದ್ರ ಮತ್ತು ತುರ್ತು ಚಿಕಿತ್ಸಾಲಯವು ಮೊದಲ ಮಹಡಿಯಲ್ಲಿದೆ, ಮುಖ್ಯ ಕಾರಿಡಾರ್‌ನಲ್ಲಿ ನೇರವಾಗಿ ನಿಮ್ಮ ಬಲಭಾಗದಲ್ಲಿದೆ.',
        te: 'వైద్య సహాయ కేంద్రం మరియు అత్యవసర క్లినిక్ మొదటి అంతస్తులో, ప్రధాన కారిడార్‌లో నేరుగా మీ కుడి వైపున ఉంది.'
      };
      quickReplies = ['Thank you', 'Turn right', 'Where is the elevator?'];
    } else if (textUpper.includes('WATER')) {
      responseText = 'Drinking water stations are located just past Counter 2 and near the central elevators.';
      translations = {
        en: 'Drinking water stations are located just past Counter 2 and near the central elevators.',
        hi: 'पीने के पानी का स्टेशन काउंटर 2 के ठीक आगे और केंद्रीय लिफ्ट के पास स्थित है।',
        kn: 'ಕುಡಿಯುವ ನೀರಿನ ಕೇಂದ್ರಗಳು ಕೌಂಟರ್ 2 ರ ನಂತರ ಮತ್ತು ಕೇಂದ್ರೀಯ ಲಿಫ್ಟ್‌ಗಳ ಬಳಿ ಇವೆ.',
        te: 'తాగునీటి కేంద్రాలు కೌంటర్ 2 దాటి మరియు సెంట్రల్ ఎలివేటర్ల దగ్గర ఉన్నాయి.'
      };
      quickReplies = ['Thank you', 'Where is it?'];
    } else if (textUpper.includes('FOOD')) {
      responseText = 'The cafeteria and food court are located on the Ground Floor behind the reception lobby.';
      translations = {
        en: 'The cafeteria and food court are located on the Ground Floor behind the reception lobby.',
        hi: 'कैफेटेरिया और फ़ूड कोर्ट भूतल पर रिसेप्शन लॉबी के पीछे स्थित हैं।',
        kn: 'ಕ್ಯಾಫೆಟೇರಿಯಾ ಮತ್ತು ಫುಡ್ ಕೋರ್ಟ್ ನೆಲಮಹಡಿಯಲ್ಲಿ ಸ್ವಾಗತ ಲಾಬಿಯ ಹಿಂಭಾಗದಲ್ಲಿದೆ.',
        te: 'క్యాఫెటేరియా మరియు ఫుడ్ కోర్ట్ గ్రౌండ్ ఫ్లోర్‌లో రిసెప్షన్ లాబీ వెనుక ఉన్నాయి.'
      };
      quickReplies = ['Thank you', 'Where is the elevator?'];
    } else if (textUpper.includes('BATHROOM') || textUpper.includes('WASHROOM')) {
      responseText = 'Restrooms are available down the hall to the left, marked with accessible tactile signage.';
      translations = {
        en: 'Restrooms are available down the hall to the left, marked with accessible tactile signage.',
        hi: 'शौचालय बाईं ओर गलियारे में उपलब्ध हैं, जिन पर सुलभ स्पर्श संकेत लगे हैं।',
        kn: 'ಶೌಚಾಲಯಗಳು ಕಾರಿಡಾರ್‌ನ ಎಡಭಾಗದಲ್ಲಿ ಲಭ್ಯವಿದ್ದು, ಸ್ಪರ್ಶ ಸಂಕೇತಗಳನ್ನು ಹೊಂದಿವೆ.',
        te: 'శౌచాలయాలు కారిడార్‌లో ఎడమ వైపున అందుబాటులో ఉన్నాయి, స్పర్శ సంకేతాలతో గుర్తించబడ్డాయి.'
      };
      quickReplies = ['Thank you', 'Turn left'];
    } else if (textUpper.includes('WHERE')) {
      responseText = 'Please tell me which department or location you are searching for, and I will guide you step by step.';
      translations = {
        en: 'Please tell me which department or location you are searching for, and I will guide you step by step.',
        hi: 'कृपया बताएं कि आप किस विभाग या स्थान की तलाश कर रहे हैं, और मैं कदम दर कदम आपका मार्गदर्शन करूँगा।',
        kn: 'ದಯವಿಟ್ಟು ನೀವು ಯಾವ ವಿಭಾಗ ಅಥವಾ ಸ್ಥಳವನ್ನು ಹುಡುಕುತ್ತಿದ್ದೀರಿ ಎಂದು ತಿಳಿಸಿ, ನಾನು ನಿಮಗೆ ಹಂತ-ಹಂತವಾಗಿ ಮಾರ್ಗದರ್ಶನ ನೀಡುತ್ತೇನೆ.',
        te: 'దయచేసి మీరు ఏ విభాగం లేదా స్థలాన్ని వెతుకుతున్నారో చెప్పండి, నేను మీకు దశలవారీగా మార్గనిర్దేశనం చేస్తాను.'
      };
      quickReplies = ['Hospital', 'Registration', 'Washroom'];
    } else if (textUpper.includes('THANK')) {
      responseText = 'You are very welcome! Feel free to sign or ask whenever you need anything else.';
      translations = {
        en: 'You are very welcome! Feel free to sign or ask whenever you need anything else.',
        hi: 'आपका बहुत स्वागत है! जब भी आपको किसी अन्य चीज़ की आवश्यकता हो, बेझिझक संकेत करें या पूछें।',
        kn: 'ನಿಮಗೆ ಹೃತ್ಪೂರ್ವಕ ಸ್ವಾಗತ! ನಿಮಗೆ ಬೇರೆ ಯಾವುದೇ ಅಗತ್ಯವಿದ್ದಾಗ ಸಂಕೋಚವಿಲ್ಲದೆ ಸಂಜ್ಞೆ ಮಾಡಿ ಅಥವಾ ಕೇಳಿ.',
        te: 'మీకు సాదర స్వాగతం! మీకు ఇంకా ఏదైనా అవసరమైనప్పుడు నిస్సందేహంగా సంజ్ఞ చేయండి లేదా అడగండి.'
      };
      quickReplies = ['Goodbye', 'Have a nice day'];
    }

    return {
      sessionId: input.sessionId,
      replyToMessageId: replyToMessage.id,
      text: responseText,
      translations,
      suggestedQuickReplies: quickReplies,
      latencyMs: Math.max(12, Date.now() - startTime),
      model: 'SignBridge-ContextEngine-v1'
    };
  }
}

export const conversationalAgent = new ConversationalAgent();
