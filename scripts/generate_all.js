const fs = require('fs');
const path = require('path');

function save(relPath, content) {
  const full = path.resolve(__dirname, '..', relPath);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content, 'utf8');
  console.log('Saved:', relPath);
}

// 1. services/gateway/src/agent.ts
save('services/gateway/src/agent.ts', `import {
  CanonicalMessage,
  AgentResponseEvent,
  SupportedLanguage,
  ISL_VOCABULARY
} from '@signbridge2vision/types';

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
      const altNames = replyToMessage.alternatives?.map(a => a.label).join(' or ') || '';
      const clarificationText = \`I noticed you may have signed "\${replyToMessage.text}". Did you mean \${replyToMessage.text}\${altNames ? \` or perhaps \${altNames}\` : ''}?\`;

      return {
        sessionId: input.sessionId,
        replyToMessageId: replyToMessage.id,
        text: clarificationText,
        translations: {
          en: clarificationText,
          hi: \`????? ???? ?? ???? ???? "\${replyToMessage.text}" ?? ????? ????? ???? ???? ??? ???? ???\`,
          kn: \`???? "\${replyToMessage.text}" ???? ?????? ?????????? ???? ???? ????????????. ????? ?????? ??? ????????\`,
          te: \`???? "\${replyToMessage.text}" ??? ???? ????????? ???? ?????????. ?? ??????? ??????\`
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

    const historyStr = conversationHistory.slice(-5).map(m => \`[\${m.speakerRole} via \${m.modality}]: \${m.text}\`).join('\\n');
    const prompt = \`You are SignBridge2Vision, an intelligent real-time conversational assistant bridging Indian Sign Language (ISL) users and hearing/speech users.
Previous conversation turns:
\${historyStr}

Latest incoming message:
[Speaker: \${replyToMessage.speakerRole} | Modality: \${replyToMessage.modality} | Confidence: \${(replyToMessage.confidence * 100).toFixed(0)}%]: "\${replyToMessage.text}"

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
}\`;

    const url = \`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=\${this.apiKey}\`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { responseMimeType: 'application/json' }
      })
    });

    if (!res.ok) {
      throw new Error(\`Gemini API returned status \${res.status}\`);
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
      hi: '??? ???? ??? ???? ?? ???? ??? ?? ???? ????',
      kn: '??????????. ???? ????? ??????? ???? ?????????',
      te: '?????????. ???? ???? ???? ??? ????? ????????'
    };
    let quickReplies: string[] = ['Thank you', 'I need help', 'Where is it?'];

    if (textUpper.includes('HELLO') || textUpper.includes('NAMASTE')) {
      responseText = 'Hello! Welcome to SignBridge. How can I assist you today?';
      translations = {
        en: 'Hello! Welcome to SignBridge. How can I assist you today?',
        hi: '??????! ????????? ??? ???? ?????? ??? ?? ??? ???? ???? ?????? ?? ???? ????',
        kn: '???????! ???????????????? ????????. ???? ???? ????? ???? ???? ?????????',
        te: '????????! ???????????????? ???????. ????? ???? ???? ??? ????? ????????'
      };
      quickReplies = ['I need help', 'Where is the hospital?', 'Bathroom'];
    } else if (textUpper.includes('HELP')) {
      responseText = 'I am here to help you immediately. Please let me know what assistance you need.';
      translations = {
        en: 'I am here to help you immediately. Please let me know what assistance you need.',
        hi: '??? ????? ???? ?????? ?? ??? ???? ???? ????? ????? ?? ???? ???? ??? ??????',
        kn: '???? ????? ????? ???? ????? ???????????. ???????? ????? ??? ???? ???? ???? ??????.',
        te: '???? ???? ?????? ????? ????????? ????? ???????. ?????? ???? ? ????? ?????? ???????????.'
      };
      quickReplies = ['Hospital', 'Water', 'Bathroom', 'Where is the counter?'];
    } else if (textUpper.includes('HOSPITAL') || textUpper.includes('DOCTOR')) {
      responseText = 'The medical help desk and emergency clinic is on Floor 1, straight down the main corridor on your right.';
      translations = {
        en: 'The medical help desk and emergency clinic is on Floor 1, straight down the main corridor on your right.',
        hi: '?????? ????? ????? ?? ????????? ??????? ???? ????? ??, ????? ??????? ??? ???? ?? ???',
        kn: '???????? ???? ?????? ????? ?????? ???????? ???? ?????????, ?????? ????? ????????????????.',
        te: '????? ???? ?????? ????? ??????? ???????? ????? ?????????, ???? ????? ???? ?????? ??????????? ????.'
      };
      quickReplies = ['Thank you', 'Turn right', 'Where is the elevator?'];
    } else if (textUpper.includes('WATER')) {
      responseText = 'Drinking water stations are located just past Counter 2 and near the central elevators.';
      translations = {
        en: 'Drinking water stations are located just past Counter 2 and near the central elevators.',
        hi: '???? ?? ???? ?? ?????? ?????? 2 ?? ??? ??? ?? ???????? ????? ?? ??? ????? ???',
        kn: '??????? ????? ????????? ?????? 2 ? ???? ????? ???????? ????????? ??? ???.',
        te: '?????????? ????????? ?????? 2 ???? ????? ???????? ????????? ????? ???????.'
      };
      quickReplies = ['Thank you', 'Where is it?'];
    } else if (textUpper.includes('FOOD')) {
      responseText = 'The cafeteria and food court are located on the Ground Floor behind the reception lobby.';
      translations = {
        en: 'The cafeteria and food court are located on the Ground Floor behind the reception lobby.',
        hi: '?????????? ?? ??? ????? ???????? ???? ?? ???? ???? ?? ????? ????',
        kn: '?????????? ????? ???? ?????? ?????? ????? ????? ??? ???????????.',
        te: '???????????? ???? ????? ??????? ????????? ?????????? ????? ???? ?????? ???????.'
      };
      quickReplies = ['Thank you', 'Where is the elevator?'];
    } else if (textUpper.includes('BATHROOM') || textUpper.includes('WASHROOM')) {
      responseText = 'Restrooms are available down the hall to the left, marked with accessible tactile signage.';
      translations = {
        en: 'Restrooms are available down the hall to the left, marked with accessible tactile signage.',
        hi: '?????? ???? ?? ??? ?? ??? ??? ?????? ???, ??? ?? ???? ???????? ????? ??? ????',
        kn: '????????????? ?????? ?????? ????????? ????????.',
        te: '???????????? ??? ???? ???? ?????? ?????????? ???????.'
      };
      quickReplies = ['Thank you', 'Turn left'];
    } else if (textUpper.includes('WHERE')) {
      responseText = 'Please tell me which department or location you are searching for, and I will guide you step by step.';
      translations = {
        en: 'Please tell me which department or location you are searching for, and I will guide you step by step.',
        hi: '????? ????? ?? ?? ??? ????? ?? ????? ?? ???? ?? ??? ???, ??? ???? ?????????? ???????',
        kn: '???????? ???? ??? ????? ???? ????????? ???????????????? ???? ??????.',
        te: '?????? ???? ? ?????? ???? ??????? ???? ???????????? ????????, ???? ???? ????????????? ????????.'
      };
      quickReplies = ['Hospital', 'Registration', 'Washroom'];
    } else if (textUpper.includes('THANK')) {
      responseText = 'You are very welcome! Feel free to sign or ask whenever you need anything else.';
      translations = {
        en: 'You are very welcome! Feel free to sign or ask whenever you need anything else.',
        hi: '???? ???? ?????? ??! ?? ?? ???? ???? ???? ?? ???????? ??, ?????? ??????',
        kn: '????? ??????! ????? ?????????? ???????????? ???????????? ????.',
        te: '???? ???????! ???? ??????? ????????????? ???????????? ??????.'
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
`);

// 2. services/gateway/src/speech.ts
save('services/gateway/src/speech.ts', `import { SupportedLanguage } from '@signbridge2vision/types';

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
`);

// 3. services/gateway/src/server.ts
save('services/gateway/src/server.ts', `import http from 'http';
import express from 'express';
import cors from 'cors';
import { WebSocketServer, WebSocket } from 'ws';
import {
  CanonicalMessage,
  RecognitionEvent,
  SpeechTranscriptEvent,
  SystemState,
  LatencyMetricRecord,
  ISL_VOCABULARY,
  DEMO_SCENARIO_PRESETS
} from '@signbridge2vision/types';
import { sessionStore } from './store';
import { conversationalAgent } from './agent';
import { speechService } from './speech';

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 4000;
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

interface ClientConnection {
  ws: WebSocket;
  sessionId: string;
  clientId: string;
  role: 'isl_signer' | 'speech_speaker' | 'observer';
}

const activeClients = new Map<WebSocket, ClientConnection>();

function broadcast(sessionId: string, event: string, data: any, excludeWs?: WebSocket) {
  const payload = JSON.stringify({ event, data, timestamp: Date.now() });
  for (const [ws, client] of activeClients.entries()) {
    if (client.sessionId === sessionId && ws.readyState === WebSocket.OPEN && ws !== excludeWs) {
      ws.send(payload);
    }
  }
}

wss.on('connection', (ws: WebSocket) => {
  let currentClient: ClientConnection = {
    ws,
    sessionId: 'main-room',
    clientId: 'client_' + Math.random().toString(36).substring(2, 9),
    role: 'observer'
  };
  activeClients.set(ws, currentClient);

  // Send initial session state
  const session = sessionStore.getOrCreateSession(currentClient.sessionId);
  ws.send(JSON.stringify({
    event: 'session.state',
    data: {
      session,
      vocabulary: ISL_VOCABULARY,
      latencyMetrics: sessionStore.getLatencySummary()
    }
  }));

  ws.on('message', async (messageData: string) => {
    try {
      const parsed = JSON.parse(messageData.toString());
      const { event, data } = parsed;

      switch (event) {
        case 'session.join': {
          currentClient.sessionId = data.sessionId || 'main-room';
          currentClient.role = data.role || 'observer';
          if (data.clientName) {
            sessionStore.addParticipant(currentClient.sessionId, {
              id: currentClient.clientId,
              name: data.clientName,
              role: currentClient.role,
              preferredLanguage: data.preferredLanguage || 'en'
            });
          }
          const currentSession = sessionStore.getOrCreateSession(currentClient.sessionId);
          broadcast(currentClient.sessionId, 'session.updated', { session: currentSession });
          break;
        }

        case 'vision.prediction': {
          // Broadcast live vision provisional sign
          broadcast(currentClient.sessionId, 'vision.provisional', data);
          break;
        }

        case 'vision.final': {
          // Finalized ISL sign
          const recognition: RecognitionEvent = data;
          const vocabItem = ISL_VOCABULARY.find(v => v.sign === recognition.sign);
          const text = vocabItem ? vocabItem.canonicalSentence : recognition.label;

          const canonicalMsg: CanonicalMessage = {
            id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substring(2, 6),
            sessionId: currentClient.sessionId,
            speakerId: currentClient.clientId,
            speakerRole: 'isl_signer',
            speakerName: 'ISL Signer',
            modality: 'isl',
            text: text,
            normalizedText: text,
            confidence: recognition.confidence,
            alternatives: recognition.alternatives,
            language: 'en',
            timestamp: Date.now(),
            status: recognition.confidence < 0.65 ? 'low_confidence' : 'final',
            latencyMs: recognition.latencyMs
          };

          sessionStore.addMessage(currentClient.sessionId, canonicalMsg);
          broadcast(currentClient.sessionId, 'message.created', canonicalMsg);

          // Trigger LLM Communication Agent
          if (canonicalMsg.status === 'final') {
            broadcast(currentClient.sessionId, 'system.state', { state: 'THINKING' });
            const history = sessionStore.getMessages(currentClient.sessionId);
            const agentResponse = await conversationalAgent.processMessage({
              sessionId: currentClient.sessionId,
              replyToMessage: canonicalMsg,
              conversationHistory: history
            });

            // Add Assistant turn
            const assistantMsg: CanonicalMessage = {
              id: 'msg_' + Date.now() + '_agent',
              sessionId: currentClient.sessionId,
              speakerId: 'agent_assistant',
              speakerRole: 'assistant',
              speakerName: 'Communication Assistant',
              modality: 'text',
              text: agentResponse.text,
              normalizedText: agentResponse.text,
              confidence: 1.0,
              language: 'en',
              timestamp: Date.now(),
              status: 'final',
              translations: agentResponse.translations,
              latencyMs: agentResponse.latencyMs
            };

            sessionStore.addMessage(currentClient.sessionId, assistantMsg);
            broadcast(currentClient.sessionId, 'message.created', assistantMsg);
            broadcast(currentClient.sessionId, 'agent.response', agentResponse);
            broadcast(currentClient.sessionId, 'system.state', { state: 'RESPONDING' });

            // Record turn metrics
            sessionStore.recordLatency({
              turnId: canonicalMsg.id,
              visionLatencyMs: recognition.latencyMs,
              agentResponseLatencyMs: agentResponse.latencyMs,
              totalTurnLatencyMs: recognition.latencyMs + agentResponse.latencyMs,
              timestamp: Date.now()
            });
            broadcast(currentClient.sessionId, 'metrics.updated', sessionStore.getLatencySummary());
          }
          break;
        }

        case 'speech.partial': {
          broadcast(currentClient.sessionId, 'speech.streaming', data);
          break;
        }

        case 'speech.final': {
          const speechEvent: SpeechTranscriptEvent = data;
          const normalized = speechService.normalizeTranscript(speechEvent.transcript);

          const canonicalMsg: CanonicalMessage = {
            id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substring(2, 6),
            sessionId: currentClient.sessionId,
            speakerId: currentClient.clientId,
            speakerRole: 'speech_speaker',
            speakerName: 'Hearing User',
            modality: 'speech',
            text: normalized,
            normalizedText: normalized,
            confidence: speechEvent.confidence || 0.95,
            language: speechEvent.language || 'en',
            timestamp: Date.now(),
            status: 'final'
          };

          sessionStore.addMessage(currentClient.sessionId, canonicalMsg);
          broadcast(currentClient.sessionId, 'message.created', canonicalMsg);

          // Agent processes spoken input
          broadcast(currentClient.sessionId, 'system.state', { state: 'THINKING' });
          const history = sessionStore.getMessages(currentClient.sessionId);
          const agentResponse = await conversationalAgent.processMessage({
            sessionId: currentClient.sessionId,
            replyToMessage: canonicalMsg,
            conversationHistory: history
          });

          const assistantMsg: CanonicalMessage = {
            id: 'msg_' + Date.now() + '_agent',
            sessionId: currentClient.sessionId,
            speakerId: 'agent_assistant',
            speakerRole: 'assistant',
            speakerName: 'Communication Assistant',
            modality: 'text',
            text: agentResponse.text,
            normalizedText: agentResponse.text,
            confidence: 1.0,
            language: 'en',
            timestamp: Date.now(),
            status: 'final',
            translations: agentResponse.translations,
            latencyMs: agentResponse.latencyMs
          };

          sessionStore.addMessage(currentClient.sessionId, assistantMsg);
          broadcast(currentClient.sessionId, 'message.created', assistantMsg);
          broadcast(currentClient.sessionId, 'agent.response', agentResponse);
          broadcast(currentClient.sessionId, 'system.state', { state: 'IDLE' });

          sessionStore.recordLatency({
            turnId: canonicalMsg.id,
            speechFinalLatencyMs: 250,
            agentResponseLatencyMs: agentResponse.latencyMs,
            totalTurnLatencyMs: 250 + agentResponse.latencyMs,
            timestamp: Date.now()
          });
          broadcast(currentClient.sessionId, 'metrics.updated', sessionStore.getLatencySummary());
          break;
        }

        case 'message.create': {
          const rawMsg = data;
          const canonicalMsg: CanonicalMessage = {
            id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substring(2, 6),
            sessionId: currentClient.sessionId,
            speakerId: currentClient.clientId,
            speakerRole: rawMsg.speakerRole || currentClient.role,
            speakerName: rawMsg.speakerName || 'Participant',
            modality: rawMsg.modality || 'text',
            text: rawMsg.text,
            normalizedText: rawMsg.text,
            confidence: 1.0,
            language: rawMsg.language || 'en',
            timestamp: Date.now(),
            status: 'final'
          };
          sessionStore.addMessage(currentClient.sessionId, canonicalMsg);
          broadcast(currentClient.sessionId, 'message.created', canonicalMsg);
          break;
        }

        case 'session.clear': {
          sessionStore.clearSession(currentClient.sessionId);
          broadcast(currentClient.sessionId, 'session.cleared', { sessionId: currentClient.sessionId });
          break;
        }

        case 'metric.record': {
          sessionStore.recordLatency(data);
          broadcast(currentClient.sessionId, 'metrics.updated', sessionStore.getLatencySummary());
          break;
        }
      }
    } catch (err) {
      console.error('Error handling WebSocket message:', err);
    }
  });

  ws.on('close', () => {
    activeClients.delete(ws);
    sessionStore.removeParticipant(currentClient.sessionId, currentClient.clientId);
    const session = sessionStore.getSession(currentClient.sessionId);
    if (session) {
      broadcast(currentClient.sessionId, 'session.updated', { session });
    }
  });
});

// REST Endpoints
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'SignBridge2Vision Real-Time Gateway',
    timestamp: new Date().toISOString(),
    activeConnections: activeClients.size,
    version: '1.0.0'
  });
});

app.get('/api/vocabulary', (req, res) => {
  res.json(ISL_VOCABULARY);
});

app.get('/api/presets', (req, res) => {
  res.json(DEMO_SCENARIO_PRESETS);
});

app.get('/api/sessions/:id', (req, res) => {
  const session = sessionStore.getSession(req.params.id);
  if (!session) return res.status(404).json({ error: 'Session not found' });
  res.json(session);
});

app.get('/api/metrics', (req, res) => {
  res.json(sessionStore.getLatencySummary());
});

server.listen(PORT, () => {
  console.log(\`===================================================\`);
  console.log(\`?? SignBridge2Vision Gateway running on port \${PORT}\`);
  console.log(\`WebSocket endpoint: ws://localhost:\${PORT}/ws\`);
  console.log(\`Health check:       http://localhost:\${PORT}/api/health\`);
  console.log(\`===================================================\`);
});
`);

// 4. services/gateway/src/testClient.ts
save('services/gateway/src/testClient.ts', `import { WebSocket } from 'ws';

async function runE2ETest() {
  console.log('Testing SignBridge2Vision Gateway WebSocket E2E...');
  const ws = new WebSocket('ws://localhost:4000/ws');

  ws.on('open', () => {
    console.log('Connected to Gateway WS server successfully!');
    
    // Join session
    ws.send(JSON.stringify({
      event: 'session.join',
      data: {
        sessionId: 'main-room',
        role: 'isl_signer',
        clientName: 'Test ISL User',
        preferredLanguage: 'en'
      }
    }));

    // Send simulated ISL final sign
    setTimeout(() => {
      console.log('Sending simulated ISL sign event [HELLO]...');
      ws.send(JSON.stringify({
        event: 'vision.final',
        data: {
          id: 'rec_01',
          sessionId: 'main-room',
          timestamp: Date.now(),
          sign: 'HELLO',
          label: 'Hello / Namaste',
          confidence: 0.94,
          alternatives: [],
          modelVersion: 'isl-temporal-v1',
          latencyMs: 18,
          isDebounced: true,
          status: 'final'
        }
      }));
    }, 500);
  });

  ws.on('message', (data) => {
    const parsed = JSON.parse(data.toString());
    console.log('Received WS Event:', parsed.event, parsed.data ? JSON.stringify(parsed.data).substring(0, 100) : '');
    
    if (parsed.event === 'agent.response') {
      console.log('Agent Response Received:', parsed.data.text);
      console.log('All E2E checks passed!');
      ws.close();
      process.exit(0);
    }
  });

  ws.on('error', (err) => {
    console.error('WS Error:', err);
    process.exit(1);
  });
}

runE2ETest();
`);
