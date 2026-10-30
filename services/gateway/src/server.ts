import http from 'http';
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
  console.log(`===================================================`);
  console.log(`?? SignBridge2Vision Gateway running on port ${PORT}`);
  console.log(`WebSocket endpoint: ws://localhost:${PORT}/ws`);
  console.log(`Health check:       http://localhost:${PORT}/api/health`);
  console.log(`===================================================`);
});
