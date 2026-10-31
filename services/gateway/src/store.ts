import {
  CanonicalMessage,
  ConversationSession,
  LatencyMetricRecord,
  LatencySummary,
  SupportedLanguage,
  SpeakerRole
} from '@signbridge/types';

export class SessionStore {
  private sessions: Map<string, ConversationSession> = new Map();
  private latencyRecords: LatencyMetricRecord[] = [];

  constructor() {
    // Initialize default main session for quick zero-config connection
    this.createSession('main-room', 'Live Interaction Studio');
  }

  createSession(id: string, title: string = 'New Conversation'): ConversationSession {
    const session: ConversationSession = {
      id,
      title,
      startedAt: Date.now(),
      participants: [],
      activeLanguage: 'en',
      messages: []
    };
    this.sessions.set(id, session);
    return session;
  }

  getSession(id: string): ConversationSession | undefined {
    return this.sessions.get(id);
  }

  getOrCreateSession(id: string): ConversationSession {
    let session = this.sessions.get(id);
    if (!session) {
      session = this.createSession(id);
    }
    return session;
  }

  addMessage(sessionId: string, message: CanonicalMessage): ConversationSession | undefined {
    const session = this.getOrCreateSession(sessionId);
    session.messages.push(message);
    return session;
  }

  getMessages(sessionId: string): CanonicalMessage[] {
    const session = this.sessions.get(sessionId);
    return session ? session.messages : [];
  }

  addParticipant(sessionId: string, participant: { id: string; name: string; role: SpeakerRole; preferredLanguage: SupportedLanguage }) {
    const session = this.getOrCreateSession(sessionId);
    const existingIndex = session.participants.findIndex((p: any) => p.id === participant.id);
    if (existingIndex >= 0) {
      session.participants[existingIndex] = participant;
    } else {
      session.participants.push(participant);
    }
    return session;
  }

  removeParticipant(sessionId: string, participantId: string) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.participants = session.participants.filter((p: any) => p.id !== participantId);
    }
  }

  recordLatency(record: LatencyMetricRecord) {
    this.latencyRecords.push(record);
    // Keep last 500 records in memory
    if (this.latencyRecords.length > 500) {
      this.latencyRecords.shift();
    }
  }

  getLatencySummary(): {
    overall: LatencySummary;
    visionSummary: LatencySummary;
    agentSummary: LatencySummary;
  } {
    const calcSummary = (values: number[]): LatencySummary => {
      if (values.length === 0) {
        return { p50: 0, p95: 0, mean: 0, min: 0, max: 0, sampleCount: 0 };
      }
      const sorted = [...values].sort((a, b) => a - b);
      const sum = sorted.reduce((acc, v) => acc + v, 0);
      return {
        p50: Number((sorted[Math.floor(sorted.length * 0.50)] || 0).toFixed(1)),
        p95: Number((sorted[Math.floor(sorted.length * 0.95)] || 0).toFixed(1)),
        mean: Number((sum / sorted.length).toFixed(1)),
        min: Number((sorted[0] || 0).toFixed(1)),
        max: Number((sorted[sorted.length - 1] || 0).toFixed(1)),
        sampleCount: sorted.length
      };
    };

    const overallLatencies = this.latencyRecords.map(r => r.totalTurnLatencyMs);
    const visionLatencies = this.latencyRecords
      .map(r => r.visionLatencyMs)
      .filter((v): v is number => typeof v === 'number');
    const agentLatencies = this.latencyRecords
      .map(r => r.agentResponseLatencyMs)
      .filter((v): v is number => typeof v === 'number');

    return {
      overall: calcSummary(overallLatencies),
      visionSummary: calcSummary(visionLatencies),
      agentSummary: calcSummary(agentLatencies)
    };
  }

  clearSession(sessionId: string) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.messages = [];
    }
  }
}

export const sessionStore = new SessionStore();
