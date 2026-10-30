import { WebSocket } from 'ws';

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
