/**
 * collectData.ts
 * -----------------------------------------------------------------------------
 * Data collection server for the SignBridge2Vision ISL gesture recognition.
 *
 * This Node.js script:
 *  1. Starts a WebSocket + Express server on port 4200.
 *  2. Serves a browser data-collection tool at http://localhost:4200.
 *  3. The browser page uses MediaPipe Hands to detect landmarks from the webcam.
 *  4. Collected sequences are sent back over WebSocket.
 *  5. Server saves them to ml/data/<sign>/sequence_<n>.json.
 *
 * Usage: npm run collect:data
 * Then open: http://localhost:4200
 *
 * Collect 30+ sequences per sign for reliable recognition.
 */

import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';
import express from 'express';
import { WebSocketServer } from 'ws';
import { ISL_VOCABULARY, CollectionSequence } from '@signbridge/types';

const PORT     = 4200;
const DATA_DIR = path.join(__dirname, '..', 'data');
const CLASS_LABELS = ISL_VOCABULARY.map(v => v.sign);

// -- Count existing sequences -----------------------------------------------
function countExisting(): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const sign of CLASS_LABELS) {
    const dir = path.join(DATA_DIR, sign.toLowerCase());
    counts[sign] = fs.existsSync(dir)
      ? fs.readdirSync(dir).filter(f => f.endsWith('.json')).length
      : 0;
  }
  return counts;
}

// -- Save sequence ----------------------------------------------------------
function saveSequence(seq: CollectionSequence): string {
  const dir = path.join(DATA_DIR, seq.sign.toLowerCase());
  fs.mkdirSync(dir, { recursive: true });
  const existing = fs.readdirSync(dir).filter(f => f.endsWith('.json')).length;
  const filename = `sequence_${String(existing + 1).padStart(4, '0')}.json`;
  const filepath = path.join(dir, filename);
  fs.writeFileSync(filepath, JSON.stringify(seq, null, 2));
  return filepath;
}

// -- Browser HTML tool -----------------------------------------------------
const HTML_TOOL = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignBridge2Vision — Data Collection Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 20px; }
  h1 { font-size: 1.4rem; color: #7dd3fc; margin-bottom: 16px; }
  .layout { display: flex; gap: 20px; width: 100%; max-width: 1100px; }
  .camera-panel { position: relative; flex: 0 0 640px; }
  video { width: 640px; height: 480px; background: #111; border-radius: 8px; transform: scaleX(-1); }
  canvas { position: absolute; top: 0; left: 0; width: 640px; height: 480px; border-radius: 8px; transform: scaleX(-1); }
  .right-panel { flex: 1; display: flex; flex-direction: column; gap: 12px; }
  .target-sign { background: #1e293b; border-radius: 10px; padding: 20px; text-align: center; }
  .target-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; }
  .target-name { font-size: 3rem; font-weight: 800; color: #7dd3fc; margin: 8px 0; }
  .target-desc { font-size: 0.85rem; color: #94a3b8; }
  .status-bar { background: #1e293b; border-radius: 8px; padding: 12px 16px; font-size: 0.9rem; }
  .status-idle   { color: #94a3b8; }
  .status-ready  { color: #34d399; }
  .status-record { color: #f97316; }
  .status-saved  { color: #22d3ee; }
  .progress-bar { height: 8px; background: #334155; border-radius: 4px; margin: 8px 0; overflow: hidden; }
  .progress-fill { height: 100%; background: #f97316; border-radius: 4px; transition: width 0.05s; }
  .counts { background: #1e293b; border-radius: 8px; padding: 12px; max-height: 240px; overflow-y: auto; }
  .count-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1e293b; font-size: 0.82rem; }
  .count-ok { color: #34d399; }
  .count-warn { color: #f97316; }
  .count-bad { color: #f43f5e; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; }
  button { padding: 10px 18px; border-radius: 8px; border: none; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: all 0.15s; }
  .btn-record { background: #f97316; color: #fff; }
  .btn-record:hover { background: #ea580c; }
  .btn-record:disabled { background: #374151; color: #6b7280; cursor: not-allowed; }
  .btn-next { background: #0ea5e9; color: #fff; }
  .btn-next:hover { background: #0284c7; }
  .btn-next:disabled { background: #374151; color: #6b7280; cursor: not-allowed; }
  .instructions { background: #1e293b; border-radius: 8px; padding: 12px 16px; font-size: 0.82rem; color: #94a3b8; line-height: 1.6; }
  .kbd { background: #334155; color: #e0e0e0; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
</style>
</head>
<body>
<h1>?? SignBridge2Vision — ISL Data Collection Tool</h1>
<div class="layout">
  <div class="camera-panel">
    <video id="video" autoplay muted playsinline></video>
    <canvas id="canvas"></canvas>
  </div>
  <div class="right-panel">
    <div class="target-sign">
      <div class="target-label">Perform This Sign</div>
      <div class="target-name" id="signName">—</div>
      <div class="target-desc" id="signDesc">Loading…</div>
    </div>
    <div class="status-bar status-idle" id="statusBar">Initialising camera…</div>
    <div>
      <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">Recording progress (30 frames)</div>
      <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
    </div>
    <div class="controls">
      <button class="btn-record" id="btnRecord" disabled>? Record (SPACE)</button>
      <button class="btn-next"   id="btnNext">Next Sign ?</button>
    </div>
    <div class="counts" id="counts">Loading counts…</div>
    <div class="instructions">
      <strong>Instructions:</strong><br>
      1. Position your hand in camera view so skeleton is visible.<br>
      2. Form the correct gesture for the sign shown above.<br>
      3. Press <kbd class="kbd">SPACE</kbd> or Record to capture 30 frames.<br>
      4. Hold the gesture steady or perform the full motion during recording.<br>
      5. Aim for <strong>30+ sequences</strong> per sign for reliable recognition.<br>
      6. Vary speed and position slightly across takes.
    </div>
  </div>
</div>

<script type="module">
import { HandLandmarker, FilesetResolver, DrawingUtils } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.js';

const video    = document.getElementById('video');
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const signName = document.getElementById('signName');
const signDesc = document.getElementById('signDesc');
const statusBar  = document.getElementById('statusBar');
const progressFill = document.getElementById('progressFill');
const btnRecord  = document.getElementById('btnRecord');
const btnNext    = document.getElementById('btnNext');
const countsDiv  = document.getElementById('counts');

const SIGNS = ${JSON.stringify(ISL_VOCABULARY.map(v => ({
  sign: v.sign,
  label: v.label,
  description: v.description,
  keyFeatures: v.keyFeatures,
})))};

const WINDOW_SIZE = 30;
let currentSignIdx = 0;
let handLandmarker = null;
let isRecording = false;
let recordedFrames = [];
let lastVideoTime = -1;
let ws = null;
let counts = {};

// WebSocket connection to server
function connectWS() {
  ws = new WebSocket('ws://localhost:${PORT}');
  ws.onopen = () => { setStatus('Connected to server. Ready to collect.', 'ready'); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'SAVED') {
      counts[msg.sign] = (counts[msg.sign] || 0) + 1;
      renderCounts();
      setStatus(\`? Sequence saved: \${msg.sign} #\${counts[msg.sign]} — path: \${msg.path}\`, 'saved');
      setTimeout(() => setStatus('Ready to record next sequence.', 'ready'), 2000);
    } else if (msg.type === 'COUNTS') {
      counts = msg.counts;
      renderCounts();
    }
  };
  ws.onerror = () => setStatus('WebSocket error — is the server running?', 'idle');
  ws.onclose = () => { setStatus('Disconnected. Reconnecting…', 'idle'); setTimeout(connectWS, 2000); };
}

// MediaPipe initialisation
async function initMediaPipe() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm'
  );
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  setStatus('Camera ready. Position hands and press RECORD.', 'ready');
  btnRecord.disabled = false;
  detectLoop();
}

// Camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
    video.srcObject = stream;
    video.addEventListener('loadeddata', initMediaPipe);
  } catch (err) {
    setStatus('Camera permission denied: ' + err.message, 'idle');
  }
}

// Detection loop
const drawUtils = { canvas: canvas, ctx: ctx };
function detectLoop() {
  if (!handLandmarker) { requestAnimationFrame(detectLoop); return; }
  if (video.readyState < 2) { requestAnimationFrame(detectLoop); return; }

  const now = performance.now();
  if (lastVideoTime === video.currentTime) { requestAnimationFrame(detectLoop); return; }
  lastVideoTime = video.currentTime;

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const result = handLandmarker.detectForVideo(video, now);

  // Draw skeleton
  if (result.landmarks && result.landmarks.length > 0) {
    const du = new DrawingUtils(ctx);
    result.landmarks.forEach((lm, i) => {
      const isRight = (result.handedness?.[i]?.[0]?.categoryName === 'Right');
      const color = isRight ? '#00ff88' : '#00ccff';
      du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color, lineWidth: 1.5 });
      du.drawLandmarks(lm, { color, fillColor: color, radius: 3 });
    });
  }

  // Recording logic
  if (isRecording) {
    const rightIdx = result.handedness?.findIndex(h => h[0]?.categoryName === 'Right');
    const leftIdx  = result.handedness?.findIndex(h => h[0]?.categoryName === 'Left');

    const rightLMs = rightIdx !== undefined && rightIdx >= 0 ? result.landmarks[rightIdx] : null;
    const leftLMs  = leftIdx  !== undefined && leftIdx  >= 0 ? result.landmarks[leftIdx]  : null;

    recordedFrames.push({
      timestamp: now,
      rightHand: rightLMs ? rightLMs.map(lm => ({ x: lm.x, y: lm.y, z: lm.z })) : null,
      leftHand:  leftLMs  ? leftLMs.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }))  : null,
      rightHandConfidence: rightIdx !== undefined && rightIdx >= 0 ? (result.handedness[rightIdx][0]?.score ?? 0) : 0,
      leftHandConfidence:  leftIdx  !== undefined && leftIdx  >= 0 ? (result.handedness[leftIdx][0]?.score ?? 0)  : 0,
    });

    const progress = Math.min(100, (recordedFrames.length / WINDOW_SIZE) * 100);
    progressFill.style.width = progress + '%';

    if (recordedFrames.length >= WINDOW_SIZE) {
      finishRecording();
    }
  }

  requestAnimationFrame(detectLoop);
}

function startRecording() {
  if (isRecording) return;
  isRecording = true;
  recordedFrames = [];
  progressFill.style.width = '0%';
  setStatus('?? Recording… hold the gesture!', 'record');
  btnRecord.disabled = true;
}

function finishRecording() {
  isRecording = false;
  btnRecord.disabled = false;
  const sign = SIGNS[currentSignIdx];
  const seq = {
    sign: sign.sign,
    sessionId: 'session_' + Date.now(),
    sampleIndex: counts[sign.sign] || 0,
    frames: recordedFrames,
    recordedAt: new Date().toISOString(),
    durationMs: recordedFrames[recordedFrames.length - 1]?.timestamp - recordedFrames[0]?.timestamp,
  };
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'SEQUENCE', data: seq }));
    setStatus('Sending sequence to server…', 'ready');
  }
  progressFill.style.width = '0%';
}

function nextSign() {
  currentSignIdx = (currentSignIdx + 1) % SIGNS.length;
  renderSign();
}

function renderSign() {
  const s = SIGNS[currentSignIdx];
  signName.textContent = s.sign;
  signDesc.textContent = s.description;
}

function renderCounts() {
  let html = '';
  for (const sign of SIGNS) {
    const n = counts[sign.sign] || 0;
    const cls = n >= 30 ? 'count-ok' : n >= 10 ? 'count-warn' : 'count-bad';
    const bar = '¦'.repeat(Math.min(n, 30)) + '¦'.repeat(Math.max(0, 30 - n));
    html += \`<div class="count-row"><span>\${sign.sign}</span><span class="\${cls}">\${String(n).padStart(2,' ')}/30  <span style="font-family:monospace;font-size:0.7rem">\${bar}</span></span></div>\`;
  }
  countsDiv.innerHTML = html;
}

function setStatus(msg, type) {
  statusBar.textContent = msg;
  statusBar.className = 'status-bar status-' + type;
}

// Keyboard shortcut
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); if (!isRecording) startRecording(); }
  if (e.code === 'ArrowRight') nextSign();
});

btnRecord.addEventListener('click', () => { if (!isRecording) startRecording(); });
btnNext.addEventListener('click', nextSign);

renderSign();
connectWS();
startCamera();
</script>
</body>
</html>`;

// -- WebSocket server -------------------------------------------------------

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocketServer({ server });

app.get('/', (_req, res) => res.type('html').send(HTML_TOOL));

wss.on('connection', (ws) => {
  // Send current counts
  ws.send(JSON.stringify({ type: 'COUNTS', counts: countExisting() }));

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      if (msg.type === 'SEQUENCE') {
        const seq: CollectionSequence = msg.data;
        const filepath = saveSequence(seq);
        const counts = countExisting();
        console.log(`? Saved: ${seq.sign} — ${seq.frames.length} frames ? ${path.basename(filepath)}`);
        ws.send(JSON.stringify({ type: 'SAVED', sign: seq.sign, path: filepath }));
        // Broadcast updated counts to all clients
        wss.clients.forEach(client => {
          if (client.readyState === 1) {
            client.send(JSON.stringify({ type: 'COUNTS', counts }));
          }
        });
      }
    } catch (err) {
      console.error('Failed to process message:', err);
    }
  });
});

// -- Start ------------------------------------------------------------------

server.listen(PORT, () => {
  console.log('------------------------------------------------------');
  console.log('SignBridge2Vision — ISL Data Collection Server');
  console.log('------------------------------------------------------');
  console.log(`Open in Chrome/Edge: http://localhost:${PORT}`);
  console.log('');
  console.log('Current sample counts:');
  const counts = countExisting();
  for (const [sign, count] of Object.entries(counts)) {
    const status = count >= 30 ? '?' : count >= 10 ? '? ' : '?';
    console.log(`  ${status} ${sign.padEnd(18)} ${count}/30`);
  }
  console.log('');
  console.log('Press Ctrl+C to stop the server.');
});
