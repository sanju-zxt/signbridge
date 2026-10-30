/**
 * LandmarkOverlay.tsx
 * Canvas-based hand skeleton renderer for SignBridge2Vision.
 *
 * Draws MediaPipe hand landmarks + bone connections on an absolutely-positioned
 * transparent canvas that sits on top of the camera feed.
 */

import React, { useRef, useEffect } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Landmark {
  x: number; // normalised [0, 1]
  y: number; // normalised [0, 1]
  z: number; // depth (unused for 2-D rendering)
}

interface LandmarkOverlayProps {
  rightLandmarks: Landmark[] | null;
  leftLandmarks: Landmark[] | null;
  width: number;
  height: number;
  isRecognized: boolean;
  recognizedAt: number | null; // Date.now() timestamp of last recognition
}

// ---------------------------------------------------------------------------
// MediaPipe hand connection pairs (21 landmarks, 0-indexed)
// ---------------------------------------------------------------------------

const CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

const RECOGNITION_FLASH_DURATION_MS = 300;

function drawHand(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  color: string,
  width: number,
  height: number,
  glowing: boolean,
): void {
  const dotRadius = glowing ? 10 : 4;
  const shadowBlur = glowing ? 20 : 8;
  const alpha = glowing ? 1.0 : 0.8;

  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = glowing ? 2.5 : 1.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = shadowBlur;

  // --- Bone connections ---
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) {
    const lmA = landmarks[a];
    const lmB = landmarks[b];
    if (!lmA || !lmB) continue;

    ctx.moveTo(lmA.x * width, lmA.y * height);
    ctx.lineTo(lmB.x * width, lmB.y * height);
  }
  ctx.stroke();

  // --- Joint dots ---
  for (const lm of landmarks) {
    if (!lm) continue;

    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, dotRadius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = shadowBlur;
    ctx.fill();
  }

  // Reset shadow so it doesn't bleed into the next draw call
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1.0;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const LandmarkOverlay: React.FC<LandmarkOverlayProps> = ({
  rightLandmarks,
  leftLandmarks,
  width,
  height,
  isRecognized,
  recognizedAt,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear previous frame
    ctx.clearRect(0, 0, width, height);

    // Determine if we are within the recognition flash window
    const now = Date.now();
    const flashActive =
      isRecognized &&
      recognizedAt !== null &&
      now - recognizedAt <= RECOGNITION_FLASH_DURATION_MS;

    // Draw right hand (lime green)
    if (rightLandmarks && rightLandmarks.length > 0) {
      drawHand(ctx, rightLandmarks, '#00ff88', width, height, flashActive);
    }

    // Draw left hand (cyan)
    if (leftLandmarks && leftLandmarks.length > 0) {
      drawHand(ctx, leftLandmarks, '#00ccff', width, height, flashActive);
    }
  }, [rightLandmarks, leftLandmarks, width, height, isRecognized, recognizedAt]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 w-full h-full pointer-events-none object-cover"
    />
  );
};

export default LandmarkOverlay;
