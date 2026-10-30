import { NextRequest, NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import { CollectionSequence } from '@signbridge2vision/types';

function getDataDir(): string {
  const fromRoot = path.resolve(process.cwd(), 'ml', 'data');
  if (fs.existsSync(path.resolve(process.cwd(), 'ml'))) return fromRoot;
  const fromAppsWeb = path.resolve(process.cwd(), '..', '..', 'ml', 'data');
  if (fs.existsSync(path.resolve(process.cwd(), '..', '..', 'ml'))) return fromAppsWeb;
  return fromRoot;
}

export async function GET() {
  try {
    const dataDir = getDataDir();
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    const counts: Record<string, number> = {};
    const entries = fs.readdirSync(dataDir, { withFileTypes: true });

    let total = 0;
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const signDir = path.join(dataDir, entry.name);
        const files = fs.readdirSync(signDir).filter((f) => f.endsWith('.json'));
        const signKey = entry.name.toUpperCase();
        counts[signKey] = files.length;
        total += files.length;
      }
    }

    return NextResponse.json({
      success: true,
      dataDir,
      counts,
      total,
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error?.message || 'Failed to list dataset' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const seq: CollectionSequence = body?.sequence;

    if (!seq || !seq.sign || !Array.isArray(seq.frames)) {
      return NextResponse.json(
        { success: false, error: 'Invalid sequence format. Missing sign or frames array.' },
        { status: 400 }
      );
    }

    if (seq.frames.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Sequence contains 0 frames. Minimum 1 frame required (30 recommended).' },
        { status: 400 }
      );
    }

    const dataDir = getDataDir();
    // Sanitize sign name for safe folder path
    const sanitizedSign = seq.sign.trim().replace(/[^a-zA-Z0-9_-]/g, '_').toUpperCase();
    const signDir = path.join(dataDir, sanitizedSign.toLowerCase());
    fs.mkdirSync(signDir, { recursive: true });

    const safeTimestamp = Date.now();
    const sampleIdx = typeof seq.sampleIndex === 'number' ? seq.sampleIndex : 1;
    const filename = `sequence_${safeTimestamp}_${sampleIdx}.json`;
    const filePath = path.join(signDir, filename);

    // Save full JSON
    fs.writeFileSync(filePath, JSON.stringify(seq, null, 2), 'utf-8');

    // Count updated files for this class
    const updatedCount = fs.readdirSync(signDir).filter((f) => f.endsWith('.json')).length;

    return NextResponse.json({
      success: true,
      sign: sanitizedSign,
      filename,
      count: updatedCount,
      path: filePath,
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error?.message || 'Failed to save sequence' },
      { status: 500 }
    );
  }
}
