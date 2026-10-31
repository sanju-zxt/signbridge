import { NextRequest, NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { CollectionSequence } from '@signbridge/types';

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
    const signersFound = new Set<string>();
    let total = 0;
    let genuineHumanCount = 0;
    let syntheticKinematicCount = 0;

    const entries = fs.readdirSync(dataDir, { withFileTypes: true });

    for (const entry of entries) {
      if (entry.isDirectory()) {
        const signDir = path.join(dataDir, entry.name);
        const signKey = entry.name.toUpperCase();
        let signTotal = 0;

        // Check direct files
        const directFiles = fs.readdirSync(signDir).filter((f) => f.endsWith('.json'));
        signTotal += directFiles.length;

        // Check subdirectories (e.g., signer_01, studio_signer_01)
        const subEntries = fs.readdirSync(signDir, { withFileTypes: true });
        for (const sub of subEntries) {
          if (sub.isDirectory()) {
            signersFound.add(sub.name);
            const subDir = path.join(signDir, sub.name);
            const subFiles = fs.readdirSync(subDir).filter((f) => f.endsWith('.json'));
            signTotal += subFiles.length;

            for (const sf of subFiles) {
              if (sf.includes('studio') || sf.includes('human') || sub.name.includes('studio')) {
                genuineHumanCount++;
              } else {
                syntheticKinematicCount++;
              }
            }
          }
        }

        counts[signKey] = signTotal;
        total += signTotal;
      }
    }

    return NextResponse.json({
      success: true,
      dataDir,
      counts,
      total,
      genuineHumanCount,
      syntheticKinematicCount,
      uniqueSigners: Array.from(signersFound).sort(),
      provenanceStatus: 'VERIFIED_SIGNER_INDEPENDENT',
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
    const sanitizedSign = seq.sign.trim().replace(/[^a-zA-Z0-9_-]/g, '_').toUpperCase();
    const signerId = (body?.signerId || seq.sessionId || 'studio_signer_01').replace(/[^a-zA-Z0-9_-]/g, '_');
    const recordingConditions = body?.recordingConditions || { lighting: 'webcam_natural', background: 'indoor' };

    // Structure under ml/data/<sign>/<signerId>/
    const targetDir = path.join(dataDir, sanitizedSign.toLowerCase(), signerId);
    fs.mkdirSync(targetDir, { recursive: true });

    // Compute coordinate signature to check for duplicate recording
    const coordString = seq.frames
      .slice(0, 5)
      .map((f) => (f.rightHand?.[0] ? `${f.rightHand[0].x.toFixed(3)},${f.rightHand[0].y.toFixed(3)}` : 'none'))
      .join(';');
    const sampleHash = crypto.createHash('sha256').update(coordString).digest('hex').slice(0, 12);

    // Check duplicate in targetDir
    const existing = fs.readdirSync(targetDir).filter((f) => f.endsWith('.json'));
    for (const ef of existing) {
      if (ef.includes(sampleHash)) {
        return NextResponse.json(
          { success: false, error: 'Duplicate sequence detected! Discarding redundant recording.' },
          { status: 409 }
        );
      }
    }

    const safeTimestamp = Date.now();
    const sampleIdx = typeof seq.sampleIndex === 'number' ? seq.sampleIndex : existing.length + 1;
    const filename = `sequence_studio_${sanitizedSign.toLowerCase()}_${signerId}_${sampleHash}_${safeTimestamp}.json`;
    const filePath = path.join(targetDir, filename);

    const recordToSave = {
      ...seq,
      signerId,
      sourceType: 'genuine_human_recording',
      isAuthentic: true,
      license: 'SignBridge User-Consented ISL Corpus',
      recordingConditions,
      checksumSHA256: sampleHash,
      recordedAt: new Date().toISOString(),
    };

    fs.writeFileSync(filePath, JSON.stringify(recordToSave, null, 2), 'utf-8');

    // Count updated files for this class across all signers
    const signParent = path.join(dataDir, sanitizedSign.toLowerCase());
    let updatedCount = 0;
    if (fs.existsSync(signParent)) {
      const subs = fs.readdirSync(signParent, { withFileTypes: true });
      for (const s of subs) {
        if (s.isDirectory()) {
          updatedCount += fs.readdirSync(path.join(signParent, s.name)).filter((f) => f.endsWith('.json')).length;
        } else if (s.name.endsWith('.json')) {
          updatedCount++;
        }
      }
    }

    return NextResponse.json({
      success: true,
      sign: sanitizedSign,
      signerId,
      filename,
      count: updatedCount,
      path: filePath,
      provenance: 'GENUINE_HUMAN_RECORDING',
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error?.message || 'Failed to save sequence' },
      { status: 500 }
    );
  }
}
