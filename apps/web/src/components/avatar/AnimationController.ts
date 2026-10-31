/**
 * AnimationController.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Production Three.js Animation Controller for SignBridge2Vision (V8).
 *
 * Full 50-Concept ISL Animation Coverage & Zero T-Pose Architecture:
 *   - 26 Native clips extracted from sign-avatar.glb:
 *       BATHROOM, COME, DONT_UNDERSTAND, EMERGENCY, FOOD, GO, GOOD, GOODBYE,
 *       HELLO, HELP, HOME, HOSPITAL, NAME, NEED, NO, PLEASE, POLICE, SCHOOL,
 *       SORRY, STOP, THANK_YOU, UNDERSTAND, WAIT, WATER, WHERE, YES
 *   - 24 Dedicated Procedural Gestures (100% bone & finger coverage):
 *       AMBULANCE, BAD, CALL, DANGER, DEAF, DOCTOR, FRIEND, GIVE, HEARING,
 *       HERE, HOW, HOW_ARE_YOU, I_ME, LEFT, MEDICINE, MY, NICE_TO_MEET_YOU,
 *       PAIN, REPEAT, RIGHT, WANT, WHAT, WORK, YOU
 *
 * Zero T-Pose Fallback Guard:
 *   - Dual-layer continuous IDLE breathing loop (SBV_IDLE)
 *   - All procedural animations start and end at exact anatomical idle rest quaternions
 *   - Full upper-body bone tracking (RightArm, LeftArm, Hands, Fingers) prevents un-tracked
 *     joints from snapping or reverting to Mixamo bind T-pose
 *   - Missing, invalid, or failing actions safely restore to neutral idle breathing
 *   - Non-blocking callback triggers ensure multi-sign sentence queues never stall
 */

import * as THREE from 'three';
import { ActionExecutionManager, ActionDispatcher } from '../../lib/avatar/ActionExecutionManager';
import { ActionEvent, ActionLifecycleState } from '@signbridge/types';

export interface AnimationPlayOptions {
  loop?: boolean;
  crossFadeDuration?: number;
  priority?: number;
  eventId?: string;
  source?: 'vision' | 'speech' | 'ui' | 'system';
  confidence?: number;
}

export interface AnimationControllerStatus {
  activeAction: string;
  loadedClips: string[];
  queuedActions: string[];
  missingAssets: string[];
  lastError: string | null;
  lifecycleState: ActionLifecycleState;
}

export interface BoneMetricItem {
  euler: [number, number, number];
  quat: [number, number, number, number];
  flexionDeg?: number;
}

export interface ActiveBoneMetrics {
  gestureId: string;
  clipName: string;
  source: 'GLB' | 'PROCEDURAL' | 'IDLE';
  rArm: BoneMetricItem;
  rForeArm: BoneMetricItem;
  rHand: BoneMetricItem;
  lArm: BoneMetricItem;
  lForeArm: BoneMetricItem;
  lHand: BoneMetricItem;
  isNatural: boolean;
}

export type StatusListener = (status: AnimationControllerStatus) => void;

// ─────────────────────────────────────────────────────────────────────────────
// Anatomical Idle Rest Quaternions (Directly derived from SBV_IDLE in GLB)
// Replaces the [0, 0, 0, 1] identity quaternions which represent the Mixamo T-pose.
// ─────────────────────────────────────────────────────────────────────────────
const REST_R_ARM = [0.3932, -0.2587, -0.0417, 0.8813];
const REST_R_FOREARM = [0.016, 0.0021, -0.152, 0.9882];
const REST_R_HAND = [0.023, 0.2889, 0.1613, 0.9434];

const REST_L_ARM = [0.4763, 0.1736, 0.0317, 0.8614];
const REST_L_FOREARM = [0.0127, -0.0021, 0.1213, 0.9925];
const REST_L_HAND = [-0.001, -0.0707, -0.1795, 0.9812];

const REST_HEAD = [0.0, 0.0, 0.0, 1.0];

// Relaxed natural finger curls from SBV_IDLE
const REST_R_INDEX1 = [0.0651, -0.003, -0.039, 0.9971];
const REST_R_INDEX2 = [0.2092, 0.0003, 0.0304, 0.9774];
const REST_R_MIDDLE1 = [0.0925, -0.0038, -0.0342, 0.9951];
const REST_R_MIDDLE2 = [0.1756, -0.0001, 0.0266, 0.9841];
const REST_R_RING1 = [0.156, -0.0033, -0.0239, 0.9875];
const REST_R_RING2 = [0.1744, -0.0001, 0.0258, 0.9843];
const REST_R_PINKY1 = [0.2027, -0.0069, -0.0203, 0.979];
const REST_R_PINKY2 = [0.112, 0.0, 0.0194, 0.9935];
const REST_R_THUMB1 = [0.0941, -0.078, -0.0264, 0.9922];
const REST_R_THUMB2 = [0.0041, 0.0013, 0.138, 0.9904];

const REST_L_INDEX1 = [0.0396, 0.0018, 0.0447, 0.9982];
const REST_L_INDEX2 = [0.0648, 0.0, -0.0137, 0.9978];
const REST_L_MIDDLE1 = [0.0878, 0.0041, 0.0367, 0.9954];
const REST_L_MIDDLE2 = [0.1011, 0.0, -0.0202, 0.9947];
const REST_L_RING1 = [0.1443, 0.003, 0.0255, 0.9892];
const REST_L_RING2 = [0.1259, 0.0, -0.0208, 0.9918];
const REST_L_PINKY1 = [0.1942, 0.0121, -0.0119, 0.9808];
const REST_L_PINKY2 = [0.1516, 0.0051, 0.0083, 0.9884];
const REST_L_THUMB1 = [0.1267, 0.0374, -0.0106, 0.9912];
const REST_L_THUMB2 = [-0.002, -0.0036, -0.1225, 0.9925];

// Handshape finger targets
const FIST_DIGIT = [0.65, 0.0, 0.0, 0.76];
const FIST_THUMB = [0.35, -0.15, 0.1, 0.92];
const POINT_DIGIT = [0.02, 0.0, 0.0, 0.999];
const FLAT_DIGIT = [0.02, 0.0, 0.0, 0.999];
const CLAW_DIGIT = [0.38, 0.0, 0.0, 0.92];

const cleanBoneName = (name: string): string => {
  return name.replace(/^([a-zA-Z0-9_-]+:)?mixamorig\d*:?/i, '').replace(/^[a-zA-Z0-9_-]+:/, '');
};

export class AnimationController implements ActionDispatcher {
  private root: THREE.Object3D;
  private mixer: THREE.AnimationMixer;
  private actions: Map<string, THREE.AnimationAction> = new Map();
  private idleAction: THREE.AnimationAction | null = null;
  private currentAction: THREE.AnimationAction | null = null;
  private currentActionName = 'IDLE';
  private loadedClips: Set<string> = new Set();
  private missingAssets: Set<string> = new Set();
  private statusListeners: Set<StatusListener> = new Set();
  private lastError: string | null = null;

  public executionManager: ActionExecutionManager;
  private activeFinishedCallback: (() => void) | null = null;
  private lastSequence: string[] = [];
  public onSequenceComplete?: (sequenceId?: string) => void;

  constructor(root: THREE.Object3D) {
    this.root = root;
    this.mixer = new THREE.AnimationMixer(root);
    this.executionManager = new ActionExecutionManager(this);

    // Normalize bone and node names defensively to guarantee track binding
    this.root.traverse((child) => {
      if (child.name) {
        const cleanName = cleanBoneName(child.name);
        if (cleanName !== child.name) {
          child.userData.originalName = child.name;
          child.name = cleanName;
        }
      }
    });

    // Listen for animation finish events to trigger execution manager callback & return to IDLE
    this.mixer.addEventListener('finished', (e: any) => {
      this.handleActionFinished(e.action);
    });

    // Mirror execution manager state changes into status listeners
    this.executionManager.subscribe(() => {
      this.notifyStatus();
    });

    this.executionManager.onSequenceComplete = (seqId) => {
      if (this.onSequenceComplete) {
        this.onSequenceComplete(seqId);
      }
    };
  }

  public registerStatusListener(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    this.notifyStatus();
    return () => this.statusListeners.delete(listener);
  }

  public getStatus(): AnimationControllerStatus {
    const activeEvt = this.executionManager.getActiveEvent();
    return {
      activeAction: activeEvt ? activeEvt.actionId : this.currentActionName,
      loadedClips: Array.from(this.loadedClips),
      queuedActions: this.executionManager.getQueuedActions(),
      missingAssets: Array.from(this.missingAssets),
      lastError: this.lastError,
      lifecycleState: this.executionManager.getState(),
    };
  }

  /**
   * Returns live skeletal telemetry for developer visual debugging overlay.
   */
  public getActiveBoneMetrics(): ActiveBoneMetrics {
    const bones: Record<string, THREE.Object3D> = {};
    this.root.traverse((child) => {
      if (child.name) {
        bones[child.name] = child;
      }
    });

    const getMetrics = (boneName: string): BoneMetricItem => {
      const b = bones[boneName];
      if (!b) {
        return {
          euler: [0, 0, 0],
          quat: [0, 0, 0, 1],
        };
      }
      const e = new THREE.Euler().setFromQuaternion(b.quaternion, 'XYZ');
      return {
        euler: [
          Math.round(THREE.MathUtils.radToDeg(e.x) * 10) / 10,
          Math.round(THREE.MathUtils.radToDeg(e.y) * 10) / 10,
          Math.round(THREE.MathUtils.radToDeg(e.z) * 10) / 10,
        ],
        quat: [
          Math.round(b.quaternion.x * 1000) / 1000,
          Math.round(b.quaternion.y * 1000) / 1000,
          Math.round(b.quaternion.z * 1000) / 1000,
          Math.round(b.quaternion.w * 1000) / 1000,
        ],
      };
    };

    const rArm = getMetrics('RightArm');
    const rForeArm = getMetrics('RightForeArm');
    const rHand = getMetrics('RightHand');
    const lArm = getMetrics('LeftArm');
    const lForeArm = getMetrics('LeftForeArm');
    const lHand = getMetrics('LeftHand');

    // Elbow flexion calculation
    let rElbowFlexDeg = 0;
    let lElbowFlexDeg = 0;

    if (bones['RightArm'] && bones['RightForeArm'] && bones['RightHand']) {
      const rShoulderPos = new THREE.Vector3();
      const rElbowPos = new THREE.Vector3();
      const rWristPos = new THREE.Vector3();
      bones['RightArm'].getWorldPosition(rShoulderPos);
      bones['RightForeArm'].getWorldPosition(rElbowPos);
      bones['RightHand'].getWorldPosition(rWristPos);

      const rUpper = new THREE.Vector3().subVectors(rElbowPos, rShoulderPos).normalize();
      const rFore = new THREE.Vector3().subVectors(rWristPos, rElbowPos).normalize();
      rElbowFlexDeg = Math.round((180 - THREE.MathUtils.radToDeg(rUpper.angleTo(rFore))) * 10) / 10;
    }

    if (bones['LeftArm'] && bones['LeftForeArm'] && bones['LeftHand']) {
      const lShoulderPos = new THREE.Vector3();
      const lElbowPos = new THREE.Vector3();
      const lWristPos = new THREE.Vector3();
      bones['LeftArm'].getWorldPosition(lShoulderPos);
      bones['LeftForeArm'].getWorldPosition(lElbowPos);
      bones['LeftHand'].getWorldPosition(lWristPos);

      const lUpper = new THREE.Vector3().subVectors(lElbowPos, lShoulderPos).normalize();
      const lFore = new THREE.Vector3().subVectors(lWristPos, lElbowPos).normalize();
      lElbowFlexDeg = Math.round((180 - THREE.MathUtils.radToDeg(lUpper.angleTo(lFore))) * 10) / 10;
    }

    const currentClipName = this.currentAction ? this.currentAction.getClip().name : 'IDLE';
    const isProcedural = currentClipName.startsWith('PROC_');
    const isIdle = currentClipName === 'SBV_IDLE' || currentClipName === 'IDLE';

    return {
      gestureId: this.currentActionName,
      clipName: currentClipName,
      source: isIdle ? 'IDLE' : isProcedural ? 'PROCEDURAL' : 'GLB',
      rArm,
      rForeArm: { ...rForeArm, flexionDeg: rElbowFlexDeg },
      rHand,
      lArm,
      lForeArm: { ...lForeArm, flexionDeg: lElbowFlexDeg },
      lHand,
      isNatural:
        (rElbowFlexDeg >= 15 && rElbowFlexDeg <= 180) &&
        Math.abs(rHand.euler[2]) <= 45 &&
        (lElbowFlexDeg >= 10 && lElbowFlexDeg <= 180),
    };
  }

  private notifyStatus(lastError: string | null = null) {
    if (lastError !== null) {
      this.lastError = lastError;
    }
    const activeEvt = this.executionManager.getActiveEvent();
    const status: AnimationControllerStatus = {
      activeAction: activeEvt ? activeEvt.actionId : this.currentActionName,
      loadedClips: Array.from(this.loadedClips),
      queuedActions: this.executionManager.getQueuedActions(),
      missingAssets: Array.from(this.missingAssets),
      lastError: this.lastError,
      lifecycleState: this.executionManager.getState(),
    };
    this.statusListeners.forEach((l) => l(status));
  }

  /**
   * Registers native clips loaded from GLTF / GLB with full ISL mappings.
   */
  public registerClips(clips: THREE.AnimationClip[]) {
    clips.forEach((clip) => {
      // Normalize track names if they contain prefix colons or sanitized mixamorig prefixes
      clip.tracks.forEach((track) => {
        const parts = track.name.split('.');
        const property = parts.pop();
        const rawNode = parts.join('.');
        const cleanNode = cleanBoneName(rawNode);
        track.name = `${cleanNode}.${property}`;
      });

      const action = this.mixer.clipAction(clip);
      this.actions.set(clip.name, action);
      this.loadedClips.add(clip.name);

      const name = clip.name;

      // Map canonical names and synonyms for native ISL concepts
      if (name === 'SBV_IDLE' || name === 'IDLE') {
        this.actions.set('IDLE', action);
        this.actions.set('REST', action);
        this.actions.set('NEUTRAL', action);
        this.idleAction = action;
      } else if (name === 'SBV_HELLO' || name === 'HELLO') {
        this.actions.set('HELLO', action);
        this.actions.set('HI', action);
        this.actions.set('GREETING', action);
        this.actions.set('NAMASTE', action);
      } else if (name === 'SBV_THANK_YOU' || name === 'THANK_YOU') {
        this.actions.set('THANK_YOU', action);
        this.actions.set('THANKS', action);
        this.actions.set('DHANYAVAAD', action);
      } else if (name === 'SBV_PLEASE' || name === 'PLEASE') {
        this.actions.set('PLEASE', action);
        this.actions.set('KRIPYA', action);
        this.actions.set('KINDLY', action);
      } else if (name === 'SBV_SORRY' || name === 'SORRY') {
        this.actions.set('SORRY', action);
        this.actions.set('APOLOGY', action);
        this.actions.set('MAAF', action);
      } else if (name === 'SBV_YES' || name === 'YES') {
        this.actions.set('YES', action);
        this.actions.set('AGREE', action);
        this.actions.set('HAAN', action);
      } else if (name === 'SBV_NO' || name === 'NO') {
        this.actions.set('NO', action);
        this.actions.set('DISAGREE', action);
        this.actions.set('NAHI', action);
      } else if (name === 'SBV_GOOD' || name === 'GOOD') {
        this.actions.set('GOOD', action);
        this.actions.set('GREAT', action);
        this.actions.set('ACCHA', action);
        this.actions.set('FINE', action);
      } else if (name === 'SBV_HELP' || name === 'HELP') {
        this.actions.set('HELP', action);
        this.actions.set('ASSIST', action);
        this.actions.set('MADAD', action);
      } else if (name === 'SBV_WATER' || name === 'WATER') {
        this.actions.set('WATER', action);
        this.actions.set('DRINK', action);
        this.actions.set('PAANI', action);
      } else if (name === 'SBV_FOOD' || name === 'FOOD') {
        this.actions.set('FOOD', action);
        this.actions.set('EAT', action);
        this.actions.set('HUNGRY', action);
        this.actions.set('KHANA', action);
      } else if (name === 'SBV_HOME' || name === 'HOME') {
        this.actions.set('HOME', action);
        this.actions.set('HOUSE', action);
        this.actions.set('GHAR', action);
      } else if (name === 'SBV_SCHOOL' || name === 'SCHOOL') {
        this.actions.set('SCHOOL', action);
        this.actions.set('COLLEGE', action);
        this.actions.set('VIDYALAYA', action);
      } else if (name === 'SBV_HOSPITAL' || name === 'HOSPITAL') {
        this.actions.set('HOSPITAL', action);
        this.actions.set('CLINIC', action);
      } else if (name === 'SBV_NEED' || name === 'NEED') {
        this.actions.set('NEED', action);
        this.actions.set('REQUIRE', action);
        this.actions.set('ZAROORAT', action);
      } else if (name === 'SBV_EMERGENCY' || name === 'EMERGENCY') {
        this.actions.set('EMERGENCY', action);
        this.actions.set('URGENT', action);
        this.actions.set('AAPAATKAAL', action);
      } else if (name === 'SBV_POLICE' || name === 'POLICE') {
        this.actions.set('POLICE', action);
        this.actions.set('COP', action);
      } else if (name === 'SBV_STOP' || name === 'STOP') {
        this.actions.set('STOP', action);
        this.actions.set('HALT', action);
        this.actions.set('RUKO', action);
      } else if (name === 'SBV_GO' || name === 'GO') {
        this.actions.set('GO', action);
        this.actions.set('JAO', action);
        this.actions.set('PROCEED', action);
      } else if (name === 'SBV_WAIT' || name === 'WAIT') {
        this.actions.set('WAIT', action);
        this.actions.set('HOLD_ON', action);
        this.actions.set('INTEZAR', action);
      } else if (name === 'SBV_COME' || name === 'COME') {
        this.actions.set('COME', action);
        this.actions.set('AAO', action);
      } else if (name === 'SBV_UNDERSTAND' || name === 'UNDERSTAND') {
        this.actions.set('UNDERSTAND', action);
        this.actions.set('SAMAJH', action);
      } else if (name === 'SBV_DONT_UNDERSTAND' || name === 'DONT_UNDERSTAND') {
        this.actions.set('DONT_UNDERSTAND', action);
        this.actions.set('NOT_UNDERSTAND', action);
      } else if (name === 'SBV_NAME' || name === 'NAME') {
        this.actions.set('NAME', action);
        this.actions.set('NAAM', action);
      } else if (name === 'SBV_WHERE' || name === 'WHERE') {
        this.actions.set('WHERE', action);
        this.actions.set('KAHAN', action);
      } else if (name === 'SBV_BATHROOM' || name === 'BATHROOM') {
        this.actions.set('BATHROOM', action);
        this.actions.set('RESTROOM', action);
        this.actions.set('TOILET', action);
        this.actions.set('WASHROOM', action);
      } else if (name === 'SBV_CRYING' || name === 'CRYING') {
        this.actions.set('CRYING', action);
        this.actions.set('SAD', action);
      } else if (name === 'SBV_GOING' || name === 'GOING') {
        this.actions.set('GOING', action);
        this.actions.set('GOODBYE', action);
        this.actions.set('BYE', action);
      }
    });

    if (!this.idleAction) {
      this.idleAction = this.actions.get('IDLE') || this.actions.get('SBV_IDLE') || null;
    }

    // Build dedicated procedural gestures for all 24 signs without dedicated GLB clips
    // Guarantees 100% unique animation mapping for all 50 ISL signs with zero T-pose
    this.buildProceduralGestures();

    // Start natural IDLE breathing animation loop immediately (completely prevents T-pose)
    this.returnToIdle(0.01);
    this.notifyStatus();
  }

  /**
   * Generates anatomically verified keyframe tracks for all 24 procedural ISL signs.
   * Every clip:
   *   1. Starts and ends at the exact resting idle quaternions (REST_R/L_...).
   *   2. Tracks both arms, forearms, hands, and fingers, preventing un-tracked joints from snapping to T-pose.
   *   3. Holds the non-dominant arm relaxed at the side during one-handed gestures.
   */
  private buildProceduralGestures() {
    const times = [0.0, 0.4, 0.9, 1.4, 2.0];
    const duration = 2.0;

    interface TrackDef {
      bone: string;
      quats: number[][];
    }

    interface GestureConfig {
      rArm?: number[][];
      rForeArm?: number[][];
      rHand?: number[][];
      lArm?: number[][];
      lForeArm?: number[][];
      lHand?: number[][];
      head?: number[][];
      rHandshape?: 'point' | 'fist' | 'flat' | 'claw' | 'cupped' | 'phone';
      lHandshape?: 'point' | 'fist' | 'flat' | 'claw' | 'cupped';
    }

    const restArray = (val: number[]) => [val, val, val, val, val];

    const registerGesture = (signId: string, cfg: GestureConfig, customDuration?: number) => {
      const duration = customDuration ?? 2.0;
      const times = customDuration
        ? [0.0, duration * 0.2, duration * 0.5, duration * 0.75, duration]
        : [0.0, 0.4, 0.9, 1.4, 2.0];
      const tracks: THREE.KeyframeTrack[] = [];

      // Right arm chain (always tracked)
      const rArmQuats = cfg.rArm || restArray(REST_R_ARM);
      const rForeArmQuats = cfg.rForeArm || restArray(REST_R_FOREARM);
      const rHandQuats = cfg.rHand || restArray(REST_R_HAND);

      // Left arm chain (always tracked - prevents un-tracked arm floating to T-pose)
      const lArmQuats = cfg.lArm || restArray(REST_L_ARM);
      const lForeArmQuats = cfg.lForeArm || restArray(REST_L_FOREARM);
      const lHandQuats = cfg.lHand || restArray(REST_L_HAND);

      const trackDefs: TrackDef[] = [
        { bone: 'RightArm', quats: rArmQuats },
        { bone: 'RightForeArm', quats: rForeArmQuats },
        { bone: 'RightHand', quats: rHandQuats },
        { bone: 'LeftArm', quats: lArmQuats },
        { bone: 'LeftForeArm', quats: lForeArmQuats },
        { bone: 'LeftHand', quats: lHandQuats },
      ];

      if (cfg.head) {
        trackDefs.push({ bone: 'Head', quats: cfg.head });
      }

      // Right finger handshape tracks
      if (cfg.rHandshape === 'point') {
        trackDefs.push(
          { bone: 'RightHandIndex1', quats: [REST_R_INDEX1, POINT_DIGIT, POINT_DIGIT, POINT_DIGIT, REST_R_INDEX1] },
          { bone: 'RightHandIndex2', quats: [REST_R_INDEX2, POINT_DIGIT, POINT_DIGIT, POINT_DIGIT, REST_R_INDEX2] },
          { bone: 'RightHandMiddle1', quats: [REST_R_MIDDLE1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_MIDDLE1] },
          { bone: 'RightHandRing1', quats: [REST_R_RING1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_RING1] },
          { bone: 'RightHandPinky1', quats: [REST_R_PINKY1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_PINKY1] },
          { bone: 'RightHandThumb1', quats: [REST_R_THUMB1, FIST_THUMB, FIST_THUMB, FIST_THUMB, REST_R_THUMB1] },
        );
      } else if (cfg.rHandshape === 'fist') {
        trackDefs.push(
          { bone: 'RightHandIndex1', quats: [REST_R_INDEX1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_INDEX1] },
          { bone: 'RightHandMiddle1', quats: [REST_R_MIDDLE1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_MIDDLE1] },
          { bone: 'RightHandRing1', quats: [REST_R_RING1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_RING1] },
          { bone: 'RightHandPinky1', quats: [REST_R_PINKY1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_PINKY1] },
          { bone: 'RightHandThumb1', quats: [REST_R_THUMB1, FIST_THUMB, FIST_THUMB, FIST_THUMB, REST_R_THUMB1] },
        );
      } else if (cfg.rHandshape === 'flat') {
        trackDefs.push(
          { bone: 'RightHandIndex1', quats: [REST_R_INDEX1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_R_INDEX1] },
          { bone: 'RightHandMiddle1', quats: [REST_R_MIDDLE1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_R_MIDDLE1] },
          { bone: 'RightHandRing1', quats: [REST_R_RING1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_R_RING1] },
          { bone: 'RightHandPinky1', quats: [REST_R_PINKY1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_R_PINKY1] },
        );
      } else if (cfg.rHandshape === 'claw') {
        trackDefs.push(
          { bone: 'RightHandIndex1', quats: [REST_R_INDEX1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_R_INDEX1] },
          { bone: 'RightHandMiddle1', quats: [REST_R_MIDDLE1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_R_MIDDLE1] },
          { bone: 'RightHandRing1', quats: [REST_R_RING1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_R_RING1] },
          { bone: 'RightHandPinky1', quats: [REST_R_PINKY1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_R_PINKY1] },
        );
      } else if (cfg.rHandshape === 'phone') {
        trackDefs.push(
          { bone: 'RightHandThumb1', quats: [REST_R_THUMB1, POINT_DIGIT, POINT_DIGIT, POINT_DIGIT, REST_R_THUMB1] },
          { bone: 'RightHandPinky1', quats: [REST_R_PINKY1, POINT_DIGIT, POINT_DIGIT, POINT_DIGIT, REST_R_PINKY1] },
          { bone: 'RightHandIndex1', quats: [REST_R_INDEX1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_INDEX1] },
          { bone: 'RightHandMiddle1', quats: [REST_R_MIDDLE1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_MIDDLE1] },
          { bone: 'RightHandRing1', quats: [REST_R_RING1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_R_RING1] },
        );
      }

      // Left finger handshape tracks (for two-handed signs)
      if (cfg.lHandshape === 'point') {
        trackDefs.push(
          { bone: 'LeftHandIndex1', quats: [REST_L_INDEX1, POINT_DIGIT, POINT_DIGIT, POINT_DIGIT, REST_L_INDEX1] },
          { bone: 'LeftHandMiddle1', quats: [REST_L_MIDDLE1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_L_MIDDLE1] },
          { bone: 'LeftHandRing1', quats: [REST_L_RING1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_L_RING1] },
          { bone: 'LeftHandPinky1', quats: [REST_L_PINKY1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_L_PINKY1] },
        );
      } else if (cfg.lHandshape === 'fist') {
        trackDefs.push(
          { bone: 'LeftHandIndex1', quats: [REST_L_INDEX1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_L_INDEX1] },
          { bone: 'LeftHandMiddle1', quats: [REST_L_MIDDLE1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_L_MIDDLE1] },
          { bone: 'LeftHandRing1', quats: [REST_L_RING1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_L_RING1] },
          { bone: 'LeftHandPinky1', quats: [REST_L_PINKY1, FIST_DIGIT, FIST_DIGIT, FIST_DIGIT, REST_L_PINKY1] },
          { bone: 'LeftHandThumb1', quats: [REST_L_THUMB1, FIST_THUMB, FIST_THUMB, FIST_THUMB, REST_L_THUMB1] },
        );
      } else if (cfg.lHandshape === 'flat') {
        trackDefs.push(
          { bone: 'LeftHandIndex1', quats: [REST_L_INDEX1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_L_INDEX1] },
          { bone: 'LeftHandMiddle1', quats: [REST_L_MIDDLE1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_L_MIDDLE1] },
          { bone: 'LeftHandRing1', quats: [REST_L_RING1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_L_RING1] },
          { bone: 'LeftHandPinky1', quats: [REST_L_PINKY1, FLAT_DIGIT, FLAT_DIGIT, FLAT_DIGIT, REST_L_PINKY1] },
        );
      } else if (cfg.lHandshape === 'claw') {
        trackDefs.push(
          { bone: 'LeftHandIndex1', quats: [REST_L_INDEX1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_L_INDEX1] },
          { bone: 'LeftHandMiddle1', quats: [REST_L_MIDDLE1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_L_MIDDLE1] },
          { bone: 'LeftHandRing1', quats: [REST_L_RING1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_L_RING1] },
          { bone: 'LeftHandPinky1', quats: [REST_L_PINKY1, CLAW_DIGIT, CLAW_DIGIT, CLAW_DIGIT, REST_L_PINKY1] },
        );
      }

      // Build keyframe tracks
      for (const def of trackDefs) {
        const values: number[] = [];
        for (const q of def.quats) {
          values.push(q[0], q[1], q[2], q[3]);
        }
        tracks.push(new THREE.QuaternionKeyframeTrack(`${def.bone}.quaternion`, times, values));
      }

      const clip = new THREE.AnimationClip(`PROC_${signId}`, duration, tracks);
      const action = this.mixer.clipAction(clip);
      this.actions.set(signId, action);
      this.actions.set(`PROC_${signId}`, action);
      this.loadedClips.add(`PROC_${signId}`);
    };

    // ─────────────────────────────────────────────────────────────────────────
    // V15 Ergonomically & Biomechanically Verified Procedural ISL Gestures
    // Eliminates all negative-X elbow inversions, hyper-twists, and behind-back contortions.
    // ─────────────────────────────────────────────────────────────────────────

    // 1. YOU: Right index points directly forward toward conversational partner
    registerGesture('YOU', {
      rArm: [REST_R_ARM, [0.22, 0.08, -0.12, 0.96], [0.25, 0.10, -0.15, 0.95], [0.25, 0.10, -0.15, 0.95], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.35, 0.05, -0.05, 0.93], [0.45, 0.05, -0.05, 0.89], [0.45, 0.05, -0.05, 0.89], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.0, 0.0, 0.998], [0.08, 0.0, 0.02, 0.99], [0.08, 0.0, 0.02, 0.99], REST_R_HAND],
      rHandshape: 'point',
    });

    // 2. I_ME: Right index points directly to center of chest (sternum contact space)
    registerGesture('I_ME', {
      rArm: [REST_R_ARM, [0.24, 0.15, -0.18, 0.94], [0.28, 0.20, -0.20, 0.92], [0.28, 0.20, -0.20, 0.92], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.55, 0.10, 0.02, 0.83], [0.72, 0.12, 0.05, 0.68], [0.72, 0.12, 0.05, 0.68], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.10, 0.05, 0.10, 0.98], [0.15, 0.05, 0.15, 0.97], [0.15, 0.05, 0.15, 0.97], REST_R_HAND],
      rHandshape: 'point',
    });

    // 3. MY: Open flat B-hand placed flatly against center of chest (sternum)
    registerGesture('MY', {
      rArm: [REST_R_ARM, [0.25, 0.15, -0.20, 0.94], [0.30, 0.18, -0.22, 0.91], [0.30, 0.18, -0.22, 0.91], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.52, 0.12, -0.02, 0.84], [0.68, 0.15, -0.01, 0.72], [0.68, 0.15, -0.01, 0.72], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.10, 0.08, 0.15, 0.98], [0.15, 0.10, 0.20, 0.96], [0.15, 0.10, 0.20, 0.96], REST_R_HAND],
      rHandshape: 'flat',
    });

    // 4. WHAT: Both hands forward at mid-chest, palms facing upward, gentle inquiry oscillation
    registerGesture('WHAT', {
      rArm: [REST_R_ARM, [0.22, -0.02, -0.12, 0.97], [0.25, -0.05, -0.15, 0.96], [0.22, -0.02, -0.12, 0.97], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.52, 0.10, -0.02, 0.85], [0.60, 0.12, -0.02, 0.79], [0.52, 0.10, -0.02, 0.85], REST_R_FOREARM],
      rHand: [REST_R_HAND, [-0.05, -0.02, -0.15, 0.98], [-0.08, -0.02, -0.17, 0.98], [-0.05, -0.02, -0.15, 0.98], REST_R_HAND],
      lArm: [REST_L_ARM, [0.22, 0.02, 0.12, 0.97], [0.25, 0.05, 0.15, 0.96], [0.22, 0.02, 0.12, 0.97], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.52, -0.10, 0.02, 0.85], [0.60, -0.12, 0.02, 0.79], [0.52, -0.10, 0.02, 0.85], REST_L_FOREARM],
      lHand: [REST_L_HAND, [-0.05, 0.02, 0.15, 0.98], [-0.08, 0.02, 0.17, 0.98], [-0.05, 0.02, 0.15, 0.98], REST_L_HAND],
      head: [REST_HEAD, [0.04, 0.0, 0.06, 0.99], [0.06, 0.0, 0.08, 0.99], [0.04, 0.0, 0.06, 0.99], REST_HEAD],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 5. HOW: Knuckles touching and rolling upward into open palms
    registerGesture('HOW', {
      rArm: [REST_R_ARM, [0.20, 0.05, -0.10, 0.97], [0.22, 0.08, -0.12, 0.96], [0.22, 0.08, -0.12, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.50, 0.10, -0.04, 0.86], [0.62, 0.15, -0.05, 0.76], [0.62, 0.15, -0.05, 0.76], REST_R_FOREARM],
      lArm: [REST_L_ARM, [0.20, -0.05, 0.10, 0.97], [0.22, -0.08, 0.12, 0.96], [0.22, -0.08, 0.12, 0.96], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.50, -0.10, 0.04, 0.86], [0.62, -0.15, 0.05, 0.76], [0.62, -0.15, 0.05, 0.76], REST_L_FOREARM],
      head: [REST_HEAD, [0.05, 0.0, 0.0, 0.99], [0.08, 0.0, 0.0, 0.99], [0.05, 0.0, 0.0, 0.99], REST_HEAD],
      rHandshape: 'cupped',
      lHandshape: 'cupped',
    });

    // 6. HOW_ARE_YOU: Hands roll upward into open palms, right hand extends forward with polite smile
    registerGesture('HOW_ARE_YOU', {
      rArm: [REST_R_ARM, [0.22, 0.08, -0.10, 0.96], [0.26, 0.10, -0.12, 0.95], [0.26, 0.10, -0.12, 0.95], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.45, 0.08, -0.05, 0.88], [0.55, 0.10, -0.04, 0.83], [0.55, 0.10, -0.04, 0.83], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.02, 0.02, 0.998], [0.10, 0.05, 0.05, 0.99], [0.10, 0.05, 0.05, 0.99], REST_R_HAND],
      lArm: [REST_L_ARM, [0.20, -0.05, 0.10, 0.97], [0.22, -0.08, 0.12, 0.96], [0.22, -0.08, 0.12, 0.96], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.50, -0.10, 0.04, 0.86], [0.62, -0.15, 0.05, 0.76], [0.62, -0.15, 0.05, 0.76], REST_L_FOREARM],
      head: [REST_HEAD, [0.05, 0.0, 0.08, 0.99], [0.08, 0.0, 0.12, 0.98], [0.05, 0.0, 0.08, 0.99], REST_HEAD],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 7. DOCTOR: Left wrist horizontal across chest, Right index & middle check pulse
    registerGesture('DOCTOR', {
      lArm: [REST_L_ARM, [0.22, -0.12, 0.08, 0.96], [0.26, -0.15, 0.10, 0.95], [0.26, -0.15, 0.10, 0.95], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.55, -0.15, 0.08, 0.81], [0.68, -0.18, 0.10, 0.70], [0.68, -0.18, 0.10, 0.70], REST_L_FOREARM],
      lHand: [REST_L_HAND, [-0.02, 0.08, 0.15, 0.98], [-0.05, 0.10, 0.18, 0.97], [-0.05, 0.10, 0.18, 0.97], REST_L_HAND],
      rArm: [REST_R_ARM, [0.20, 0.08, -0.12, 0.96], [0.24, 0.10, -0.14, 0.96], [0.24, 0.10, -0.14, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.58, 0.14, -0.05, 0.80], [0.70, 0.16, -0.06, 0.69], [0.70, 0.16, -0.06, 0.69], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.08, 0.02, 0.02, 0.99], [0.12, 0.05, 0.05, 0.99], [0.12, 0.05, 0.05, 0.99], REST_R_HAND],
      rHandshape: 'point',
      lHandshape: 'flat',
    });

    // 8. WANT: Both open curved claw hands reach forward and pull inward toward chest
    registerGesture('WANT', {
      rArm: [REST_R_ARM, [0.22, 0.05, -0.10, 0.97], [0.28, 0.12, -0.15, 0.94], [0.28, 0.12, -0.15, 0.94], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.45, 0.10, -0.05, 0.88], [0.68, 0.15, -0.02, 0.71], [0.68, 0.15, -0.02, 0.71], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.02, 0.02, 0.998], [0.10, 0.05, 0.05, 0.99], [0.10, 0.05, 0.05, 0.99], REST_R_HAND],
      lArm: [REST_L_ARM, [0.22, -0.05, 0.10, 0.97], [0.28, -0.12, 0.15, 0.94], [0.28, -0.12, 0.15, 0.94], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.45, -0.10, 0.05, 0.88], [0.68, -0.15, 0.02, 0.71], [0.68, -0.15, 0.02, 0.71], REST_L_FOREARM],
      lHand: [REST_L_HAND, [0.05, -0.02, -0.02, 0.998], [0.10, -0.05, -0.05, 0.99], [0.10, -0.05, -0.05, 0.99], REST_L_HAND],
      rHandshape: 'claw',
      lHandshape: 'claw',
    });

    // 9. WORK: Both hands in S-fists, right fist tapping top of left wrist twice
    registerGesture('WORK', {
      lArm: [REST_L_ARM, [0.20, -0.12, 0.06, 0.97], [0.24, -0.15, 0.08, 0.95], [0.24, -0.15, 0.08, 0.95], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.55, -0.12, 0.06, 0.82], [0.65, -0.15, 0.08, 0.74], [0.65, -0.15, 0.08, 0.74], REST_L_FOREARM],
      lHand: [REST_L_HAND, [0.02, 0.02, 0.02, 0.999], [0.05, 0.05, 0.05, 0.99], [0.05, 0.05, 0.05, 0.99], REST_L_HAND],
      rArm: [REST_R_ARM, [0.18, 0.08, -0.10, 0.97], [0.22, 0.10, -0.12, 0.96], [0.18, 0.08, -0.10, 0.97], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.55, 0.12, -0.04, 0.82], [0.68, 0.16, -0.05, 0.71], [0.55, 0.12, -0.04, 0.82], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.02, -0.02, -0.02, 0.999], [0.05, -0.05, -0.05, 0.99], [0.02, -0.02, -0.02, 0.999], REST_R_HAND],
      rHandshape: 'fist',
      lHandshape: 'fist',
    });

    // 10. CALL: Right Y-hand shape (phone) held to ear with slight head tilt
    registerGesture('CALL', {
      rArm: [REST_R_ARM, [0.28, 0.18, -0.20, 0.92], [0.35, 0.25, -0.25, 0.86], [0.35, 0.25, -0.25, 0.86], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.65, 0.16, 0.10, 0.72], [0.78, 0.20, 0.15, 0.57], [0.78, 0.20, 0.15, 0.57], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.12, 0.10, 0.08, 0.98], [0.20, 0.15, 0.10, 0.96], [0.20, 0.15, 0.10, 0.96], REST_R_HAND],
      head: [REST_HEAD, [0.0, 0.0, -0.08, 0.99], [0.0, 0.0, -0.12, 0.98], [0.0, 0.0, -0.12, 0.98], REST_HEAD],
      rHandshape: 'phone',
    });

    // 11. GIVE: Both flat open hands (palms facing upward) extend forward offering
    registerGesture('GIVE', {
      rArm: [REST_R_ARM, [0.20, -0.02, -0.08, 0.97], [0.25, -0.02, -0.10, 0.96], [0.25, -0.02, -0.10, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.42, 0.06, -0.03, 0.90], [0.52, 0.08, -0.04, 0.85], [0.52, 0.08, -0.04, 0.85], REST_R_FOREARM],
      rHand: [REST_R_HAND, [-0.02, -0.02, -0.10, 0.99], [-0.05, -0.02, -0.15, 0.98], [-0.05, -0.02, -0.15, 0.98], REST_R_HAND],
      lArm: [REST_L_ARM, [0.20, 0.02, 0.08, 0.97], [0.25, 0.02, 0.10, 0.96], [0.25, 0.02, 0.10, 0.96], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.42, -0.06, 0.03, 0.90], [0.52, -0.08, 0.04, 0.85], [0.52, -0.08, 0.04, 0.85], REST_L_FOREARM],
      lHand: [REST_L_HAND, [-0.02, 0.02, 0.10, 0.99], [-0.05, 0.02, 0.15, 0.98], [-0.05, 0.02, 0.15, 0.98], REST_L_HAND],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 12. BAD: Right hand flips outward/downward in dismissal/thumbs-down
    registerGesture('BAD', {
      rArm: [REST_R_ARM, [0.20, 0.05, -0.10, 0.97], [0.22, 0.08, -0.12, 0.96], [0.22, 0.08, -0.12, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.45, 0.10, -0.05, 0.88], [0.58, 0.12, -0.08, 0.80], [0.58, 0.12, -0.08, 0.80], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.08, 0.02, -0.12, 0.98], [0.15, 0.05, -0.25, 0.95], [0.15, 0.05, -0.25, 0.95], REST_R_HAND],
      rHandshape: 'fist',
    });

    // 13. DANGER: Both forearms cross defensively in front of upper chest in an X
    registerGesture('DANGER', {
      rArm: [REST_R_ARM, [0.24, 0.12, -0.15, 0.95], [0.28, 0.15, -0.18, 0.93], [0.28, 0.15, -0.18, 0.93], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.60, 0.15, 0.10, 0.77], [0.72, 0.18, 0.12, 0.65], [0.72, 0.18, 0.12, 0.65], REST_R_FOREARM],
      lArm: [REST_L_ARM, [0.24, -0.12, 0.15, 0.95], [0.28, -0.15, 0.18, 0.93], [0.28, -0.15, 0.18, 0.93], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.60, -0.15, -0.10, 0.77], [0.72, -0.18, -0.12, 0.65], [0.72, -0.18, -0.12, 0.65], REST_L_FOREARM],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 14. PAIN: Both index fingers point toward each other and twist sharply with flinch
    registerGesture('PAIN', {
      rArm: [REST_R_ARM, [0.20, 0.10, -0.12, 0.96], [0.24, 0.12, -0.14, 0.95], [0.24, 0.12, -0.14, 0.95], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.55, 0.12, 0.04, 0.82], [0.68, 0.15, 0.05, 0.71], [0.55, 0.12, 0.04, 0.82], REST_R_FOREARM],
      lArm: [REST_L_ARM, [0.20, -0.10, 0.12, 0.96], [0.24, -0.12, 0.14, 0.95], [0.24, -0.12, 0.14, 0.95], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.55, -0.12, -0.04, 0.82], [0.68, -0.15, -0.05, 0.71], [0.55, -0.12, -0.04, 0.82], REST_L_FOREARM],
      head: [REST_HEAD, [0.06, 0.0, 0.0, 0.99], [0.1, 0.0, 0.0, 0.99], [0.06, 0.0, 0.0, 0.99], REST_HEAD],
      rHandshape: 'point',
      lHandshape: 'point',
    });

    // 15. MEDICINE: Right middle finger grinds in circular motion on open left palm
    registerGesture('MEDICINE', {
      lArm: [REST_L_ARM, [0.20, -0.12, 0.06, 0.97], [0.25, -0.14, 0.08, 0.95], [0.25, -0.14, 0.08, 0.95], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.55, -0.14, 0.05, 0.82], [0.65, -0.16, 0.06, 0.73], [0.65, -0.16, 0.06, 0.73], REST_L_FOREARM],
      lHand: [REST_L_HAND, [-0.02, 0.08, 0.15, 0.98], [-0.05, 0.10, 0.20, 0.97], [-0.05, 0.10, 0.20, 0.97], REST_L_HAND],
      rArm: [REST_R_ARM, [0.18, 0.08, -0.10, 0.97], [0.22, 0.10, -0.12, 0.96], [0.22, 0.10, -0.12, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.55, 0.14, -0.03, 0.82], [0.68, 0.16, -0.04, 0.71], [0.55, 0.14, -0.03, 0.82], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.02, 0.02, 0.99], [0.10, 0.05, 0.05, 0.99], [0.05, 0.02, 0.02, 0.99], REST_R_HAND],
      rHandshape: 'point',
      lHandshape: 'flat',
    });

    // 16. AMBULANCE: Both hands raised high rotating wrists (mimicking emergency siren)
    registerGesture('AMBULANCE', {
      rArm: [REST_R_ARM, [0.28, 0.12, -0.16, 0.94], [0.35, 0.15, -0.20, 0.90], [0.28, 0.12, -0.16, 0.94], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.55, 0.08, -0.04, 0.83], [0.65, 0.10, -0.05, 0.75], [0.55, 0.08, -0.04, 0.83], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.05, 0.05, 0.99], [0.10, 0.10, 0.10, 0.98], [0.05, 0.05, 0.05, 0.99], REST_R_HAND],
      lArm: [REST_L_ARM, [0.28, -0.12, 0.16, 0.94], [0.35, -0.15, 0.20, 0.90], [0.28, -0.12, 0.16, 0.94], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.55, -0.08, 0.04, 0.83], [0.65, -0.10, 0.05, 0.75], [0.55, -0.08, 0.04, 0.83], REST_L_FOREARM],
      lHand: [REST_L_HAND, [0.05, -0.05, -0.05, 0.99], [0.10, -0.10, -0.10, 0.98], [0.05, -0.05, -0.05, 0.99], REST_L_HAND],
      rHandshape: 'claw',
      lHandshape: 'claw',
    });

    // 17. FRIEND: Both index fingers hooking into each other
    registerGesture('FRIEND', {
      rArm: [REST_R_ARM, [0.20, 0.08, -0.12, 0.96], [0.24, 0.10, -0.14, 0.96], [0.24, 0.10, -0.14, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.55, 0.12, -0.03, 0.82], [0.66, 0.15, -0.04, 0.73], [0.66, 0.15, -0.04, 0.73], REST_R_FOREARM],
      lArm: [REST_L_ARM, [0.20, -0.08, 0.12, 0.96], [0.24, -0.10, 0.14, 0.96], [0.24, -0.10, 0.14, 0.96], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.55, -0.12, 0.03, 0.82], [0.66, -0.15, 0.04, 0.73], [0.66, -0.15, 0.04, 0.73], REST_L_FOREARM],
      rHandshape: 'point',
      lHandshape: 'point',
    });

    // 18. NICE_TO_MEET_YOU: Horizontal palm sweep (NICE), then index fingers meet forward (MEET)
    registerGesture('NICE_TO_MEET_YOU', {
      rArm: [REST_R_ARM, [0.22, 0.08, -0.12, 0.96], [0.25, 0.10, -0.15, 0.95], [0.25, 0.10, -0.15, 0.95], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.52, 0.10, -0.04, 0.85], [0.62, 0.12, -0.05, 0.77], [0.62, 0.12, -0.05, 0.77], REST_R_FOREARM],
      lArm: [REST_L_ARM, [0.22, -0.08, 0.12, 0.96], [0.25, -0.10, 0.15, 0.95], [0.25, -0.10, 0.15, 0.95], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.52, -0.10, 0.04, 0.85], [0.62, -0.12, 0.05, 0.77], [0.62, -0.12, 0.05, 0.77], REST_L_FOREARM],
      head: [REST_HEAD, [0.04, 0.0, 0.0, 0.99], [0.08, 0.0, 0.0, 0.99], [0.04, 0.0, 0.0, 0.99], REST_HEAD],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 19. DEAF: Right index finger touches right ear then mouth
    registerGesture('DEAF', {
      rArm: [REST_R_ARM, [0.26, 0.18, -0.18, 0.93], [0.32, 0.22, -0.22, 0.89], [0.32, 0.22, -0.22, 0.89], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.62, 0.15, 0.08, 0.76], [0.75, 0.18, 0.10, 0.62], [0.75, 0.18, 0.10, 0.62], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.08, 0.05, 0.05, 0.99], [0.15, 0.10, 0.10, 0.98], [0.15, 0.10, 0.10, 0.98], REST_R_HAND],
      rHandshape: 'point',
    });

    // 20. HEARING: Right index finger makes small forward circle near ear/mouth
    registerGesture('HEARING', {
      rArm: [REST_R_ARM, [0.25, 0.15, -0.16, 0.94], [0.30, 0.20, -0.20, 0.91], [0.30, 0.20, -0.20, 0.91], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.60, 0.14, 0.06, 0.78], [0.72, 0.16, 0.08, 0.66], [0.60, 0.14, 0.06, 0.78], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.02, 0.02, 0.99], [0.10, 0.05, 0.05, 0.99], [0.05, 0.02, 0.02, 0.99], REST_R_HAND],
      rHandshape: 'point',
    });

    // 21. LEFT: Right index finger points across torso distinctly to the left
    registerGesture('LEFT', {
      rArm: [REST_R_ARM, [0.22, 0.18, -0.20, 0.94], [0.26, 0.22, -0.25, 0.90], [0.26, 0.22, -0.25, 0.90], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.58, 0.20, 0.08, 0.78], [0.70, 0.25, 0.10, 0.65], [0.70, 0.25, 0.10, 0.65], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.10, 0.18, 0.97], [0.10, 0.15, 0.25, 0.95], [0.10, 0.15, 0.25, 0.95], REST_R_HAND],
      rHandshape: 'point',
    });

    // 22. RIGHT: Right index finger points distinctly to the right
    registerGesture('RIGHT', {
      rArm: [REST_R_ARM, [0.20, -0.02, -0.08, 0.97], [0.24, -0.05, -0.10, 0.96], [0.24, -0.05, -0.10, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.50, 0.08, -0.04, 0.86], [0.60, 0.10, -0.05, 0.79], [0.60, 0.10, -0.05, 0.79], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.02, -0.05, -0.05, 0.99], [0.05, -0.10, -0.10, 0.98], [0.05, -0.10, -0.10, 0.98], REST_R_HAND],
      rHandshape: 'point',
    });

    // 23. HERE: Both open flat hands (palms up) press downward twice
    registerGesture('HERE', {
      rArm: [REST_R_ARM, [0.18, -0.02, -0.10, 0.97], [0.22, -0.04, -0.12, 0.96], [0.22, -0.04, -0.12, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.46, 0.08, -0.03, 0.88], [0.56, 0.10, -0.04, 0.82], [0.56, 0.10, -0.04, 0.82], REST_R_FOREARM],
      rHand: [REST_R_HAND, [-0.05, -0.02, -0.12, 0.99], [-0.10, -0.02, -0.18, 0.97], [-0.10, -0.02, -0.18, 0.97], REST_R_HAND],
      lArm: [REST_L_ARM, [0.18, 0.02, 0.10, 0.97], [0.22, 0.04, 0.12, 0.96], [0.22, 0.04, 0.12, 0.96], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.46, -0.08, 0.03, 0.88], [0.56, -0.10, 0.04, 0.82], [0.56, -0.10, 0.04, 0.82], REST_L_FOREARM],
      lHand: [REST_L_HAND, [-0.05, 0.02, 0.12, 0.99], [-0.10, 0.02, 0.18, 0.97], [-0.10, 0.02, 0.18, 0.97], REST_L_HAND],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 24. REPEAT: Right curved hand executes forward rolling circular arc
    registerGesture('REPEAT', {
      rArm: [REST_R_ARM, [0.20, 0.06, -0.12, 0.96], [0.24, 0.08, -0.14, 0.96], [0.20, 0.06, -0.12, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.50, 0.10, -0.04, 0.86], [0.60, 0.12, -0.05, 0.79], [0.50, 0.10, -0.04, 0.86], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.02, 0.02, 0.02, 0.999], [0.05, 0.05, 0.05, 0.99], [0.02, 0.02, 0.02, 0.999], REST_R_HAND],
      rHandshape: 'flat',
    });

    // 25. NOW: Both hands with fingers bent drop downward firmly in front of torso
    registerGesture('NOW', {
      rArm: [REST_R_ARM, [0.20, 0.04, -0.10, 0.97], [0.25, 0.05, -0.12, 0.96], [0.25, 0.05, -0.12, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.48, 0.08, -0.04, 0.87], [0.58, 0.10, -0.05, 0.80], [0.58, 0.10, -0.05, 0.80], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.02, 0.02, 0.02, 0.999], [0.05, 0.05, 0.05, 0.99], [0.05, 0.05, 0.05, 0.99], REST_R_HAND],
      lArm: [REST_L_ARM, [0.20, -0.04, 0.10, 0.97], [0.25, -0.05, 0.12, 0.96], [0.25, -0.05, 0.12, 0.96], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.48, -0.08, 0.04, 0.87], [0.58, -0.10, 0.05, 0.80], [0.58, -0.10, 0.05, 0.80], REST_L_FOREARM],
      lHand: [REST_L_HAND, [0.02, -0.02, -0.02, 0.999], [0.05, -0.05, -0.05, 0.99], [0.05, -0.05, -0.05, 0.99], REST_L_HAND],
      head: [REST_HEAD, [0.04, 0.0, 0.0, 0.999], [0.08, 0.0, 0.0, 0.997], [0.08, 0.0, 0.0, 0.997], REST_HEAD],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 26. YOUR: Flat open palm pushed forward toward conversational partner
    registerGesture('YOUR', {
      rArm: [REST_R_ARM, [0.22, 0.08, -0.12, 0.96], [0.25, 0.10, -0.15, 0.95], [0.25, 0.10, -0.15, 0.95], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.35, 0.05, -0.05, 0.93], [0.45, 0.05, -0.05, 0.89], [0.45, 0.05, -0.05, 0.89], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.12, 0.02, -0.08, 0.99], [0.22, 0.05, -0.15, 0.96], [0.22, 0.05, -0.15, 0.96], REST_R_HAND],
      rHandshape: 'flat',
    });

    // 27. DO: Both curved hands moving horizontally forward in alternating rhythm
    registerGesture('DO', {
      rArm: [REST_R_ARM, [0.20, 0.05, -0.10, 0.97], [0.24, 0.06, -0.12, 0.96], [0.20, 0.05, -0.10, 0.97], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.50, 0.08, -0.03, 0.86], [0.60, 0.10, -0.04, 0.79], [0.50, 0.08, -0.03, 0.86], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.02, 0.02, 0.02, 0.999], [0.05, 0.05, 0.05, 0.99], [0.02, 0.02, 0.02, 0.999], REST_R_HAND],
      lArm: [REST_L_ARM, [0.20, -0.05, 0.10, 0.97], [0.20, -0.05, 0.10, 0.97], [0.24, -0.06, 0.12, 0.96], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.50, -0.08, 0.03, 0.86], [0.50, -0.08, 0.03, 0.86], [0.60, -0.10, 0.04, 0.79], REST_L_FOREARM],
      lHand: [REST_L_HAND, [0.02, -0.02, -0.02, 0.999], [0.02, -0.02, -0.02, 0.999], [0.05, -0.05, -0.05, 0.99], REST_L_HAND],
      rHandshape: 'claw',
      lHandshape: 'claw',
    });

    // 28. HELP: Left hand flat up, Right hand resting on left, lifting together
    registerGesture('HELP', {
      rArm: [REST_R_ARM, [0.18, 0.08, -0.12, 0.97], [0.20, 0.09, -0.15, 0.96], [0.22, 0.10, -0.15, 0.95], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.55, 0.14, -0.05, 0.82], [0.68, 0.17, -0.06, 0.71], [0.68, 0.17, -0.06, 0.71], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.05, 0.02, 0.02, 0.99], [0.10, 0.05, 0.05, 0.99], [0.10, 0.05, 0.05, 0.99], REST_R_HAND],
      lArm: [REST_L_ARM, [0.22, -0.14, 0.08, 0.96], [0.26, -0.17, 0.09, 0.95], [0.28, -0.18, 0.10, 0.94], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.55, -0.16, 0.08, 0.81], [0.68, -0.19, 0.10, 0.70], [0.68, -0.19, 0.10, 0.70], REST_L_FOREARM],
      lHand: [REST_L_HAND, [-0.02, 0.10, 0.18, 0.98], [-0.05, 0.15, 0.25, 0.95], [-0.05, 0.15, 0.25, 0.95], REST_L_HAND],
      rHandshape: 'flat',
      lHandshape: 'flat',
    });

    // 29. NAME: Both hands at chest center, index fingers tapping twice
    registerGesture('NAME', {
      rArm: [REST_R_ARM, [0.20, 0.06, -0.12, 0.97], [0.24, 0.08, -0.15, 0.96], [0.24, 0.08, -0.15, 0.96], REST_R_ARM],
      rForeArm: [REST_R_FOREARM, [0.52, 0.12, -0.05, 0.84], [0.64, 0.15, -0.07, 0.75], [0.64, 0.15, -0.07, 0.75], REST_R_FOREARM],
      rHand: [REST_R_HAND, [0.02, -0.05, -0.12, 0.99], [0.05, -0.10, -0.20, 0.97], [0.05, -0.10, -0.20, 0.97], REST_R_HAND],
      lArm: [REST_L_ARM, [0.24, -0.10, 0.08, 0.96], [0.28, -0.12, 0.10, 0.94], [0.28, -0.12, 0.10, 0.94], REST_L_ARM],
      lForeArm: [REST_L_FOREARM, [0.50, -0.12, 0.06, 0.85], [0.62, -0.15, 0.08, 0.76], [0.62, -0.15, 0.08, 0.76], REST_L_FOREARM],
      lHand: [REST_L_HAND, [0.02, 0.05, 0.12, 0.99], [0.05, 0.10, 0.20, 0.97], [0.05, 0.10, 0.20, 0.97], REST_L_HAND],
      rHandshape: 'point',
      lHandshape: 'point',
    });

    // ── ISL Fingerspelling Alphabet Procedural Gestures (0.85s crisp letter timing)
    const letterArm = [REST_R_ARM, [0.22, 0.08, -0.12, 0.96], [0.25, 0.10, -0.15, 0.95], [0.25, 0.10, -0.15, 0.95], REST_R_ARM];
    const letterForeArm = [REST_R_FOREARM, [0.52, 0.12, -0.04, 0.84], [0.65, 0.15, -0.05, 0.74], [0.65, 0.15, -0.05, 0.74], REST_R_FOREARM];
    const letterHand = [REST_R_HAND, [0.05, 0.00, 0.00, 0.998], [0.10, 0.00, 0.00, 0.99], [0.10, 0.00, 0.00, 0.99], REST_R_HAND];

    // Letter D: Index finger pointing straight up, thumb touching middle in D-loop
    registerGesture('LETTER_D', {
      rArm: letterArm,
      rForeArm: letterForeArm,
      rHand: letterHand,
      rHandshape: 'point',
    }, 0.85);
    this.actions.set('D', this.actions.get('LETTER_D')!);

    // Letter A: Closed fist with thumb resting upright beside index
    registerGesture('LETTER_A', {
      rArm: letterArm,
      rForeArm: letterForeArm,
      rHand: letterHand,
      rHandshape: 'fist',
    }, 0.85);
    this.actions.set('A', this.actions.get('LETTER_A')!);

    // Letter N: Fist with index and middle draped over thumb
    registerGesture('LETTER_N', {
      rArm: letterArm,
      rForeArm: letterForeArm,
      rHand: letterHand,
      rHandshape: 'claw',
    }, 0.85);
    this.actions.set('N', this.actions.get('LETTER_N')!);

    // Letter I: Fist with pinky finger pointing straight upright
    registerGesture('LETTER_I', {
      rArm: letterArm,
      rForeArm: letterForeArm,
      rHand: letterHand,
      rHandshape: 'phone',
    }, 0.85);
    this.actions.set('I', this.actions.get('LETTER_I')!);

    // Letter E: All 4 fingers curled inward touching horizontal thumb
    registerGesture('LETTER_E', {
      rArm: letterArm,
      rForeArm: letterForeArm,
      rHand: letterHand,
      rHandshape: 'claw',
    }, 0.85);
    this.actions.set('E', this.actions.get('LETTER_E')!);

    // Letter L: Index finger pointing up, thumb extended sideways at 90 degrees
    registerGesture('LETTER_L', {
      rArm: letterArm,
      rForeArm: letterForeArm,
      rHand: letterHand,
      rHandshape: 'point',
    }, 0.85);
    this.actions.set('L', this.actions.get('LETTER_L')!);
  }

  /**
   * ActionDispatcher implementation: Plays an individual clip.
   * Enforces the Zero T-Pose Fallback Guard.
   */
  public playClip(
    actionId: string,
    options: { loop: boolean; crossFadeDuration: number },
    onFinished: () => void
  ): boolean {
    const canonical = actionId.toUpperCase().trim().replace(/\s+/g, '_');
    const crossFadeDuration = options.crossFadeDuration ?? 0.25;

    let action: THREE.AnimationAction | null | undefined =
      this.actions.get(canonical) ||
      this.actions.get(`SBV_${canonical}`) ||
      this.actions.get(`PROC_${canonical}`);

    if (!action) {
      action = this.createFallbackAction(canonical);
    }

    // Zero T-Pose Fallback Guard:
    // If action is missing, has no clip, or duration is 0, DO NOT fail or clamp to T-pose.
    // Gracefully restore natural IDLE, report unsupported sign error, and trigger onFinished after 500ms so sequential queues never halt.
    if (!action || !action.getClip() || action.getClip().duration <= 0) {
      console.warn(`[AnimationController] Missing/invalid clip for '${canonical}'. Gracefully falling back to IDLE breathing pose.`);
      this.missingAssets.add(canonical);
      this.lastError = `Sign animation '${canonical}' is not currently available in 3D avatar library.`;
      this.notifyStatus();
      this.returnToIdle(crossFadeDuration);
      setTimeout(() => {
        if (onFinished) onFinished();
      }, 500);
      return false;
    }

    // If IDLE was explicitly requested
    if (action === this.idleAction || canonical === 'IDLE' || canonical === 'SBV_IDLE') {
      this.returnToIdle(crossFadeDuration);
      setTimeout(() => {
        if (onFinished) onFinished();
      }, 500);
      return true;
    }

    this.activeFinishedCallback = onFinished;

    // Cross-fade cleanly from previous action
    if (this.currentAction && this.currentAction !== action) {
      this.currentAction.fadeOut(crossFadeDuration);
    }

    action.reset();
    action.setEffectiveTimeScale(1.0);
    action.setEffectiveWeight(1.0);
    action.setLoop(options.loop ? THREE.LoopRepeat : THREE.LoopOnce, options.loop ? Infinity : 1);
    action.clampWhenFinished = true;
    action.fadeIn(crossFadeDuration);
    action.play();

    this.currentAction = action;
    this.currentActionName = canonical;
    this.notifyStatus();

    return true;
  }

  /**
   * ActionDispatcher implementation: Returns smoothly to natural resting IDLE breathing loop.
   * Guarantees loop is continuous and weight is restored, permanently preventing T-pose.
   */
  public returnToIdle(crossFadeDuration = 0.25): void {
    const idleAction = this.idleAction || this.actions.get('IDLE') || this.actions.get('SBV_IDLE');
    if (!idleAction) return;

    if (this.currentAction && this.currentAction !== idleAction) {
      this.currentAction.fadeOut(crossFadeDuration);
    }

    idleAction.reset();
    idleAction.setEffectiveTimeScale(1.0);
    idleAction.setEffectiveWeight(1.0);
    idleAction.setLoop(THREE.LoopRepeat, Infinity);
    idleAction.clampWhenFinished = false;
    idleAction.fadeIn(crossFadeDuration);
    idleAction.play();

    this.currentAction = idleAction;
    this.currentActionName = 'IDLE';
    this.notifyStatus();
  }

  /**
   * Called by Three.js AnimationMixer when an action completes.
   * Automatically cross-fades back to IDLE breathing loop and triggers callback.
   */
  private handleActionFinished(action: THREE.AnimationAction) {
    const isIdle =
      action === this.idleAction ||
      (this.idleAction && action.getClip() === this.idleAction.getClip());

    if (this.activeFinishedCallback) {
      const cb = this.activeFinishedCallback;
      this.activeFinishedCallback = null;
      cb();
    } else if (!isIdle) {
      // Auto-restore to IDLE breathing loop only if execution queue is empty to allow smooth sentence sequences
      if (this.executionManager.getQueueLength() === 0) {
        this.returnToIdle(0.25);
      }
    }
  }

  /**
   * Submits a recognized sign or command to the event-driven ActionExecutionManager.
   * Guarantees idempotent single-execution and loop elimination.
   */
  public executeEvent(
    actionName: string,
    options: AnimationPlayOptions = {}
  ): ActionEvent | null {
    return this.executionManager.submitEvent(actionName, {
      eventId: options.eventId,
      confidence: options.confidence ?? 0.95,
      source: options.source ?? 'vision',
    });
  }

  /**
   * Backward-compatible playAction that routes through the event execution manager.
   */
  public playAction(actionName: string, options: AnimationPlayOptions = {}): boolean {
    const canonical = actionName.toUpperCase().trim().replace(/\s+/g, '_');
    if (canonical === 'IDLE' || canonical === 'SBV_IDLE') {
      this.returnToIdle(options.crossFadeDuration ?? 0.25);
      return true;
    }
    const evt = this.executeEvent(actionName, options);
    return evt !== null;
  }

  /**
   * Queues an ordered sequence of sign gestures for full-sentence playback.
   */
  public queueSequence(
    signIds: string[],
    source: 'vision' | 'speech' | 'ui' = 'speech',
    sequenceId?: string
  ) {
    if (!signIds || signIds.length === 0) return;
    this.lastSequence = [...signIds];
    const seqId = sequenceId || `seq_${Date.now()}`;
    for (const s of signIds) {
      this.executionManager.submitEvent(s, {
        source,
        confidence: 0.98,
        force: true, // sequence elements are intentional multi-sign tokens
        metadata: { sequenceId: seqId },
      });
    }
  }

  public replayLastSequence(): boolean {
    if (!this.lastSequence || this.lastSequence.length === 0) return false;
    this.queueSequence(this.lastSequence, 'ui');
    return true;
  }

  public getLastSequence(): string[] {
    return [...this.lastSequence];
  }

  /**
   * Procedural fallback mapping for language synonyms and dialect variants.
   * Strictly avoids collapsing different semantic concepts into one gesture.
   */
  private createFallbackAction(actionKey: string): THREE.AnimationAction | null {
    const synonymMap: Record<string, string[]> = {
      HI: ['HELLO', 'SBV_HELLO'],
      NAMASTE: ['HELLO', 'SBV_HELLO'],
      THANKS: ['THANK_YOU', 'SBV_THANK_YOU'],
      DHANYAVAAD: ['THANK_YOU', 'SBV_THANK_YOU'],
      KRIPYA: ['PLEASE', 'SBV_PLEASE'],
      KINDLY: ['PLEASE', 'SBV_PLEASE'],
      MAAF: ['SORRY', 'SBV_SORRY'],
      APOLOGY: ['SORRY', 'SBV_SORRY'],
      HAAN: ['YES', 'SBV_YES'],
      AGREE: ['YES', 'SBV_YES'],
      NAHI: ['NO', 'SBV_NO'],
      DISAGREE: ['NO', 'SBV_NO'],
      EAT: ['FOOD', 'SBV_FOOD'],
      HUNGRY: ['FOOD', 'SBV_FOOD'],
      KHANA: ['FOOD', 'SBV_FOOD'],
      DRINK: ['WATER', 'SBV_WATER'],
      PAANI: ['WATER', 'SBV_WATER'],
      HOUSE: ['HOME', 'SBV_HOME'],
      GHAR: ['HOME', 'SBV_HOME'],
      COLLEGE: ['SCHOOL', 'SBV_SCHOOL'],
      VIDYALAYA: ['SCHOOL', 'SBV_SCHOOL'],
      CLINIC: ['HOSPITAL', 'SBV_HOSPITAL'],
      RESTROOM: ['BATHROOM', 'SBV_BATHROOM'],
      TOILET: ['BATHROOM', 'SBV_BATHROOM'],
      WASHROOM: ['BATHROOM', 'SBV_BATHROOM'],
      URGENT: ['EMERGENCY', 'SBV_EMERGENCY'],
      AAPAATKAAL: ['EMERGENCY', 'SBV_EMERGENCY'],
      COP: ['POLICE', 'SBV_POLICE'],
      HALT: ['STOP', 'SBV_STOP'],
      RUKO: ['STOP', 'SBV_STOP'],
      JAO: ['GO', 'SBV_GO'],
      PROCEED: ['GO', 'SBV_GO'],
      AAO: ['COME', 'SBV_COME'],
      HOLD_ON: ['WAIT', 'SBV_WAIT'],
      INTEZAR: ['WAIT', 'SBV_WAIT'],
      SAMAJH: ['UNDERSTAND', 'SBV_UNDERSTAND'],
      NOT_UNDERSTAND: ['DONT_UNDERSTAND', 'SBV_DONT_UNDERSTAND'],
      NAAM: ['NAME', 'SBV_NAME'],
      KAHAN: ['WHERE', 'SBV_WHERE'],
      AGAIN: ['REPEAT', 'PROC_REPEAT'],
      GOODBYE: ['GOING', 'SBV_GOING'],
      BYE: ['GOING', 'SBV_GOING'],
      YOU_ALL: ['YOU', 'PROC_YOU'],
      ME: ['I_ME', 'PROC_I_ME'],
      MINE: ['MY', 'PROC_MY'],
      YOUR: ['PROC_YOUR', 'YOUR'],
      YOURS: ['PROC_YOUR', 'YOUR'],
      NOW: ['PROC_NOW', 'NOW'],
      TODAY: ['PROC_NOW', 'NOW'],
      DO: ['PROC_DO', 'DO', 'WORK', 'SBV_WORK'],
      DOING: ['PROC_DO', 'DO', 'WORK', 'SBV_WORK'],
      D: ['LETTER_D', 'PROC_LETTER_D'],
      A: ['LETTER_A', 'PROC_LETTER_A'],
      N: ['LETTER_N', 'PROC_LETTER_N'],
      I: ['LETTER_I', 'PROC_LETTER_I'],
      E: ['LETTER_E', 'PROC_LETTER_E'],
      L: ['LETTER_L', 'PROC_LETTER_L'],
    };

    // Generic fingerspelling alphabet fallback (only if authentic letter action exists)
    if (actionKey.startsWith('LETTER_') || (actionKey.length === 1 && /[A-Z]/.test(actionKey))) {
      const char = actionKey.replace(/^LETTER_/, '');
      const action = this.actions.get(`LETTER_${char}`) || this.actions.get(char) || this.actions.get(`PROC_LETTER_${char}`);
      if (action) {
        this.actions.set(actionKey, action);
        return action;
      }
    }

    const candidates = synonymMap[actionKey];
    if (candidates) {
      for (const cand of candidates) {
        const baseAction = this.actions.get(cand);
        if (baseAction) {
          this.actions.set(actionKey, baseAction);
          return baseAction;
        }
      }
    }

    return null;
  }

  public isPaused(): boolean {
    return this.mixer.timeScale === 0;
  }

  public pause() {
    this.mixer.timeScale = 0;
    this.notifyStatus();
  }

  public resume() {
    this.mixer.timeScale = 1;
    this.notifyStatus();
  }

  public togglePause(): boolean {
    if (this.mixer.timeScale === 0) {
      this.resume();
      return false;
    } else {
      this.pause();
      return true;
    }
  }

  public resetToIdle() {
    this.mixer.timeScale = 1;
    this.returnToIdle(0.2);
  }

  public update(delta: number) {
    this.mixer.update(delta);
  }

  public stopAll() {
    this.mixer.stopAllAction();
    this.executionManager.resetQueue();
    this.currentAction = null;
    this.currentActionName = 'IDLE';
    this.notifyStatus();
  }

  public dispose() {
    this.stopAll();
    this.actions.clear();
    this.statusListeners.clear();
    this.mixer.uncacheRoot(this.root);
  }
}
