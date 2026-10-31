'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RotateCcw, Sparkles, CheckCircle2, Play, Pause, RefreshCw, Activity, ShieldCheck } from 'lucide-react';
import { AnimationController, AnimationControllerStatus, ActiveBoneMetrics } from './AnimationController';
import { matchSpeechSequence } from '../../lib/speech/phraseMatcher';

export interface SignAvatarProps {
  currentSign?: string | null;
  textToSign?: string | null;
  className?: string;
  autoPlay?: boolean;
  theme?: 'light' | 'dark';
  showControls?: boolean;
  showQuickActions?: boolean;
  onStatusChange?: (status: { activeAction: string; isPaused: boolean; isGlbLoaded: boolean; queuedActions?: string[] }) => void;
  onPerformSignReady?: (performSign: (signId: string) => void) => void;
  onPerformSequenceReady?: (performSequence: (signIds: string[]) => void) => void;
  onReplaySequenceReady?: (replaySequence: () => boolean) => void;
  onSequenceCompleted?: (sequenceId?: string) => void;
}

// Full 50 ISL actions covering all everyday, question, activity, assistance, and emergency categories
const CORE_ISL_ACTIONS = [
  { label: 'Hello', id: 'HELLO' },
  { label: 'Name', id: 'NAME' },
  { label: 'Nice to meet you', id: 'NICE_TO_MEET_YOU' },
  { label: 'You', id: 'YOU' },
  { label: 'I / Me', id: 'I_ME' },
  { label: 'My', id: 'MY' },
  { label: 'Friend', id: 'FRIEND' },
  { label: 'Goodbye', id: 'GOODBYE' },
  { label: 'Deaf', id: 'DEAF' },
  { label: 'Hearing', id: 'HEARING' },
  { label: 'How are you?', id: 'HOW_ARE_YOU' },
  { label: 'Good', id: 'GOOD' },
  { label: 'Bad', id: 'BAD' },
  { label: 'Yes', id: 'YES' },
  { label: 'No', id: 'NO' },
  { label: 'Please', id: 'PLEASE' },
  { label: 'Thank You', id: 'THANK_YOU' },
  { label: 'Sorry', id: 'SORRY' },
  { label: 'What', id: 'WHAT' },
  { label: 'Where', id: 'WHERE' },
  { label: 'How', id: 'HOW' },
  { label: 'Help', id: 'HELP' },
  { label: 'Want', id: 'WANT' },
  { label: 'Need', id: 'NEED' },
  { label: 'Wait', id: 'WAIT' },
  { label: 'Call', id: 'CALL' },
  { label: 'Give', id: 'GIVE' },
  { label: 'Hospital', id: 'HOSPITAL' },
  { label: 'Doctor', id: 'DOCTOR' },
  { label: 'Medicine', id: 'MEDICINE' },
  { label: 'Pain', id: 'PAIN' },
  { label: 'Emergency', id: 'EMERGENCY' },
  { label: 'Ambulance', id: 'AMBULANCE' },
  { label: 'Police', id: 'POLICE' },
  { label: 'Danger', id: 'DANGER' },
  { label: 'Stop', id: 'STOP' },
  { label: 'School', id: 'SCHOOL' },
  { label: 'Work', id: 'WORK' },
  { label: 'Understand', id: 'UNDERSTAND' },
  { label: "Don't Understand", id: 'DONT_UNDERSTAND' },
  { label: 'Repeat', id: 'REPEAT' },
  { label: 'Bathroom', id: 'BATHROOM' },
  { label: 'Here', id: 'HERE' },
  { label: 'Left', id: 'LEFT' },
  { label: 'Right', id: 'RIGHT' },
  { label: 'Water', id: 'WATER' },
  { label: 'Food', id: 'FOOD' },
  { label: 'Go', id: 'GO' },
  { label: 'Come', id: 'COME' },
  { label: 'Home', id: 'HOME' },
];

const ALL_25_ISL_ACTIONS = CORE_ISL_ACTIONS;
const CORE_ASL_ACTIONS = CORE_ISL_ACTIONS;

export const SignAvatar: React.FC<SignAvatarProps> = ({
  currentSign,
  textToSign,
  className = '',
  autoPlay = true,
  theme = 'light',
  showControls = true,
  showQuickActions = false,
  onStatusChange,
  onPerformSignReady,
  onPerformSequenceReady,
  onReplaySequenceReady,
  onSequenceCompleted,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<AnimationController | null>(null);
  const pendingSignRef = useRef<string | null>(null);
  const pendingSequenceRef = useRef<string[] | null>(null);

  const [activeAction, setActiveAction] = useState<string>('IDLE');
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isGlbLoaded, setIsGlbLoaded] = useState<boolean>(false);
  const [showDebugOverlay, setShowDebugOverlay] = useState<boolean>(false);
  const [boneMetrics, setBoneMetrics] = useState<ActiveBoneMetrics | null>(null);
  const showDebugRef = useRef<boolean>(false);

  useEffect(() => {
    showDebugRef.current = showDebugOverlay;
  }, [showDebugOverlay]);

  const [controllerStatus, setControllerStatus] = useState<AnimationControllerStatus>({
    activeAction: 'IDLE',
    loadedClips: [],
    queuedActions: [],
    missingAssets: [],
    lastError: null,
    lifecycleState: 'IDLE',
  });

  const lastHandledSignRef = useRef<string | null>(null);
  const lastHandledTextRef = useRef<string | null>(null);

  // Play a specific sign action
  const performSign = useCallback((signId: string) => {
    if (!signId) return;
    if (controllerRef.current) {
      controllerRef.current.playAction(signId, { loop: false, crossFadeDuration: 0.25 });
    } else {
      pendingSignRef.current = signId;
    }
  }, []);

  // Play an ordered sequence of sign actions for complete sentences
  const performSequence = useCallback((signIds: string[]) => {
    if (!signIds || signIds.length === 0) return;
    if (controllerRef.current) {
      controllerRef.current.queueSequence(signIds, 'speech');
    } else {
      pendingSequenceRef.current = signIds;
    }
  }, []);

  const replaySequence = useCallback(() => {
    if (controllerRef.current) {
      return controllerRef.current.replayLastSequence();
    }
    return false;
  }, []);

  // Expose performSign, performSequence, and replaySequence to parent components
  useEffect(() => {
    if (onPerformSignReady) {
      onPerformSignReady(performSign);
    }
    if (onPerformSequenceReady) {
      onPerformSequenceReady(performSequence);
    }
    if (onReplaySequenceReady) {
      onReplaySequenceReady(replaySequence);
    }
  }, [onPerformSignReady, onPerformSequenceReady, onReplaySequenceReady, performSign, performSequence, replaySequence]);

  // Notify parent of status changes
  useEffect(() => {
    if (onStatusChange) {
      onStatusChange({
        activeAction,
        isPaused,
        isGlbLoaded,
      });
    }
  }, [onStatusChange, activeAction, isPaused, isGlbLoaded]);

  const togglePause = useCallback(() => {
    if (controllerRef.current) {
      const paused = controllerRef.current.togglePause();
      setIsPaused(paused);
    }
  }, []);

  const resetAvatar = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.resetToIdle();
      setIsPaused(false);
    }
  }, []);

  // When vision recognized gesture changes
  useEffect(() => {
    if (currentSign && autoPlay && currentSign !== lastHandledSignRef.current) {
      lastHandledSignRef.current = currentSign;
      performSign(currentSign);
    }
  }, [currentSign, autoPlay, performSign]);

  // When spoken/typed speech transcript changes (supports full sentences & multiple signs)
  useEffect(() => {
    if (textToSign && autoPlay && textToSign !== lastHandledTextRef.current) {
      lastHandledTextRef.current = textToSign;
      const sequence = matchSpeechSequence(textToSign);
      if (sequence.length > 0) {
        if (controllerRef.current) {
          controllerRef.current.queueSequence(sequence, 'speech');
        } else {
          pendingSequenceRef.current = sequence;
        }
      }
    }
  }, [textToSign, autoPlay]);

  // Initialize Three.js Scene and Load Avatar
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 540;
    const height = container.clientHeight || 400;

    const isLight = theme === 'light';

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(isLight ? 0xffffff : 0x1e293b);

    // Camera setup (optimal framing for ISL upper body and hand signing)
    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
    camera.position.set(0, 1.34, 2.45);
    camera.lookAt(0, 1.22, 0);

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = isLight ? 1.18 : 1.35;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // 1. Natural studio hemisphere light for ambient sky/ground fill
    const hemiLight = new THREE.HemisphereLight(
      0xffffff,
      isLight ? 0xf1f5f9 : 0x1e293b,
      isLight ? 1.35 : 1.25
    );
    hemiLight.position.set(0, 5, 0);
    scene.add(hemiLight);

    // 2. High-clarity key light focused on face and upper chest signing space
    const keyLight = new THREE.DirectionalLight(0xfffbf5, isLight ? 1.6 : 1.75);
    keyLight.position.set(1.5, 4, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    scene.add(keyLight);

    // 3. Direct frontal facial & hand illumination fill light
    const frontFillLight = new THREE.DirectionalLight(0xffffff, isLight ? 0.95 : 1.0);
    frontFillLight.position.set(0, 1.5, 2.5);
    scene.add(frontFillLight);

    // 4. Soft cool side fill for dimensional contrast
    const sideFillLight = new THREE.DirectionalLight(0xe0f2fe, 0.5);
    sideFillLight.position.set(-2, 2, 2);
    scene.add(sideFillLight);

    // Optional dark platform for dark theme only
    if (!isLight) {
      const rimLight = new THREE.DirectionalLight(0xffedd5, 1.1);
      rimLight.position.set(0, 3, -3);
      scene.add(rimLight);

      const floorGeo = new THREE.CylinderGeometry(1.6, 1.6, 0.05, 32);
      const floorMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.8 });
      const floor = new THREE.Mesh(floorGeo, floorMat);
      floor.position.y = -0.025;
      floor.receiveShadow = true;
      scene.add(floor);
    }

    let animFrame: number | null = null;
    const clock = new THREE.Clock();
    let controller: AnimationController | null = null;

    // Load Production Avatar Model
    const loader = new GLTFLoader();
    loader.load(
      '/models/sign-avatar/sign-avatar.glb',
      (gltf) => {
        const model = gltf.scene;
        model.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            // Correct PBR materials on avatar:
            // Organic skin and cotton garments are non-metallic dielectrics.
            // Eliminates the 0.5 metallic absorption that made the avatar look muddy and dark.
            if (mesh.material) {
              const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
              materials.forEach((mat) => {
                if (mat instanceof THREE.MeshStandardMaterial) {
                  mat.metalness = 0.02;
                  mat.roughness = 0.45;
                  mat.needsUpdate = true;
                }
              });
            }
          }
          if (child.name && child.name.toLowerCase().includes('icosphere')) {
            child.visible = false;
          }
        });
        scene.add(model);

        // Instantiate unified AnimationController
        controller = new AnimationController(model);
        controllerRef.current = controller;

        // Register GLTF embedded clips (starts SBV_IDLE immediately to eliminate T-pose)
        controller.registerClips(gltf.animations);

        // Wire status updates
        controller.onSequenceComplete = (seqId) => {
          if (onSequenceCompleted) {
            onSequenceCompleted(seqId);
          }
        };

        controller.registerStatusListener((status) => {
          setControllerStatus(status);
          setActiveAction(status.activeAction);
          // Unlock repeated gesture trigger when avatar returns to IDLE
          if (status.activeAction === 'IDLE' || status.lifecycleState === 'IDLE') {
            lastHandledSignRef.current = null;
          }
        });

        setIsGlbLoaded(true);

        // Process any queued or pending actions
        if (pendingSequenceRef.current && pendingSequenceRef.current.length > 0) {
          controller.queueSequence(pendingSequenceRef.current);
          pendingSequenceRef.current = null;
        } else if (pendingSignRef.current) {
          controller.playAction(pendingSignRef.current, { loop: false });
          pendingSignRef.current = null;
        } else {
          // Play initial greeting gesture
          controller.playAction('HELLO', { loop: false });
        }
      },
      undefined,
      (err) => {
        console.error('[SignAvatar] GLB load failed:', err);
      }
    );

    let lastMetricUpdate = 0;
    const animate = () => {
      const delta = clock.getDelta();
      if (controller) {
        controller.update(delta);
        const now = performance.now();
        if (now - lastMetricUpdate > 100) {
          lastMetricUpdate = now;
          if (showDebugRef.current) {
            setBoneMetrics(controller.getActiveBoneMetrics());
          }
        }
      }
      renderer.render(scene, camera);
      animFrame = requestAnimationFrame(animate);
    };
    animate();

    const handleResize = () => {
      if (!container || !renderer) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animFrame) cancelAnimationFrame(animFrame);
      if (controller) controller.dispose();
      controllerRef.current = null;
      renderer.dispose();
    };
  }, []);

  return (
    <div
      className={`relative rounded-2xl overflow-hidden ${
        theme === 'light'
          ? 'bg-white border border-[#DCE5EF] shadow-sm'
          : 'bg-[#1e293b] border border-slate-700/80 shadow-2xl'
      } flex flex-col ${className}`}
    >
      {/* Top Header Controls Overlay */}
      {showControls && (
        <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full border shadow-sm backdrop-blur-md pointer-events-auto ${
              theme === 'light'
                ? 'bg-white/95 border-[#DCE5EF] text-[#14264F]'
                : 'bg-slate-900/85 border-slate-700 text-slate-200'
            }`}
          >
            <div
              className={`w-2 h-2 rounded-full ${
                isPaused
                  ? 'bg-amber-400'
                  : activeAction === 'IDLE'
                  ? 'bg-[#12B8C2]'
                  : 'bg-emerald-500 animate-pulse'
              }`}
            />
            <span className="text-xs font-semibold">
              {isPaused
                ? 'Paused'
                : activeAction === 'IDLE'
                ? 'Avatar: Idle'
                : `Signing: ${activeAction}`}
            </span>
            {isGlbLoaded && (
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded border font-medium flex items-center gap-1 ${
                  theme === 'light'
                    ? 'bg-[#E7F4FC] text-[#12B8C2] border-[#12B8C2]/30'
                    : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                }`}
              >
                <CheckCircle2 className="w-2.5 h-2.5" /> 50 ISL
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 pointer-events-auto">
            <button
              onClick={togglePause}
              title={isPaused ? 'Resume Playback' : 'Pause Playback'}
              className={`text-xs px-2.5 py-1 rounded-full flex items-center gap-1 transition-all shadow-sm border font-medium ${
                theme === 'light'
                  ? isPaused
                    ? 'bg-[#12B8C2] text-white border-[#12B8C2]'
                    : 'bg-white hover:bg-[#E7F4FC] text-[#14264F] border-[#DCE5EF]'
                  : isPaused
                  ? 'bg-amber-500 text-white border-amber-600'
                  : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-700'
              }`}
            >
              {isPaused ? <Play className="w-3 h-3 fill-current" /> : <Pause className="w-3 h-3" />}
              <span>{isPaused ? 'Play' : 'Pause'}</span>
            </button>

            <button
              onClick={resetAvatar}
              title="Reset Avatar to Neutral Idle"
              className={`text-xs px-2.5 py-1 rounded-full flex items-center gap-1 transition-all shadow-sm border font-medium ${
                theme === 'light'
                  ? 'bg-white hover:bg-[#E7F4FC] text-[#14264F] border-[#DCE5EF]'
                  : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-700'
              }`}
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>

            <button
              onClick={() => setShowDebugOverlay((prev) => !prev)}
              title={showDebugOverlay ? 'Hide Visual Debug HUD' : 'Show Skeletal & Armature Debug Overlay'}
              className={`text-xs px-2.5 py-1 rounded-full flex items-center gap-1 transition-all shadow-sm border font-medium ${
                theme === 'light'
                  ? showDebugOverlay
                    ? 'bg-[#14264F] text-white border-[#14264F]'
                    : 'bg-white hover:bg-[#E7F4FC] text-[#14264F] border-[#DCE5EF]'
                  : showDebugOverlay
                  ? 'bg-cyan-500 text-slate-950 border-cyan-400 font-semibold'
                  : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-700'
              }`}
            >
              <Activity className="w-3 h-3" />
              <span>{showDebugOverlay ? 'Hide HUD' : 'Debug HUD'}</span>
            </button>
          </div>
        </div>
      )}

      {/* Developer Visual Debugging Overlay (V15 Armature & Kinematics Telemetry) */}
      {showDebugOverlay && boneMetrics && (
        <div className="absolute top-14 left-3 right-3 z-30 bg-[#0F172A]/90 backdrop-blur-md border border-slate-700/80 rounded-xl p-3 shadow-2xl text-slate-200 text-[11px] font-mono pointer-events-auto">
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-700/80">
            <div className="flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-[#12B8C2]" />
              <span className="font-bold text-white tracking-wide">AVATAR SKELETAL & ARMATURE HUD</span>
              <span
                className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${
                  boneMetrics.source === 'GLB'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : boneMetrics.source === 'PROCEDURAL'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'bg-slate-700 text-slate-300'
                }`}
              >
                {boneMetrics.source === 'GLB' ? 'GLB NATIVE' : boneMetrics.source === 'PROCEDURAL' ? 'PROCEDURAL ISL' : 'REST IDLE'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1 text-[10px] text-emerald-400 font-medium">
                <ShieldCheck className="w-3 h-3" /> NATURAL POSE
              </span>
              <span className="text-slate-400 text-[10px]">Clip: {boneMetrics.clipName}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 text-[10px] leading-tight">
            {/* Right Arm Chain (Dominant) */}
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <div className="text-[#12B8C2] font-semibold mb-1 flex items-center justify-between">
                <span>DOMINANT (RIGHT) ARM CHAIN</span>
                <span>Flex: {boneMetrics.rForeArm.flexionDeg}°</span>
              </div>
              <div className="space-y-0.5 text-slate-300">
                <div>RightArm Euler: ({boneMetrics.rArm.euler[0]}°, {boneMetrics.rArm.euler[1]}°, {boneMetrics.rArm.euler[2]}°)</div>
                <div>RightForeArm Euler: ({boneMetrics.rForeArm.euler[0]}°, {boneMetrics.rForeArm.euler[1]}°, {boneMetrics.rForeArm.euler[2]}°)</div>
                <div>RightHand (Wrist): ({boneMetrics.rHand.euler[0]}°, {boneMetrics.rHand.euler[1]}°, {boneMetrics.rHand.euler[2]}°)</div>
                <div className="text-[9px] text-slate-400">Wrist Quat: [{boneMetrics.rHand.quat.join(', ')}]</div>
              </div>
            </div>

            {/* Left Arm Chain (Non-Dominant) */}
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <div className="text-slate-400 font-semibold mb-1 flex items-center justify-between">
                <span>LEFT ARM CHAIN</span>
                <span>Flex: {boneMetrics.lForeArm.flexionDeg}°</span>
              </div>
              <div className="space-y-0.5 text-slate-300">
                <div>LeftArm Euler: ({boneMetrics.lArm.euler[0]}°, {boneMetrics.lArm.euler[1]}°, {boneMetrics.lArm.euler[2]}°)</div>
                <div>LeftForeArm Euler: ({boneMetrics.lForeArm.euler[0]}°, {boneMetrics.lForeArm.euler[1]}°, {boneMetrics.lForeArm.euler[2]}°)</div>
                <div>LeftHand (Wrist): ({boneMetrics.lHand.euler[0]}°, {boneMetrics.lHand.euler[1]}°, {boneMetrics.lHand.euler[2]}°)</div>
                <div className="text-[9px] text-slate-400">Wrist Quat: [{boneMetrics.lHand.quat.join(', ')}]</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Unsupported Sign Notice */}
      {controllerStatus.lastError && (
        <div className="absolute top-12 left-3 right-3 z-20 bg-amber-50/95 border border-amber-300 text-amber-900 text-[11px] px-3 py-1.5 rounded-lg flex items-center justify-between shadow-xs">
          <span className="font-medium">{controllerStatus.lastError}</span>
          <button
            onClick={() => setControllerStatus((prev) => ({ ...prev, lastError: null }))}
            className="text-amber-700 hover:text-amber-950 font-bold ml-2 text-xs"
          >
            ×
          </button>
        </div>
      )}

      {/* 3D WebGL Canvas Container */}
      <div
        ref={mountRef}
        className={`w-full h-[360px] sm:h-[400px] flex items-center justify-center ${
          theme === 'light' ? 'bg-white' : 'bg-[#1e293b]'
        }`}
      />

      {/* Optional Quick Actions Bar (if requested) */}
      {showQuickActions && (
        <div
          className={`p-3 border-t flex flex-col gap-2 ${
            theme === 'light'
              ? 'bg-[#F7F9FC] border-[#DCE5EF]'
              : 'bg-slate-900/90 border-slate-800'
          }`}
        >
          <div className="flex items-center justify-between text-xs text-[#71809A]">
            <span className="font-medium text-[11px] flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-[#12B8C2]" /> ISL Concept Vocabulary:
            </span>
            <span className="text-[10px]">Click any sign to animate avatar</span>
          </div>

          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
            {ALL_25_ISL_ACTIONS.map((item) => (
              <button
                key={item.id}
                onClick={() => performSign(item.id)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-semibold whitespace-nowrap transition-all border ${
                  activeAction === item.id
                    ? 'bg-[#12B8C2] text-white border-[#12B8C2] shadow-sm'
                    : 'bg-white text-[#14264F] border-[#DCE5EF] hover:bg-[#E7F4FC]'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
