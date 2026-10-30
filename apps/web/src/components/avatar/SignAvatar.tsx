'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RotateCcw, Sparkles, CheckCircle2, ChevronRight } from 'lucide-react';
import { AnimationController, AnimationControllerStatus } from './AnimationController';
import { matchSpeechSequence } from '../../lib/speech/phraseMatcher';

interface SignAvatarProps {
  currentSign?: string | null;
  textToSign?: string | null;
  className?: string;
  autoPlay?: boolean;
}

// Full 25-concept ISL actions for direct demonstration and testing
const ALL_25_ISL_ACTIONS = [
  { label: 'Hello', id: 'HELLO' },
  { label: 'Thank You', id: 'THANK_YOU' },
  { label: 'Please', id: 'PLEASE' },
  { label: 'Sorry', id: 'SORRY' },
  { label: 'Yes', id: 'YES' },
  { label: 'No', id: 'NO' },
  { label: 'Good', id: 'GOOD' },
  { label: 'Help', id: 'HELP' },
  { label: 'Water', id: 'WATER' },
  { label: 'Food', id: 'FOOD' },
  { label: 'Home', id: 'HOME' },
  { label: 'School', id: 'SCHOOL' },
  { label: 'Hospital', id: 'HOSPITAL' },
  { label: 'Need', id: 'NEED' },
  { label: 'Emergency', id: 'EMERGENCY' },
  { label: 'Police', id: 'POLICE' },
  { label: 'Stop', id: 'STOP' },
  { label: 'Go', id: 'GO' },
  { label: 'Wait', id: 'WAIT' },
  { label: 'Come', id: 'COME' },
  { label: 'Understand', id: 'UNDERSTAND' },
  { label: "Don't Understand", id: 'DONT_UNDERSTAND' },
  { label: 'Name', id: 'NAME' },
  { label: 'Where', id: 'WHERE' },
  { label: 'Bathroom', id: 'BATHROOM' },
];

export const SignAvatar: React.FC<SignAvatarProps> = ({
  currentSign,
  textToSign,
  className = '',
  autoPlay = true,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<AnimationController | null>(null);
  const pendingSignRef = useRef<string | null>(null);
  const pendingSequenceRef = useRef<string[] | null>(null);

  const [activeAction, setActiveAction] = useState<string>('IDLE');
  const [isGlbLoaded, setIsGlbLoaded] = useState<boolean>(false);
  const [controllerStatus, setControllerStatus] = useState<AnimationControllerStatus>({
    activeAction: 'IDLE',
    loadedClips: [],
    queuedActions: [],
    missingAssets: [],
    lastError: null,
  });

  // Play a specific sign action
  const performSign = useCallback((signId: string) => {
    if (!signId) return;
    if (controllerRef.current) {
      controllerRef.current.playAction(signId, { loop: false, crossFadeDuration: 0.25 });
    } else {
      pendingSignRef.current = signId;
    }
  }, []);

  // When vision recognized gesture changes
  useEffect(() => {
    if (currentSign && autoPlay) {
      performSign(currentSign);
    }
  }, [currentSign, autoPlay, performSign]);

  // When spoken/typed speech transcript changes (supports full sentences & multiple signs)
  useEffect(() => {
    if (textToSign && autoPlay) {
      const sequence = matchSpeechSequence(textToSign);
      if (sequence.length > 0) {
        if (controllerRef.current) {
          controllerRef.current.queueSequence(sequence);
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

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1e293b);

    // Camera setup (optimal framing for ISL upper body and hand signing)
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 1.35, 2.7);
    camera.lookAt(0, 1.25, 0);

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Natural studio lighting for skin & clothes
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfff5e6, 1.4);
    keyLight.position.set(1.5, 4, 3);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x90c0ff, 0.6);
    fillLight.position.set(-2, 2, 2);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.8);
    rimLight.position.set(0, 3, -3);
    scene.add(rimLight);

    // Platform Shadow Receiver
    const floorGeo = new THREE.CylinderGeometry(1.6, 1.6, 0.05, 32);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.8 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.position.y = -0.025;
    floor.receiveShadow = true;
    scene.add(floor);

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
            child.castShadow = true;
            child.receiveShadow = true;
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
        controller.registerStatusListener((status) => {
          setControllerStatus(status);
          setActiveAction(status.activeAction);
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

    const animate = () => {
      const delta = clock.getDelta();
      if (controller) {
        controller.update(delta);
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
    <div className={`relative rounded-2xl overflow-hidden bg-[#1e293b] border border-slate-700/80 shadow-2xl flex flex-col ${className}`}>
      {/* Top Header Badge */}
      <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-700 shadow-lg">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-semibold text-slate-200">
            {activeAction === 'IDLE' ? 'Avatar: Idle' : `Signing: ${activeAction}`}
          </span>
          {isGlbLoaded && (
            <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/30 flex items-center gap-1">
              <CheckCircle2 className="w-2.5 h-2.5" /> 25 ISL Signs Active
            </span>
          )}
          {controllerStatus.queuedActions.length > 0 && (
            <span className="text-[10px] bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded border border-sky-500/30 flex items-center gap-1">
              Queued: {controllerStatus.queuedActions.length}
            </span>
          )}
        </div>

        {activeAction !== 'IDLE' && (
          <button
            onClick={() => performSign(activeAction)}
            className="pointer-events-auto bg-sky-500/90 hover:bg-sky-500 text-white text-xs px-2.5 py-1 rounded-full flex items-center gap-1 transition-all shadow-md"
          >
            <RotateCcw className="w-3 h-3" /> Replay
          </button>
        )}
      </div>

      {/* 3D WebGL Canvas Container */}
      <div ref={mountRef} className="w-full h-[380px] sm:h-[420px]" />

      {/* 25-Sign Quick Actions Bar */}
      <div className="p-3 bg-slate-900/90 border-t border-slate-800 flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium text-[11px] flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-sky-400" /> 25 ISL Concept Vocabulary:
          </span>
          <span className="text-[10px] text-slate-500">Click any sign to animate avatar</span>
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs scrollbar-thin scrollbar-thumb-slate-700">
          {ALL_25_ISL_ACTIONS.map((item) => (
            <button
              key={item.id}
              onClick={() => performSign(item.id)}
              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold whitespace-nowrap transition-all ${
                activeAction === item.id
                  ? 'bg-sky-500 text-white shadow-md shadow-sky-500/30'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
