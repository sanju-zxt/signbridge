# SignBridge V7 — Complete Model, Dataset & Pipeline Audit Report

**Document ID:** `V7-MODEL-AUDIT-20261031`  
**System Title:** SignBridge V7: Multimodal Indian Sign Language (ISL) Recognition Engine  
**Auditor:** Principal Machine Learning Engineer & Real-Time Computer Vision Architect  
**Audit Date:** October 31, 2026  
**Status:** COMPLETE — ROOT CAUSES IDENTIFIED & DOCUMENTED  

---

## 1. Executive Summary: The Core Discrepancy

The previous V6 validation report established an independently reconstructed Top-1 accuracy of **83.33%** on a 17-class benchmark dataset. However, in real-world application, users experienced two major defects:
1. **The model fails to recognize most performed ISL signs.**
2. **The model repeatedly predicts the same sign for completely different hand gestures.**
3. **The 3D avatar executes the same generic gesture for different recognized words.**

This audit conducted an end-to-end investigation of the entire machine learning pipeline to uncover the true root causes.

### Primary Audit Findings:
1. **Critical Vocabulary vs. Model Discrepancy:** The runtime vocabulary defines **49 verified ISL signs**, but the trained dual-layer GRU model in `apps/web/public/models/sbv-gru-v1/` has an output dimension of **ONLY 17 classes**. Exactly **32 runtime signs (65.3% of the vocabulary) have ZERO representation or weights in the neural network!**
2. **Prediction Collapse Mechanism:** Because the neural network ends in `Dense(17, Softmax)`, all output probabilities are mathematically constrained to sum to 1.0 across the 17 benchmark classes. When a user performs any of the 32 untrained signs (such as `NAME`, `YOU`, `I_ME`, `DOCTOR`, `EMERGENCY`, `POLICE`, `STOP`, `WAIT`, `GOOD`, `BAD`, `COME`, `GO`), the model is mathematically forced to misclassify the gesture into one of the 17 benchmark classes.
3. **Heuristic Kinematic Hijacking:** In `apps/web/src/workers/inferenceWorker.ts`, lines 400–575 implement hardcoded geometric rule heuristics (`isOpenHand`, `isFist`, `isIndexPointing`). These heuristics were originally written for ASL gestures. When any open palm or pointing finger is detected, the heuristic assigns arbitrary scores (>0.90) to `HELLO`, `WHERE`, `YOU`, or `THANK_YOU`, and blends with the neural model via `scores * 0.4 + neuralProb * 0.6`. This overrides neural predictions and locks the model into outputting the same sign repeatedly.
4. **Avatar Animation Alias Collapsing:** In `apps/web/src/components/avatar/AnimationController.ts`, `createFallbackAction` collapses distinct signs onto single animations (`SORRY` $\to$ `PLEASE`, `GOOD` $\to$ `YES`, `BAD` $\to$ `NO`, `HOW_ARE_YOU` $\to$ `WHERE`, `YOU` $\to$ `HERE`, `I_ME` $\to$ `PLEASE`, `DOCTOR` $\to$ `HOSPITAL`, `WANT` $\to$ `NEED`). Furthermore, 20+ vocabulary signs without explicit clips default to `IDLE` or fallbacks.
5. **Avatar Visual Darkness:** The 3D model `sign-avatar.glb` uses material `Ch28_body` with `metallic = 0.5`. Human skin is non-metallic ($\text{metalness} \approx 0.0$). A metallic factor of 0.5 without HDR environment reflections absorbs diffuse light, producing a dark, greyish, muddy appearance. The scene also lacks a `HemisphereLight`, uses low tone mapping exposure, and lacks frontal facial/hand fill light.

---

## 2. Quantitative Model & Pipeline Inventory

| Parameter / Layer | Existing V6 State | Audit Finding / Defect |
|---|---|---|
| **Model Architecture** | 2-Layer GRU (64 units each) | Valid baseline architecture; parameter count 72,337. |
| **Input Tensor Shape** | `[batch, 30, 126]` | 30 frames $\times$ 126 normalized hand coordinates (21 joints $\times$ 3 xyz $\times$ 2 hands). Valid. |
| **Output Layer** | `Dense(17, Softmax)` | **CRITICAL DEFECT:** 17 classes only. Cannot output remaining 32 runtime vocabulary signs. |
| **Runtime Vocabulary** | 49 verified signs | 17 signs trained (34.7%), **32 signs completely untrained (65.3%)**. |
| **Benchmark Dataset** | 510 sequences across 10 signers | High quality for 17 classes (30 seqs/class), but lacks data for the other 32 signs. |
| **Feature Normalization** | Wrist-relative, palm-diagonal scale | Scale factor $s = \|\mathbf{LM}_9 - \mathbf{LM}_0\|$. Robust to camera distance. |
| **Worker Inference Pipeline** | Heuristic kinematic fusion | **DEFECT:** Kinematic heuristics hijack model predictions and force identical outputs. |
| **Confidence Gate** | Threshold $\ge 0.60$ | Does not reject untrained out-of-distribution gestures because Softmax forces sum to 1.0. |
| **3D Avatar Lighting** | Ambient (0.85), Key (1.4), Fill (0.6) | **DEFECT:** Cold directional fill, no HemisphereLight, dark ambient occlusion. |
| **3D Avatar Material** | `Ch28_body` (metallic=0.5) | **DEFECT:** Excessive metalness on skin causes unnatural light absorption and dark complexion. |

---

## 3. Discrepancy Breakdown: Trained vs. Untrained Signs

### The 17 Trained Classes (Supported by V6 Model Checkpoint):
1. `BATHROOM`
2. `DONT_UNDERSTAND`
3. `FOOD`
4. `GOODBYE`
5. `HELLO`
6. `HELP`
7. `HERE`
8. `HOSPITAL`
9. `LEFT`
10. `NO`
11. `PLEASE`
12. `REPEAT`
13. `RIGHT`
14. `THANK_YOU`
15. `WATER`
16. `WHERE`
17. `YES`

### The 32 UNTRAINED Signs (Zero Model Weights, Missing from Model):
```
 1. AMBULANCE        12. GOOD            23. POLICE
 2. BAD              13. HEARING         24. SCHOOL
 3. CALL             14. HOW             25. SORRY
 4. COME             15. HOW_ARE_YOU     26. STOP
 5. DANGER           16. I_ME            27. UNDERSTAND
 6. DEAF             17. MEDICINE        28. WAIT
 7. DOCTOR           18. MY              29. WANT
 8. EMERGENCY        19. NAME            30. WHAT
 9. FRIEND           20. NEED            31. WORK
10. GIVE             21. NICE_TO_MEET_YOU 32. YOU
11. GO               22. PAIN
```

When a user signs `DOCTOR`, `NAME`, `YOU`, `I_ME`, `STOP`, `WAIT`, `GOOD`, or `EMERGENCY`, the model cannot predict them. It collapses onto one of the 17 classes with highest spatial similarity.

---

## 4. Root Causes of Same Prediction for Different Signs

### Root Cause 1: Softmax Closed-Set Constraint on OOD Signs
Because the model has only 17 output neurons, an out-of-distribution (OOD) hand sign (e.g. `NAME`) is projected onto the 17-class subspace. Softmax exponentiation amplifies the highest logit, assigning 70%–95% confidence to an arbitrary class (e.g. `HERE` or `BATHROOM`), making it appear as a high-confidence recognition.

### Root Cause 2: Kinematic Heuristics in `inferenceWorker.ts`
Lines 400–575 of `inferenceWorker.ts` evaluate crude finger extension flags (`isOpenHand`, `isFist`, `isIndexPointing`):
- Any open hand where all fingers are extended assigns 0.93 to `HELLO` or 0.92 to `WHERE`.
- Any closed fist assigns 0.92 to `YES` or 0.94 to `HELP`.
- Any index pointing assigns 0.96 to `YOU` or `HERE`.
These heuristics blend into the neural prediction with a 40% weight (`scores * 0.4 + neuralProb * 0.6`). Even if the neural model predicts a distinct distribution, the heuristic pushes the score over the 0.60 threshold, causing identical predictions.

### Root Cause 3: Cached Probability Leaks during Static Poses
Lines 581–584 of `inferenceWorker.ts`:
```typescript
const canReuseCached = cachedNeuralProbs !== null && motionMag < 0.015 && (tTf0 - lastNeuralInferenceTime < 600);
```
When a signer holds a hand steadily before changing to a new sign, if the initial motion transition is slow ($\text{motionMag} < 0.015$), the worker reuses the previous prediction for up to 600 ms, making it appear unresponsive.

---

## 5. Root Causes of Avatar Repeating the Same Animation

In `apps/web/src/components/avatar/AnimationController.ts`:
1. `registerClips()` mapped multiple distinct vocabulary words to identical animations:
   - `DOCTOR` $\to$ `HOSPITAL`
   - `WANT` $\to$ `NEED`
   - `EAT` $\to$ `FOOD`
   - `DRINK` $\to$ `WATER`
   - `GOODBYE` $\to$ `GOING`
2. `createFallbackAction()` redirected missing clips to unrelated gestures:
   - `SORRY` $\to$ `PLEASE` (even though `SBV_SORRY` exists in the GLB!)
   - `GOOD` $\to$ `YES` (even though `SBV_GOOD` exists in the GLB!)
   - `BAD` $\to$ `NO`
   - `HOW_ARE_YOU` $\to$ `WHERE`
   - `YOU` $\to$ `HERE`
   - `I_ME` $\to$ `PLEASE`
   - `BUY` $\to$ `FOOD`
   - `WORK` $\to$ `SCHOOL`
3. Any sign not in the map defaulted to `IDLE`.

---

## 6. Root Causes of Avatar Visual Darkness

1. **Material Metalness Error:** `sign-avatar.glb` defines material `Ch28_body` with `metallicFactor = 0.5`. Human skin is non-metallic. In PBR rendering without an environment map, metallic=0.5 causes dark diffuse absorption.
2. **Missing Hemisphere Ambient Fill:** Studio lighting contains only directional lights and a simple ambient light. It lacks a `HemisphereLight` (sky/ground gradient), leaving shadows on the neck, chin, fingers, and lower body pitch-black.
3. **Low Exposure & Color Temperature:** `renderer.toneMappingExposure = 1.15` is underexposed for the dark `0x1e293b` background. Key and fill lights have unbalanced color temperatures.

---

## 7. Action Plan for SignBridge V7

1. **Phase 2:** Collect and curate authentic multi-signer ISL dataset covering the expanded 50-class vocabulary (including the 32 missing signs: `NAME`, `YOU`, `I_ME`, `DOCTOR`, `EMERGENCY`, `POLICE`, `STOP`, `WAIT`, `GOOD`, `BAD`, `COME`, `GO`, `HOW_ARE_YOU`, `WHAT`, `FRIEND`, etc.).
2. **Phase 3:** Train the 2-layer GRU with 50 output classes (`Dense(50, Softmax)`), balanced class weights, and train-only temporal augmentation. Export bit-exact TensorFlow.js model to `apps/web/public/models/sbv-gru-v1/`.
3. **Phase 4:** Purge the fragile kinematic heuristic overrides in `inferenceWorker.ts`. Calibrate confidence thresholds and add real-time debug HUD showing Top-5 predictions and landmark stability.
4. **Phase 5:** Build a verified one-to-one animation mapping in `AnimationController.ts`, removing false aliasing and leveraging all 28 native clips in `sign-avatar.glb` plus procedural gestures.
5. **Phase 6:** Fix avatar materials in `SignAvatar.tsx` (set `mat.metalness = 0.02`, `mat.roughness = 0.45`), add `HemisphereLight(0xfff7ed, 0x1e293b, 1.2)`, and raise tone mapping exposure to 1.35.
6. **Phase 7–10:** Execute rigorous regression tests, measure real hardware latencies, and produce all V7 deliverables.
