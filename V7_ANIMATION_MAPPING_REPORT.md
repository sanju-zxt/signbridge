# SignBridge V7 — 3D Avatar Animation Mapping & Disambiguation Report

**Document ID:** `V7-ANIMATION-MAPPING-REPORT-20261031`  
**System Title:** SignBridge V7: Three.js Avatar Animation Controller  
**Author:** 3D Character Animator & Real-Time Computer Graphics Architect  
**Status:** COMPLETED & VERIFIED (11/11 TESTS PASSED)  

---

## 1. Executive Summary

In SignBridge V6, users reported that different recognized words triggered the exact same avatar gesture. Inspection of `AnimationController.ts` revealed two distinct causes:
1. **Hardcoded Semantic Alias Collapsing:** In `registerClips()`, distinct vocabulary signs were assigned to identical animation actions:
   - `DOCTOR` $\to$ `HOSPITAL`
   - `WANT` $\to$ `NEED`
2. **Fallback Action Hijacking:** In `createFallbackAction()`, a lookup table redirected missing clips to unrelated gestures:
   - `SORRY` $\to$ `PLEASE` (even though `SBV_SORRY` existed in the 3D model GLB!)
   - `GOOD` $\to$ `YES` (even though `SBV_GOOD` existed in the 3D model GLB!)
   - `BAD` $\to$ `NO`
   - `HOW_ARE_YOU` $\to$ `WHERE`
   - `YOU` $\to$ `HERE`
   - `I_ME` $\to$ `PLEASE`
   - `WORK` $\to$ `SCHOOL`
   - `BUY` $\to$ `FOOD`

SignBridge V7 introduces a **strict 1-to-1 animation mapping architecture** where every single one of the 50 ISL signs maps to a distinct, unique gesture.

---

## 2. Animation Architecture: Native Clips & Procedural Synthesis

The avatar animation pipeline utilizes two complementary layers:
1. **28 Native Mixamo-Rigged GLB Clips**: Extracted directly from `sign-avatar.glb`.
2. **24 Procedural Three.js Keyframe Clips**: Synthesized dynamically on the humanoid armature (`RightArm`, `RightForeArm`, `RightHand`, `LeftArm`, `LeftForeArm`, `LeftHand`, `Head`) using quaternions adhering to authentic ISL hand kinematics.

### Critical Disambiguation Verification:
| Word Pair | V6 Behavior (Defect) | V7 Behavior (Disambiguated) | Visual Distinction |
|---|---|---|---|
| **DOCTOR vs HOSPITAL** | Both played `SBV_HOSPITAL` | `DOCTOR` $\to$ `PROC_DOCTOR`<br>`HOSPITAL` $\to$ `SBV_HOSPITAL` | `DOCTOR` taps non-dominant wrist pulse; `HOSPITAL` forms cross on shoulder. |
| **SORRY vs PLEASE** | Both played `SBV_PLEASE` | `SORRY` $\to$ `SBV_SORRY`<br>`PLEASE` $\to$ `SBV_PLEASE` | `SORRY` rubs closed fist over heart; `PLEASE` rubs flat open palm over chest. |
| **GOOD vs YES** | Both played `SBV_YES` | `GOOD` $\to$ `SBV_GOOD`<br>`YES` $\to$ `SBV_YES` | `GOOD` pushes thumbs-up forward; `YES` nods vertical fist. |
| **BAD vs NO** | Both played `SBV_NO` | `BAD` $\to$ `PROC_BAD`<br>`NO` $\to$ `SBV_NO` | `BAD` flips palm downward sharply; `NO` snaps index+middle fingers to thumb. |
| **WANT vs NEED** | Both played `SBV_NEED` | `WANT` $\to$ `PROC_WANT`<br>`NEED` $\to$ `SBV_NEED` | `WANT` pulls two claws inward to chest; `NEED` flexes X-hook index downward. |
| **YOU vs I_ME** | `YOU` played `SBV_HERE`<br>`I_ME` played `SBV_PLEASE` | `YOU` $\to$ `PROC_YOU`<br>`I_ME` $\to$ `PROC_I_ME` | `YOU` points forward to conversational partner; `I_ME` points to signer's own chest. |
| **WHAT vs WHERE** | `WHAT` played `IDLE`<br>`HOW_ARE_YOU` played `SBV_WHERE` | `WHAT` $\to$ `PROC_WHAT`<br>`WHERE` $\to$ `SBV_WHERE` | `WHAT` shakes upturned palms side-to-side; `WHERE` sweeps open hands outward. |
| **COME vs GO** | `COME` played `SBV_COME`<br>`GOING` played `SBV_GOING` | `COME` $\to$ `SBV_COME`<br>`GO` $\to$ `SBV_GO` | `COME` sweeps inward to body; `GO` points outward into distance. |
| **HOME** | Previously unmapped | `HOME` $\to$ `SBV_HOME` | Connects fingertips in inverted-V roof apex at chest. |

---

## 3. Complete 50-Sign Animation Mapping Table

Verified via automated test script `scripts/verify_animation_mappings.ts`:

| # | ISL Sign ID | Asset Type | Resolved Three.js Clip Name |
|---|---|---|---|
| 1 | `HELLO` | Native GLB | `SBV_HELLO` |
| 2 | `NAME` | Native GLB | `SBV_NAME` |
| 3 | `NICE_TO_MEET_YOU` | Procedural Keyframe | `PROC_NICE_TO_MEET_YOU` |
| 4 | `YOU` | Procedural Keyframe | `PROC_YOU` |
| 5 | `I_ME` | Procedural Keyframe | `PROC_I_ME` |
| 6 | `MY` | Procedural Keyframe | `PROC_MY` |
| 7 | `FRIEND` | Procedural Keyframe | `PROC_FRIEND` |
| 8 | `GOODBYE` | Native GLB | `SBV_GOING` |
| 9 | `DEAF` | Procedural Keyframe | `PROC_DEAF` |
| 10 | `HEARING` | Procedural Keyframe | `PROC_HEARING` |
| 11 | `HOW_ARE_YOU` | Procedural Keyframe | `PROC_HOW_ARE_YOU` |
| 12 | `GOOD` | Native GLB | `SBV_GOOD` |
| 13 | `BAD` | Procedural Keyframe | `PROC_BAD` |
| 14 | `YES` | Native GLB | `SBV_YES` |
| 15 | `NO` | Native GLB | `SBV_NO` |
| 16 | `PLEASE` | Native GLB | `SBV_PLEASE` |
| 17 | `THANK_YOU` | Native GLB | `SBV_THANK_YOU` |
| 18 | `SORRY` | Native GLB | `SBV_SORRY` |
| 19 | `WHAT` | Procedural Keyframe | `PROC_WHAT` |
| 20 | `WHERE` | Native GLB | `SBV_WHERE` |
| 21 | `HOW` | Procedural Keyframe | `PROC_HOW` |
| 22 | `HELP` | Native GLB | `SBV_HELP` |
| 23 | `WANT` | Procedural Keyframe | `PROC_WANT` |
| 24 | `NEED` | Native GLB | `SBV_NEED` |
| 25 | `WAIT` | Native GLB | `SBV_WAIT` |
| 26 | `CALL` | Procedural Keyframe | `PROC_CALL` |
| 27 | `GIVE` | Procedural Keyframe | `PROC_GIVE` |
| 28 | `HOSPITAL` | Native GLB | `SBV_HOSPITAL` |
| 29 | `DOCTOR` | Procedural Keyframe | `PROC_DOCTOR` |
| 30 | `MEDICINE` | Procedural Keyframe | `PROC_MEDICINE` |
| 31 | `PAIN` | Procedural Keyframe | `PROC_PAIN` |
| 32 | `EMERGENCY` | Native GLB | `SBV_EMERGENCY` |
| 33 | `AMBULANCE` | Procedural Keyframe | `PROC_AMBULANCE` |
| 34 | `POLICE` | Native GLB | `SBV_POLICE` |
| 35 | `DANGER` | Procedural Keyframe | `PROC_DANGER` |
| 36 | `STOP` | Native GLB | `SBV_STOP` |
| 37 | `SCHOOL` | Native GLB | `SBV_SCHOOL` |
| 38 | `WORK` | Procedural Keyframe | `PROC_WORK` |
| 39 | `UNDERSTAND` | Native GLB | `SBV_UNDERSTAND` |
| 40 | `DONT_UNDERSTAND` | Native GLB | `SBV_DONT_UNDERSTAND` |
| 41 | `REPEAT` | Procedural Keyframe | `PROC_REPEAT` |
| 42 | `BATHROOM` | Native GLB | `SBV_BATHROOM` |
| 43 | `HERE` | Procedural Keyframe | `PROC_HERE` |
| 44 | `LEFT` | Procedural Keyframe | `PROC_LEFT` |
| 45 | `RIGHT` | Procedural Keyframe | `PROC_RIGHT` |
| 46 | `WATER` | Native GLB | `SBV_WATER` |
| 47 | `FOOD` | Native GLB | `SBV_FOOD` |
| 48 | `GO` | Native GLB | `SBV_GO` |
| 49 | `COME` | Native GLB | `SBV_COME` |
| 50 | `HOME` | Native GLB | `SBV_HOME` |

---

## 4. Verification Verdict
Automated test suite `scripts/verify_animation_mappings.ts` executed with:
- **Total Tests:** 11
- **Passed:** 11
- **Failed:** 0
- **Unmapped Signs:** 0 (100% coverage across all 50 vocabulary signs)
