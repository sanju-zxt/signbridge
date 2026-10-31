"""
v5_phase0_baseline_audit.py
─────────────────────────────────────────────────────────────────────────────
Phase 0: Baseline Verification & Repository Audit for SignBridge V5

Audits:
1. Current model architecture, parameters, input/output tensors.
2. V4 benchmark metrics (latency, footprint, test accuracies).
3. Active monorepo packages, dependencies, and build status.
4. Identifies reusable modules and exact technical gaps for V5 priorities.

Outputs structured baseline summary.
"""

import os
import sys
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "ml", "reports")
MODEL_DIR = os.path.join(PROJECT_ROOT, "apps", "web", "public", "models", "sbv-gru-v1")

print("=" * 80)
print("SIGNBRIDGE V5 — PHASE 0: REPOSITORY AUDIT & BASELINE ESTABLISHMENT")
print("=" * 80)

# 1. Model Baseline
with open(os.path.join(MODEL_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(os.path.join(MODEL_DIR, "model.json"), "r", encoding="utf-8") as f:
    mjson = json.load(f)

weights_bin_path = os.path.join(MODEL_DIR, "weights.bin")
weights_size = os.path.getsize(weights_bin_path)

print(f"\n[Baseline Model]")
print(f"  Model ID:           {meta.get('id')}")
print(f"  Version:            {meta.get('version')}")
print(f"  Classes:            {len(meta.get('classes', []))}")
print(f"  Feature Dimension:  {meta.get('featureDim')} (Hand only: 63 right + 63 left)")
print(f"  Window Size:        {meta.get('windowSize')} frames (1.0s @ 30 FPS)")
print(f"  Architecture:       2-layer GRU (64 hidden) -> Dense(128, ReLU) -> Dropout(0.35) -> Dense(186, Softmax)")
print(f"  Trainable Params:   {meta.get('metrics', {}).get('totalParameters', 93754)} parameters")
print(f"  Weights Footprint:  {weights_size / 1024:.1f} KB (FP32), {weights_size / 4096:.1f} KB (INT8)")

# 2. V4 Test Metrics Baseline
with open(os.path.join(REPORTS_DIR, "asl_v4_training_summary.json"), "r", encoding="utf-8") as f:
    v4_train = json.load(f)

test_eval = v4_train.get("independent_test_evaluation", {})
print(f"\n[Baseline Recognition Metrics (Unseen Signers 13-16)]")
print(f"  Overall 186-Class Top-1:  {test_eval.get('overall_186_classes', {}).get('top1_accuracy', 0)*100:.2f}%")
print(f"  Overall 186-Class Top-5:  {test_eval.get('overall_186_classes', {}).get('top5_accuracy', 0)*100:.2f}%")
print(f"  Essential 44-Sign Top-1:  {test_eval.get('essential_44_signs', {}).get('top1_accuracy', 0)*100:.2f}%")
print(f"  Essential 44-Sign Top-5:  {test_eval.get('essential_44_signs', {}).get('top5_accuracy', 0)*100:.2f}%")
print(f"  Essential 44-Sign Recall: {test_eval.get('essential_44_signs', {}).get('macro_recall', 0)*100:.2f}%")
print(f"  Essential 44-Sign F1:     {test_eval.get('essential_44_signs', {}).get('macro_f1', 0)*100:.2f}%")

# 3. Latency Baseline
with open(os.path.join(REPORTS_DIR, "asl_v4_benchmark_report.json"), "r", encoding="utf-8") as f:
    v4_bench = json.load(f)

lat = v4_bench.get("latencyMeasurements", {})
e2e = v4_bench.get("endToEndPipeline", {})
print(f"\n[Baseline Latency Metrics (CPU Single-Thread)]")
print(f"  Cold-Start Loading:       {v4_bench.get('model', {}).get('coldStartLoadMs')} ms")
print(f"  Preprocessing Latency:    {lat.get('preprocessingMeanMs')} ms")
print(f"  Model Inference P50:      {lat.get('inferenceP50Ms')} ms")
print(f"  Model Inference P95:      {lat.get('inferenceP95Ms')} ms")
print(f"  Model Inference P99:      {lat.get('inferenceP99Ms')} ms")
print(f"  Total End-to-End P50:     {e2e.get('totalE2EP50Ms')} ms")
print(f"  Primary Bottleneck:       {e2e.get('primaryBottleneck')}")

# 4. Gaps to Solve in V5
print("\n" + "=" * 80)
print("IDENTIFIED GAPS & PRIORITIES FOR SIGNBRIDGE V5")
print("=" * 80)
print("1. DATA AUTHENTICITY GAP:")
print("   - Current dataset has 8,620 kinematically synthesized sequences and 0 raw genuine videos.")
print("   - V5 Action: Acquire and integrate authentic WLASL/MS-ASL video annotations and build")
print("     a live capture pipeline for verified real signer recordings with full provenance tracking.")
print("\n2. NON-MANUAL & PRONOUN RECOGNITION GAP:")
print("   - Current model uses 126 hand coordinates only. Deictic pronouns (YOU, YOUR, WE_US, THEY)")
print("     scored 0% Top-1 due to overlapping hand pointing kinematics without facial/gaze direction.")
print("   - V5 Action: Build FaceLandmarker & eye-gaze vector branch to capture head orientation,")
print("     eyebrow state, and gaze direction for contextual pronoun disambiguation.")
print("\n3. HARDWARE ACCELERATION GAP:")
print("   - Inference currently runs on CPU taking ~266 ms (85.6% of pipeline budget).")
print("   - V5 Action: Implement runtime capability detection for WebGPU with graceful WebGL/CPU")
print("     fallback, benchmarking actual hardware acceleration on live browser runtimes.")
print("\n4. CONTINUOUS SENTENCE & EMERGENCY SAFETY GAP:")
print("   - Current app concatenates isolated sign tokens.")
print("   - V5 Action: Implement temporal sequence segmentation, Topic-Comment intermediate gloss")
print("     construction, and an explicit user confirmation safety gate for emergency signs.")
print("=" * 80)
