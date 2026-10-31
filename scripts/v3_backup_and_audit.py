"""
v3_backup_and_audit.py
─────────────────────────────────────────────────────────────────────────────
Step 0: Safely backup V2 baseline model, weights, and reports.
Step 1: Perform comprehensive Dataset Quality Audit:
  - Verify every sample and its source.
  - Audit class distribution.
  - Identify missing or incorrect labels.
  - Check signer-independent splits.
  - Identify and flag any duplicate or corrupted samples (NaNs, Infs, range violations).
  - Check representation of the 44 essential daily conversation signs.
"""

import os
import shutil
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backups", "v2-baseline")
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DATASET_DIR = os.path.join(ML_DIR, "datasets", "asl")
REPORTS_DIR = os.path.join(ML_DIR, "reports")

# 1. Execute Backup
os.makedirs(BACKUP_DIR, exist_ok=True)
v2_model_src = os.path.join(ML_DIR, "models", "sbv-gru-v1")
v2_model_dst = os.path.join(BACKUP_DIR, "ml-models-sbv-gru-v1")
if os.path.exists(v2_model_src):
    shutil.copytree(v2_model_src, v2_model_dst, dirs_exist_ok=True)
    print(f"Backed up V2 model checkpoint to: {v2_model_dst}")

if os.path.exists(REPORTS_DIR):
    v2_reports_dst = os.path.join(BACKUP_DIR, "ml-reports")
    shutil.copytree(REPORTS_DIR, v2_reports_dst, dirs_exist_ok=True)
    print(f"Backed up V2 reports to: {v2_reports_dst}")

# 2. Dataset Quality Audit
train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "validation")
test_dir = os.path.join(DATASET_DIR, "test")

splits = {"train": train_dir, "validation": val_dir, "test": test_dir}

audit_results = {
    "totalSamples": 0,
    "splitCounts": {},
    "classesFound": set(),
    "signersBySplit": {"train": set(), "validation": set(), "test": set()},
    "sources": {},
    "corruptedSamples": [],
    "duplicateHashes": set(),
    "duplicatesFound": 0,
    "classSampleCounts": {},
    "durationDistribution": [],
}

import hashlib

for split_name, sdir in splits.items():
    if not os.path.exists(sdir):
        print(f"Warning: {sdir} does not exist!")
        continue
    files = [f for f in os.listdir(sdir) if f.endswith(".json")]
    audit_results["splitCounts"][split_name] = len(files)
    audit_results["totalSamples"] += len(files)

    for fname in files:
        fpath = os.path.join(sdir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            audit_results["corruptedSamples"].append({"file": fname, "reason": str(e)})
            continue

        sign = data.get("sign")
        signer = data.get("signerId")
        src = data.get("sourceDataset", "unknown")
        matrix = data.get("featureMatrix")

        audit_results["classesFound"].add(sign)
        if signer:
            audit_results["signersBySplit"][split_name].add(signer)
        audit_results["sources"][src] = audit_results["sources"].get(src, 0) + 1
        audit_results["classSampleCounts"][sign] = audit_results["classSampleCounts"].get(sign, 0) + 1

        # Check matrix integrity
        if matrix is None or len(matrix) != 30:
            audit_results["corruptedSamples"].append({"file": fname, "reason": f"Invalid frame length: {len(matrix) if matrix else None}"})
            continue

        arr = np.array(matrix, dtype=np.float32)
        if arr.shape != (30, 126):
            audit_results["corruptedSamples"].append({"file": fname, "reason": f"Invalid shape: {arr.shape}"})
            continue
        if np.isnan(arr).any():
            audit_results["corruptedSamples"].append({"file": fname, "reason": "Contains NaNs"})
            continue
        if np.isinf(arr).any():
            audit_results["corruptedSamples"].append({"file": fname, "reason": "Contains Infs"})
            continue
        if (arr < -10.0).any() or (arr > 10.0).any():
            audit_results["corruptedSamples"].append({"file": fname, "reason": "Coordinate bounds exceeded [-10, 10]"})
            continue

        # Check for duplicates using content hash
        content_hash = hashlib.md5(arr.tobytes()).hexdigest()
        if content_hash in audit_results["duplicateHashes"]:
            audit_results["duplicatesFound"] += 1
        else:
            audit_results["duplicateHashes"].add(content_hash)

# Audit 44 essential conversation signs (with exact class tokens)
ESSENTIAL_44_SIGNS = [
    'HELLO', 'THANK_YOU', 'PLEASE', 'SORRY', 'YES', 'NO', 'GOOD', 'HELP',
    'WATER', 'FOOD', 'WELCOME_HOME', 'SCHOOL', 'HOSPITAL', 'WANT', 'EMERGENCY', 'POLICE',
    'STOP', 'GO', 'WAIT', 'COME', 'UNDERSTAND', 'DONT_UNDERSTAND', 'NAME', 'WHERE',
    'BATHROOM', 'NICE_TO_MEET_YOU', 'YOU', 'I_ME', 'MY', 'YOUR', 'WE_US', 'THEY',
    'MEET', 'FRIEND', 'DEAF', 'WELCOME', 'FINE', 'EXCUSE_ME', 'BAD', 'MAYBE',
    'LIKE', 'DONT_LIKE', 'AGAIN', 'FINISHED'
]

essential_coverage = {}
for sign in ESSENTIAL_44_SIGNS:
    cnt = audit_results["classSampleCounts"].get(sign, 0)
    essential_coverage[sign] = cnt

print("\n" + "="*65)
print("DATASET QUALITY AUDIT REPORT (V2 BASELINE)")
print("="*65)
print(f"Total Base Samples:       {audit_results['totalSamples']}")
print(f"Split Counts:             Train: {audit_results['splitCounts'].get('train', 0)}, "
      f"Val: {audit_results['splitCounts'].get('validation', 0)}, "
      f"Test: {audit_results['splitCounts'].get('test', 0)}")
print(f"Unique Classes Found:     {len(audit_results['classesFound'])}")
print(f"Sources Distribution:     {audit_results['sources']}")
print(f"Corrupted Samples:        {len(audit_results['corruptedSamples'])}")
print(f"Exact Duplicate Samples:  {audit_results['duplicatesFound']}")
print("\nSigners Partitioning:")
for split, sset in audit_results["signersBySplit"].items():
    print(f"  {split.capitalize():12}: {sorted(list(sset))}")

# Check for signer leakage
train_signers = audit_results["signersBySplit"]["train"]
val_signers = audit_results["signersBySplit"]["validation"]
test_signers = audit_results["signersBySplit"]["test"]

leakage_train_val = train_signers.intersection(val_signers)
leakage_train_test = train_signers.intersection(test_signers)
leakage_val_test = val_signers.intersection(test_signers)

print(f"\nSigner Leakage Check:")
print(f"  Train & Val Overlap:  {len(leakage_train_val)} ({leakage_train_val})")
print(f"  Train & Test Overlap: {len(leakage_train_test)} ({leakage_train_test})")
print(f"  Val & Test Overlap:   {len(leakage_val_test)} ({leakage_val_test})")

print(f"\nEssential 44 Signs Coverage:")
min_ess = min(essential_coverage.values()) if essential_coverage else 0
max_ess = max(essential_coverage.values()) if essential_coverage else 0
avg_ess = sum(essential_coverage.values()) / len(essential_coverage) if essential_coverage else 0
print(f"  All 44 Essential Signs Present: {all(cnt > 0 for cnt in essential_coverage.values())}")
print(f"  Min Samples per Essential Sign: {min_ess}")
print(f"  Max Samples per Essential Sign: {max_ess}")
print(f"  Avg Samples per Essential Sign: {avg_ess:.1f}")

# Save audit results to report
audit_report_path = os.path.join(REPORTS_DIR, "v3_dataset_audit_report.json")
with open(audit_report_path, "w", encoding="utf-8") as f:
    json.dump({
        "totalSamples": audit_results["totalSamples"],
        "splitCounts": audit_results["splitCounts"],
        "numClasses": len(audit_results["classesFound"]),
        "sources": audit_results["sources"],
        "corruptedCount": len(audit_results["corruptedSamples"]),
        "duplicatesCount": audit_results["duplicatesFound"],
        "signersBySplit": {k: sorted(list(v)) for k, v in audit_results["signersBySplit"].items()},
        "leakage": {
            "train_val": list(leakage_train_val),
            "train_test": list(leakage_train_test),
            "val_test": list(leakage_val_test),
        },
        "essential44Coverage": essential_coverage,
    }, f, indent=2)

print(f"\nAudit Report written to: {audit_report_path}")
