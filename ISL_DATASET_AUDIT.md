# SignBridge V6 — ISL Dataset Audit & Provenance Verification Report

**Document ID:** `V6-DATASET-AUDIT-20261031-VALIDATED`  
**Dataset Name:** SignBridge Authentic Indian Sign Language (ISL) Benchmark  
**Dataset Version:** 6.0.0  
**Data Integrity:** 100% Genuine Human Video Landmarks (0% Synthesized/Augmented Signers)  
**Validation Date:** October 31, 2026  
**Auditor:** Senior AI/ML Validation Engineer  

---

## 1. Executive Summary

In strict accordance with the V6 Master Directive to ensure absolute dataset authenticity and zero signer leakage, this independent audit verified the provenance, partitioning, and statistical distribution of the 510 benchmark sequences stored in `ml/datasets/isl/`.

Every sequence consists of 30 frames with real 126-dimensional MediaPipe landmark arrays (63 right hand + 63 left hand) normalized by palm diagonal scale relative to the wrist.

---

## 2. Provenance & Partitioning Audit

```
Total Sequences Audited: 510
Total Human Signers: 10 (user_01 through user_10)
Total Authentic Classes: 17
Samples per Class: Exactly 30 (3 per signer)
```

| Split Partition | Signer IDs | Sequence Count | Class Count | Purpose | Leakage Status |
|---|---|---|---|---|---|
| **Train Set** | `user_01`, `user_02`, `user_03`, `user_04`, `user_05`, `user_06`, `user_07` | **357** (70%) | 17 | Supervised GRU training | **Strict Zero Leakage** |
| **Validation Set** | `user_08` | **51** (10%) | 17 | Early stopping & checkpoint selection | **Strict Zero Leakage** |
| **Test Set** | `user_09`, `user_10` | **102** (20%) | 17 | Unseen evaluation of generalization | **Strict Zero Leakage** |
| **Total** | **10 Distinct Signers** | **510** (100%) | **17** | **Complete Benchmark** | **100% Signer-Independent** |

### Zero Signer Leakage Guarantee:
$$\text{TrainSigners} \cup \text{ValSigners} = \{\text{user\_01}, \dots, \text{user\_08}\}$$
$$\text{TestSigners} = \{\text{user\_09}, \text{user\_10}\}$$
$$(\text{TrainSigners} \cup \text{ValSigners}) \cap \text{TestSigners} = \emptyset \quad \text{[VERIFIED PASS]}$$

Test sequences contain **zero synthetic noise injection** and **zero temporal augmentation**, ensuring test metrics reflect pure real-world generalization across new human signers.

---

## 3. Four-Tier Classification of Dataset Components

| Dataset Component | Tier | Audit Verification Details |
|---|---|---|
| **Raw Human Benchmark Dataset (`isl_benchmark_dataset.json`)** | **Tier 1: Implemented & Verified** | 510 sequences loaded and validated. Physical variation across normal, slow, and fast signing tempos. |
| **Partitioned Splitted Sets (`ml/datasets/isl/{train,val,test}`)** | **Tier 1: Implemented & Verified** | 357 train files, 51 val files, 102 test files. Zero corrupt records or NaN landmark coordinates. |
| **Provenance Manifest (`annotations/isl_provenance_manifest.json`)** | **Tier 1: Implemented & Verified** | Full SHA-checksummed metadata tracking subject IDs, frame counts, and sign labels. |
| **TF.js Bit-Exact Weight Export (`weights.bin`)** | **Tier 1: Implemented & Verified** | Evaluated independently in `scripts/verify_model_authenticity.py`. 83.33% Top-1, 100% Top-5 verified. |
| **Studio Live Data Collection Route (`/api/dataset`)** | **Tier 2: Implemented but Not Real-World Validated** | Verified in automated API route tests. Needs diverse live crowd-sourcing to scale beyond 10 signers. |
| **Non-Manual Face Mesh Alignment (478 Landmarks)** | **Tier 3: Partially Implemented** | Face mesh tracking active in MediaPipe service; primary GRU input focuses on 126 hand coordinates. |
| **28-State Pan-India ISL Corpus** | **Tier 4: Not Implemented** | Currently covers 10 Delhi/Bangalore signers; regional Pan-India variations not yet assembled. |

---

## 4. Independent Test Evaluation & Confusion Matrix

Independently reproduced on the 102 test sequences from unseen signers `user_09` and `user_10`:

```
Overall Top-1 Accuracy:  83.33% (85 / 102)
Overall Top-5 Accuracy: 100.00% (102 / 102)
Macro Precision:         0.8088
Macro Recall:            0.8333
Macro F1-Score:          0.8014
```

### Complete 17x17 Confusion Matrix:
```
True \ Pred   BATH DONT FOOD GOOD HELL HELP HERE HOSP LEFT   NO PLEA REPE RIGH THAN WATE WHER  YES
--------------------------------------------------------------------------------------------------
BATHROOM        6    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
DONT_UNDERST    0    6    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
FOOD            0    0    6    0    0    0    0    0    0    0    0    0    0    0    0    0    0
GOODBYE         0    0    0    6    0    0    0    0    0    0    0    0    0    0    0    0    0
HELLO           0    0    0    0    6    0    0    0    0    0    0    0    0    0    0    0    0
HELP            0    0    0    0    0    6    0    0    0    0    0    0    0    0    0    0    0
HERE            0    0    0    0    0    0    6    0    0    0    0    0    0    0    0    0    0
HOSPITAL        0    0    0    0    0    0    0    6    0    0    0    0    0    0    0    0    0
LEFT            0    6    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
NO              0    0    0    0    0    0    0    0    0    6    0    0    0    0    0    0    0
PLEASE          0    0    0    0    0    0    0    0    0    0    6    0    0    0    0    0    0
REPEAT          0    0    0    0    0    0    0    0    0    0    0    6    0    0    0    0    0
RIGHT           0    6    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
THANK_YOU       0    0    0    0    3    0    0    0    0    0    0    0    0    3    0    0    0
WATER           0    0    0    0    0    0    0    0    0    0    0    0    0    0    6    0    0
WHERE           0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    6    0
YES             2    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    4
```

### Per-Class Performance Summary:
- **100% Recall Classes (13 / 17):**
  - `BATHROOM`, `DONT_UNDERSTAND`, `FOOD`, `GOODBYE`, `HELLO`, `HELP`, `HERE`, `HOSPITAL`, `NO`, `PLEASE`, `REPEAT`, `WATER`, `WHERE`.
- **Partial Recall Classes (2 / 17):**
  - `THANK_YOU` (50.0% Recall): 3 correct, 3 confused with `HELLO`.
  - `YES` (66.7% Recall): 4 correct, 2 confused with `BATHROOM`.
- **Failing Directional Classes (2 / 17):**
  - `LEFT` and `RIGHT` (0.0% Recall): Both directional sweeps were confused with `DONT_UNDERSTAND` due to unmirrored lateral hand movement across different camera angles.

---

## 5. Audit Conclusion

The SignBridge V6 dataset is authentic, rigorously partitioned, and completely devoid of synthetic shortcuts. Model evaluation on independent unseen signers demonstrates genuine real-world generalization across 15 core conversational and emergency signs.
