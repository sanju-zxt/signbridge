"""
ISL Model Evaluator & Latency Benchmark
SignBridge2Vision
"""

import os
import time
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from models import ISL_GRU_Classifier
from dataset_loader import generate_benchmark_dataset, ISL_CLASSES, FEATURE_DIM, SEQUENCE_LENGTH

def evaluate_model(model_dir="ml/models/sbv-gru-v1"):
    print("=" * 60)
    print("SIGNBRIDGE2VISION — MODEL EVALUATION & LATENCY BENCHMARK")
    print("=" * 60)

    weights_path = os.path.join(model_dir, "model_best.pt")
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return

    model = ISL_GRU_Classifier(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(ISL_CLASSES),
        dropout=0.0
    ) 
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    # Generate independent test samples
    samples = generate_benchmark_dataset(num_signers=4, samples_per_class=40)
    test_samples = [s for s in samples if s[2] == "signer_004"]

    x_test = torch.tensor(np.array([s[0] for s in test_samples]), dtype=torch.float32)
    y_test = np.array([s[1] for s in test_samples])

    print(f"Evaluating {len(test_samples)} test sequences from independent signer_004...")

    # Measure latency across 100 repeated single-sequence inferences
    latencies = []
    single_input = x_test[0:1]
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(single_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    avg_latency = np.mean(latencies[5:])  # ignore warmups
    p95_latency = np.percentile(latencies[5:], 95)

    # Classification Report
    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1).numpy()

    report = classification_report(y_test, preds, target_names=ISL_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print("\nPer-Class Performance Metrics:")
    print("-" * 60)
    print(f"{'Class Name':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    for c in ISL_CLASSES:
        m = report.get(c, {})
        print(f"{c:<18} | {m.get('precision', 0)*100:6.1f}%    | {m.get('recall', 0)*100:6.1f}%    | {m.get('f1-score', 0)*100:6.1f}%")
    print("-" * 60)

    print(f"\nOverall Test Accuracy:        {report['accuracy']*100:.2f}%")
    print(f"Average Inference Latency:   {avg_latency:.2f} ms")
    print(f"95th Percentile Latency:     {p95_latency:.2f} ms")
    print("=" * 60)

    # Save benchmark report to evaluation folder
    eval_dir = "ml/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    report_out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accuracy": round(report['accuracy'], 4),
        "macro_f1": round(report['macro avg']['f1-score'], 4),
        "weighted_f1": round(report['weighted avg']['f1-score'], 4),
        "avg_latency_ms": round(float(avg_latency), 2),
        "p95_latency_ms": round(float(p95_latency), 2),
        "per_class": {c: {k: round(v, 4) for k, v in report[c].items()} for c in ISL_CLASSES}
    }
    with open(os.path.join(eval_dir, "benchmark_report.json"), "w") as f:
        json.dump(report_out, f, indent=2)

    print(f"Saved evaluation report to {os.path.join(eval_dir, 'benchmark_report.json')}")

if __name__ == "__main__":
    evaluate_model()
