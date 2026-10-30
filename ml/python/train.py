"""
ISL Neural Network Training Pipeline
SignBridge2Vision
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models import ISL_GRU_Classifier, ISL_LSTM_Classifier
from dataset_loader import ISLDataset, generate_benchmark_dataset, ISL_CLASSES, FEATURE_DIM, SEQUENCE_LENGTH

def parse_args():
    parser = argparse.ArgumentParser(description="Train ISL Temporal Sequence Classifier")
    parser.add_argument("--dataset", type=str, default="ml/datasets/processed", help="Path to processed dataset")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=30, help="Temporal window length")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm"], help="Model architecture")
    parser.add_argument("--output", type=str, default="ml/models/sbv-gru-v1", help="Output directory for model checkpoints")
    return parser.parse_args()


def train():
    args = parse_args()
    print("=" * 60)
    print("SIGNBRIDGE2VISION — ISL MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Model Architecture: {args.model.upper()}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Sequence Length:    {args.sequence_length}")
    print(f"Learning Rate:      {args.learning_rate}")
    print(f"Num Classes:        {len(ISL_CLASSES)} ({', '.join(ISL_CLASSES)})")
    print("=" * 60)

    # 1. Load / Generate Dataset with Signer-Independent Split
    all_samples = generate_benchmark_dataset(num_signers=4, samples_per_class=60)
    print(f"Total sequences generated: {len(all_samples)}")

    # Split by signer: train on signer_001, signer_002, signer_003; test on signer_004
    train_samples = [s for s in all_samples if s[2] in ["signer_001", "signer_002"]]
    val_samples   = [s for s in all_samples if s[2] == "signer_003"]
    test_samples  = [s for s in all_samples if s[2] == "signer_004"]

    print(f"Signer Split -> Train: {len(train_samples)} (signer_001, 002) | Val: {len(val_samples)} (signer_003) | Test: {len(test_samples)} (signer_004)")

    train_dataset = ISLDataset(train_samples, augment=True)
    val_dataset   = ISLDataset(val_samples, augment=False)
    test_dataset  = ISLDataset(test_samples, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Instantiate Model
    if args.model == "gru":
        model = ISL_GRU_Classifier(
            input_dim=FEATURE_DIM,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(ISL_CLASSES),
            dropout=0.3
        )
    else:
        model = ISL_LSTM_Classifier(
            input_dim=FEATURE_DIM,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(ISL_CLASSES),
            dropout=0.3
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_weights_path = os.path.join(args.output, "model_best.pt")
    os.makedirs(args.output, exist_ok=True)

    print("\nStarting Training Epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1).numpy()
            train_preds.extend(preds)
            train_targets.extend(y_batch.numpy())

        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                v_logits = model(x_val)
                v_loss = criterion(v_logits, y_val)
                val_loss += v_loss.item() * x_val.size(0)
                v_preds = torch.argmax(v_logits, dim=1).numpy()
                val_preds.extend(v_preds)
                val_targets.extend(y_val.numpy())

        val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s | Best Val Accuracy: {best_val_acc*100:.2f}%")

    # 3. Final Evaluation on Unseen Test Signer (signer_004)
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            t_logits = model(x_test)
            t_preds = torch.argmax(t_logits, dim=1).numpy()
            test_preds.extend(t_preds)
            test_targets.extend(y_test.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(test_targets, test_preds)

    print("\n" + "=" * 60)
    print("UNSEEN TEST SIGNER (SIGNER_004) EVALUATION REPORT")
    print("=" * 60)
    print(f"Signer-Independent Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"Weighted Precision:                {prec * 100:.2f}%")
    print(f"Weighted Recall:                   {recall * 100:.2f}%")
    print(f"Weighted F1 Score:                 {f1 * 100:.2f}%")
    print("=" * 60)

    # 4. Export ONNX model
    dummy_input = torch.randn(1, args.sequence_length, FEATURE_DIM)
    onnx_path = os.path.join(args.output, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["landmarks_sequence"],
        output_names=["probabilities"],
        dynamic_axes={"landmarks_sequence": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        opset_version=14
    )
    print(f"Exported ONNX model to: {onnx_path}")

    # 5. Save Metadata
    meta = {
        "modelId": f"SBV-{args.model.upper()}-v1",
        "framework": "PyTorch / ONNX",
        "architecture": f"Bidirectional {args.model.upper()} with Attention",
        "sequenceLength": args.sequence_length,
        "featureDim": FEATURE_DIM,
        "numClasses": len(ISL_CLASSES),
        "classes": ISL_CLASSES,
        "metrics": {
            "testAccuracy": round(float(test_acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(recall), 4),
            "f1Score": round(float(f1), 4),
            "signerIndependent": True,
            "testSigner": "signer_004"
        },
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model metadata to {os.path.join(args.output, 'metadata.json')}")
    print("=" * 60)

if __name__ == "__main__":
    train()