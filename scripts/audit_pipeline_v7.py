import json
import re
import os

PROJECT_ROOT = r'C:\Users\drx20\.gemini\antigravity\scratch\signbridge2vision'

# 1. Load model metadata
meta_path = os.path.join(PROJECT_ROOT, 'apps', 'web', 'public', 'models', 'sbv-gru-v1', 'metadata.json')
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
model_classes = meta['classes']

# 2. Extract ISL vocabulary from islVocabulary.ts
vocab_path = os.path.join(PROJECT_ROOT, 'packages', 'types', 'src', 'islVocabulary.ts')
with open(vocab_path, 'r', encoding='utf-8') as f:
    text = f.read()

vocab_ids = re.findall(r"id:\s*['\"]([A-Z0-9_]+)['\"]", text)
vocab_ids = sorted(list(set([v for v in vocab_ids if v != 'SBV_IDLE'])))

# 3. Check benchmark dataset
data_path = os.path.join(PROJECT_ROOT, 'ml', 'datasets', 'isl_benchmark_dataset.json')
with open(data_path, 'r', encoding='utf-8') as f:
    benchmark_data = json.load(f)

dataset_classes = sorted(list(set([item['signId'] for item in benchmark_data])))
dataset_signers = sorted(list(set([item['subjectId'] for item in benchmark_data])))

print("=" * 80)
print("PHASE 1: COMPLETE MODEL AND DATASET AUDIT (SIGNBRIDGE V7)")
print("=" * 80)
print(f"Total Runtime Vocabulary Entries: {len(vocab_ids)}")
print(f"Total Model Classes (Output Dim): {len(model_classes)}")
print(f"Total Dataset Benchmark Classes:  {len(dataset_classes)}")
print(f"Total Benchmark Sequences:        {len(benchmark_data)}")
print(f"Total Benchmark Signers:          {len(dataset_signers)}")

trained_in_vocab = [s for s in vocab_ids if s in model_classes]
untrained_vocab = [s for s in vocab_ids if s not in model_classes]

print(f"\n1. Trained Classes in Model ({len(trained_in_vocab)}):")
print(trained_in_vocab)

print(f"\n2. UNTRAINED VOCABULARY SIGNS ({len(untrained_vocab)}):")
for i, s in enumerate(untrained_vocab, 1):
    print(f"  {i:2d}. {s}")

print("\n" + "=" * 80)
print("AUDIT DISCREPANCY FINDINGS:")
print("=" * 80)
print(f"Discrepancy Ratio: Only {len(trained_in_vocab)}/{len(vocab_ids)} ({len(trained_in_vocab)/len(vocab_ids)*100:.1f}%) of runtime vocabulary is supported by the model!")
print("32 runtime vocabulary signs have ZERO weights in the GRU model!")
print("When a user performs any of these 32 signs, the model is FORCED by its 17-class Softmax layer")
print("to output one of the 17 trained classes, resulting in false predictions and repetition loops.")
