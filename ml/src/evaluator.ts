import * as fs from 'fs';
import * as path from 'path';
import { MLBenchmarkMetrics, MLClassMetric, ConfusionMatrixEntry } from '@signbridge/types';
import { classifyTemporalSequence, SIGN_FEATURE_TEMPLATES } from './temporalClassifier';
import { generateFullBenchmarkDataset, BenchmarkSample } from './datasetGenerator';

export function runMLEvaluation(): MLBenchmarkMetrics {
  console.log('===========================================================');
  console.log('SIGNBRIDGE2VISION: ISL TEMPORAL MODEL EVALUATION BENCHMARK');
  console.log('===========================================================');

  const datasetPath = path.resolve(__dirname, '../datasets/isl_benchmark_dataset.json');
  let samples: BenchmarkSample[];
  if (fs.existsSync(datasetPath)) {
    samples = JSON.parse(fs.readFileSync(datasetPath, 'utf8'));
  } else {
    samples = generateFullBenchmarkDataset();
    const dir = path.dirname(datasetPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(datasetPath, JSON.stringify(samples, null, 2), 'utf8');
  }

  const signs = Object.keys(SIGN_FEATURE_TEMPLATES);
  const confusionMap: Record<string, Record<string, number>> = {};
  for (const actual of signs) {
    confusionMap[actual] = {};
    for (const pred of signs) {
      confusionMap[actual][pred] = 0;
    }
  }

  const latencies: number[] = [];
  let totalCorrect = 0;

  for (const sample of samples) {
    const t0 = performance.now();
    const prediction = classifyTemporalSequence(sample.frames);
    const latency = performance.now() - t0;
    latencies.push(latency);

    const actual = sample.signId;
    const predicted = prediction.sign;

    if (confusionMap[actual] && confusionMap[actual][predicted] !== undefined) {
      confusionMap[actual][predicted]++;
    }

    if (actual === predicted) {
      totalCorrect++;
    }
  }

  const classMetrics: MLClassMetric[] = [];
  let macroPrecision = 0;
  let macroRecall = 0;
  let macroF1 = 0;

  for (const sign of signs) {
    const tp = confusionMap[sign][sign] || 0;
    let fn = 0;
    let fp = 0;
    let support = 0;

    for (const other of signs) {
      if (other !== sign) {
        fn += confusionMap[sign][other] || 0;
        fp += confusionMap[other][sign] || 0;
      }
      support += confusionMap[sign][other] || 0;
    }

    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    macroPrecision += precision;
    macroRecall += recall;
    macroF1 += f1;

    classMetrics.push({
      sign,
      precision: Number(precision.toFixed(4)),
      recall: Number(recall.toFixed(4)),
      f1Score: Number(f1.toFixed(4)),
      support
    });
  }

  const numClasses = signs.length;
  macroPrecision /= numClasses;
  macroRecall /= numClasses;
  macroF1 /= numClasses;

  const accuracy = totalCorrect / samples.length;

  latencies.sort((a, b) => a - b);
  const p50 = latencies[Math.floor(latencies.length * 0.50)] || 0;
  const p95 = latencies[Math.floor(latencies.length * 0.95)] || 0;

  const confusionMatrix: ConfusionMatrixEntry[] = [];
  for (const actual of signs) {
    for (const predicted of signs) {
      confusionMatrix.push({
        actual,
        predicted,
        count: confusionMap[actual][predicted] || 0
      });
    }
  }

  const result: MLBenchmarkMetrics = {
    totalSamples: samples.length,
    accuracy: Number(accuracy.toFixed(4)),
    macroF1: Number(macroF1.toFixed(4)),
    weightedF1: Number(macroF1.toFixed(4)),
    p50InferenceLatencyMs: Number(p50.toFixed(2)),
    p95InferenceLatencyMs: Number(p95.toFixed(2)),
    classMetrics,
    confusionMatrix,
    evaluatedAt: new Date().toISOString()
  };

  console.log('Evaluated ' + samples.length + ' temporal sequences across ' + numClasses + ' ISL vocabulary classes.');
  console.log('Overall Accuracy: ' + (result.accuracy * 100).toFixed(2) + '%');
  console.log('Macro F1-Score:   ' + (result.macroF1 * 100).toFixed(2) + '%');
  console.log('Inference Latency: p50 = ' + result.p50InferenceLatencyMs + ' ms, p95 = ' + result.p95InferenceLatencyMs + ' ms\n');

  console.log('Per-Class Performance Breakdown:');
  console.log('----------------------------------------------------------------------');
  console.log('| Sign Class          | Precision | Recall  | F1-Score | Samples     |');
  console.log('----------------------------------------------------------------------');
  for (const cm of classMetrics) {
    console.log(
      '| ' + cm.sign.padEnd(19) + ' | ' + (cm.precision * 100).toFixed(1).padStart(7) + '% | ' + (cm.recall * 100).toFixed(1).padStart(5) + '% | ' + (cm.f1Score * 100).toFixed(1).padStart(6) + '% | ' + cm.support.toString().padStart(11) + ' |'
    );
  }
  console.log('----------------------------------------------------------------------\n');

  const reportPath = path.resolve(__dirname, '../evaluation/benchmark_report.json');
  fs.writeFileSync(reportPath, JSON.stringify(result, null, 2), 'utf8');
  console.log('Full report and confusion matrix saved to ' + reportPath);

  return result;
}

if (require.main === module) {
  runMLEvaluation();
}
