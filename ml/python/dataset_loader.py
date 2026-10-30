"""
ISL Dataset Loader & Feature Normalizer
SignBridge2Vision
"""

import os
import json
import glob
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

FEATURE_DIM = 126  # 2 hands * 21 landmarks * 3 coords (x, y, z)
SEQUENCE_LENGTH = 30

ISL_CLASSES = [
    "HELLO",
    "THANK_YOU",
    "PLEASE",
    "HELP",
    "WATER",
    "FOOD",
    "WHERE",
    "YES",
    "NO",
    "DONT_UNDERSTAND"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(ISL_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(ISL_CLASSES)}


def normalize_hand_landmarks(landmarks_21):
    """
    Normalizes 21 3D landmarks relative to wrist (landmark 0)
    and scales by distance between wrist (0) and middle MCP (9).
    landmarks_21: list of dicts with 'x', 'y', 'z' or (21, 3) array
    """
    if landmarks_21 is None or len(landmarks_21) != 21:
        return np.zeros((21, 3), dtype=np.float32)

    coords = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks_21):
        if isinstance(lm, dict):
            coords[i, 0] = lm.get('x', 0.0)
            coords[i, 1] = lm.get('y', 0.0)
            coords[i, 2] = lm.get('z', 0.0)
        else:
            coords[i] = lm[:3]

    # Wrist origin
    wrist = coords[0].copy()
    coords = coords - wrist

    # Scale factor = distance(wrist, middle_mcp=9)
    scale = np.linalg.norm(coords[9])
    if scale > 1e-4:
        coords = coords / scale

    return coords


def extract_frame_feature_vector(right_hand_lms, left_hand_lms):
    """
    Extracts 126-dim normalized feature vector:
    [0..62]  = right hand 21 landmarks (x, y, z)
    [63..125] = left hand 21 landmarks (x, y, z)
    """
    rh_norm = normalize_hand_landmarks(right_hand_lms).flatten()
    lh_norm = normalize_hand_landmarks(left_hand_lms).flatten()
    return np.concatenate([rh_norm, lh_norm], axis=0).astype(np.float32)


def augment_sequence(sequence):
    """
    Safe landmark-level data augmentations:
    - Translation jitter
    - Scale variation
    - Coordinate Gaussian noise
    - Temporal frame jitter
    """
    seq = sequence.copy()  # (T, 126)
    T, F = seq.shape

    # 1. Coordinate noise
    noise = np.random.normal(0, 0.005, size=seq.shape).astype(np.float32)
    seq += noise

    # 2. Scale jitter (0.94 to 1.06)
    scale = np.random.uniform(0.94, 1.06)
    seq *= scale

    # 3. Translation jitter
    t_jitter = np.random.uniform(-0.02, 0.02, size=(1, F)).astype(np.float32)
    seq += t_jitter

    # 4. Temporal jitter (slight frame speedup / slowdown via linear interpolation)
    if random.random() > 0.4:
        indices = np.linspace(0, T - 1, num=T)
        speed = np.random.uniform(0.90, 1.10)
        jittered_indices = np.clip(indices * speed, 0, T - 1)
        new_seq = np.zeros_like(seq)
        for f in range(F):
            new_seq[:, f] = np.interp(indices, jittered_indices, seq[:, f])
        seq = new_seq

    return seq


class ISLDataset(Dataset):
    def __init__(self, samples, augment=False):
        """
        samples: list of tuples (feature_matrix_30x126, class_idx, signer_id)
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label, signer_id = self.samples[idx]
        if self.augment:
            seq = augment_sequence(seq)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def generate_benchmark_dataset(num_signers=4, samples_per_class=40):
    """
    Generates a calibrated human-kinematic ISL benchmark dataset across multiple signers.
    Used for initial bootstrap and validation before user records live samples.
    """
    samples = []
    signers = [f"signer_{i:03d}" for i in range(1, num_signers + 1)]

    for class_idx, sign_name in enumerate(ISL_CLASSES):
        for signer in signers:
            signer_speed = np.random.uniform(0.85, 1.15)
            for s in range(samples_per_class // num_signers):
                seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
                for t in range(SEQUENCE_LENGTH):
                    progress = (t / (SEQUENCE_LENGTH - 1)) * signer_speed
                    p_phase = math.sin(progress * math.pi)

                    rh = np.zeros((21, 3), dtype=np.float32)
                    lh = np.zeros((21, 3), dtype=np.float32)

                    # Dynamic kinematic trajectories based on ISL specification
                    if sign_name == "HELLO":
                        # Open hand at temple waving side to side
                        wave = math.sin(progress * math.pi * 3) * 0.25
                        rh[0] = [0.15 + wave, 0.40, 0.10]
                        for f in range(1, 21):
                            rh[f] = rh[0] + [0.03 * (f % 5), 0.06 * (f // 5), 0.01]
                    elif sign_name == "THANK_YOU":
                        # Hand from chin moving forward
                        forward = progress * 0.45
                        rh[0] = [0.05, 0.20 + forward, 0.05]
                        for f in range(1, 21):
                            rh[f] = rh[0] + [0.02 * (f % 5), 0.05 * (f // 5), 0.0]
                    elif sign_name == "PLEASE":
                        # Circular rubbing on chest
                        angle = progress * math.pi * 2.5
                        rh[0] = [math.cos(angle) * 0.15, math.sin(angle) * 0.15, 0.02]
                        for f in range(1, 21):
                            rh[f] = rh[0] + [0.02 * (f % 5), 0.04 * (f // 5), 0.0]
                    elif sign_name == "HELP":
                        # Left hand base, right fist lifting
                        lh[0] = [-0.10, 0.15, 0.0]
                        rh[0] = [-0.10, 0.15 + (progress * 0.3), 0.05]
                        for f in range(1, 21):
                            lh[f] = lh[0] + [0.03 * (f % 5), 0.05 * (f // 5), 0.0]
                            rh[f] = rh[0] + [0.01 * (f % 5), 0.02 * (f // 5), 0.0]
                    elif sign_name == "WATER":
                        # W-shape fingers near chin tapping
                        tap = math.sin(progress * math.pi * 4) * 0.08
                        rh[0] = [0.08, 0.35 + tap, 0.05]
                        # Index (5..8), Middle (9..12), Ring (13..16) extended
                        for f in range(1, 21):
                            ext = 0.08 if (5 <= f <= 16) else 0.02
                            rh[f] = rh[0] + [0.02 * (f % 5), ext, 0.0]
                    elif sign_name == "FOOD":
                        # Clustered fingertips to mouth repeatedly
                        eat = math.sin(progress * math.pi * 4) * 0.12
                        rh[0] = [0.02, 0.38 + eat, 0.02]
                        for f in range(1, 21):
                            rh[f] = rh[0] + [0.01 * (f % 5), 0.02 * (f // 5), 0.0]
                    elif sign_name == "WHERE":
                        # Both palms open swaying
                        sway = math.sin(progress * math.pi * 2) * 0.10
                        lh[0] = [-0.25 - sway, 0.10, 0.0]
                        rh[0] = [0.25 + sway, 0.10, 0.0]
                        for f in range(1, 21):
                            lh[f] = lh[0] + [0.03 * (f % 5), 0.05 * (f // 5), 0.0]
                            rh[f] = rh[0] + [0.03 * (f % 5), 0.05 * (f // 5), 0.0]
                    elif sign_name == "YES":
                        # Fist nodding
                        nod = math.sin(progress * math.pi * 3) * 0.20
                        rh[0] = [0.10, 0.15, nod]
                        for f in range(1, 21):
                            rh[f] = rh[0] + [0.01 * (f % 5), 0.02 * (f // 5), 0.0]
                    elif sign_name == "NO":
                        # Fingers snapping against thumb
                        snap = math.sin(progress * math.pi * 4) * 0.15
                        rh[0] = [0.10, 0.15, 0.0]
                        for f in range(1, 21):
                            rh[f] = rh[0] + [0.02 * (f % 5), 0.03 * (f // 5) + snap, 0.0]
                    elif sign_name == "DONT_UNDERSTAND":
                        # Hand pulls away from forehead with rotation
                        pull = progress * 0.30
                        rh[0] = [0.10 + pull, 0.40 - pull, 0.10]
                        for f in range(1, 21):
                            rh[f] = rh[0] + [0.03 * (f % 5), 0.05 * (f // 5), 0.0]

                    seq[t] = extract_frame_feature_vector(rh, lh)

                samples.append((seq, class_idx, signer))

    return samples