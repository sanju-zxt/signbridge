"""
ISL Temporal Neural Network Architectures
SignBridge2Vision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ISL_GRU_Classifier(nn.Module):
    """
    Bidirectional GRU with Temporal Pooling and Linear Classifier Head.
    Input shape: (batch_size, sequence_length=30, feature_dim=126)
    Output shape: (batch_size, num_classes=10)
    """
    def __init__(self, input_dim=126, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.3):
        super(ISL_GRU_Classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Self-attention pooling layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_norm(x)
        gru_out, _ = self.gru(x)  # (B, T, 2*H)
        
        # Compute attention weights over time
        attn_weights = F.softmax(self.attention(gru_out), dim=1)  # (B, T, 1)
        context = torch.sum(gru_out * attn_weights, dim=1)        # (B, 2*H)
        
        logits = self.fc(context)  # (B, num_classes)
        return logits


class ISL_LSTM_Classifier(nn.Module):
    """
    Multi-layer LSTM Classifier baseline.
    Input shape: (batch_size, sequence_length=30, feature_dim=126)
    """
    def __init__(self, input_dim=126, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.3):
        super(ISL_LSTM_Classifier, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Last time step
        logits = self.fc(last_hidden)
        return logits