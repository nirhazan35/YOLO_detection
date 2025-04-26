"""
LSTM Architecture for Road Accident Detection

This module defines the deep learning architecture for processing features extracted
from video frames to detect road accidents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to weight frame importance in sequence.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.W(hidden_states))
        attention = F.softmax(self.v(energy).squeeze(-1), dim=1)
        context = torch.bmm(attention.unsqueeze(1), hidden_states).squeeze(1)
        return context


class FusionGate(nn.Module):
    """
    Gated fusion mechanism for combining RGB and flow features.
    """
    def __init__(self, rgb_dim=281, flow_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(rgb_dim + flow_dim, 128),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat, flow_feat):
        combined = torch.cat([rgb_feat, flow_feat], dim=2)
        gate = self.gate(combined)
        fused = gate * rgb_feat + (1 - gate) * flow_feat
        return fused


class AccidentLSTM(nn.Module):
    """
    Bidirectional LSTM model with attention for accident detection.
    
    The model processes sequences of combined features extracted from video frames,
    including object detection features, spatial features, and optical flow features.
    """
    def __init__(self, input_dim=409, hidden_dim=256, output_dim=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_dim*2)
        
        # Attention layer
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Temporal pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.fc(pooled)
        return output


class EnhancedAccidentLSTM(nn.Module):
    """
    Enhanced LSTM model with separate processing paths for RGB and flow features,
    followed by a gated fusion mechanism.
    """
    def __init__(self, rgb_dim=281, flow_dim=128, hidden_dim=256, output_dim=1, num_layers=2):
        super().__init__()
        
        # Separate LSTMs for RGB and flow features
        self.rgb_lstm = nn.LSTM(
            input_size=rgb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.flow_lstm = nn.LSTM(
            input_size=flow_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Custom attention
        self.rgb_attention = TemporalAttention(hidden_dim*2)
        self.flow_attention = TemporalAttention(hidden_dim*2)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Split input into RGB and flow components
        # Assuming the first 281 features are RGB and the rest 128 are flow
        rgb_features = x[:, :, :281]
        flow_features = x[:, :, 281:]
        
        # Process through LSTMs
        rgb_out, _ = self.rgb_lstm(rgb_features)
        flow_out, _ = self.flow_lstm(flow_features)
        
        # Apply attention
        rgb_context = self.rgb_attention(rgb_out)
        flow_context = self.flow_attention(flow_out)
        
        # Fusion
        combined = torch.cat([rgb_context, flow_context], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.fc(fused)
        return output 