"""
LSTM Architecture Module for Road Accident Detection

This module contains the LSTM-based deep learning architecture
for detecting road accidents from video sequences.
"""

from src.LSTM_architecture.model import AccidentLSTM, EnhancedAccidentLSTM, TemporalAttention, FusionGate
from src.LSTM_architecture.config import MODEL_CONFIG, MODEL_PATHS, FEATURE_PATHS 