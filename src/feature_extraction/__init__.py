"""
Feature extraction package for road accident detection
"""

from .feature_extractor import YOLO11FeatureExtractor, CrossModalFusion
from .flow_feature_extractor import (
    SimpleFlowFeatureExtractor, 
    TorchFlowFeatureExtractor, 
    I3DFlowFeatureExtractor,
    TemporalDifferenceExtractor
)
from . import utils
from . import config 