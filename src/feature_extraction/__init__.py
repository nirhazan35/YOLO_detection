"""
Feature extraction package for road accident detection
"""

from .feature_extractor import YOLO11FeatureExtractor
from .flow_feature_extractor import SimpleFlowFeatureExtractor, TorchFlowFeatureExtractor
from . import utils
from . import config 