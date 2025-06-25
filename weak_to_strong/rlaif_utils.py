"""
Utilities for RLAIF-based weak-to-strong training with deception mitigation.
This module provides helper functions for partial observability, trajectory tracking, 
and deception detection in the alignment process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class TrajectoryStep:
    """Single step in a trajectory for partial observability modeling."""
    input_ids: torch.Tensor
    logits: torch.Tensor
    action: int
    reward: float
    log_prob: float
    confidence: float
    timestamp: int


class DeceptionDetector:
    """
    Detector for identifying potential deceptive behavior in weak model supervision.
    Uses statistical and pattern-based methods to detect inconsistencies.
    """
    
    def __init__(self, confidence_threshold: float = 0.8, consistency_window: int = 10):
        self.confidence_threshold = confidence_threshold
        self.consistency_window = consistency_window
        self.prediction_history = deque(maxlen=consistency_window)
        self.confidence_history = deque(maxlen=consistency_window)
        
    def add_prediction(self, prediction: int, confidence: float):
        """Add a new prediction to the history."""
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
    def detect_deception_signals(self) -> Dict[str, float]:
        """
        Detect various signals that might indicate deceptive behavior.
        
        Returns:
            Dictionary with deception signal scores
        """
        if len(self.prediction_history) < 3:
            return {"insufficient_data": 1.0}
            
        signals = {}
        
        # Signal 1: High confidence with inconsistent predictions
        if len(self.prediction_history) >= self.consistency_window:
            recent_preds = list(self.prediction_history)
            recent_confs = list(self.confidence_history)
            
            # Check for prediction flipping with high confidence
            flips = sum(1 for i in range(1, len(recent_preds)) 
                       if recent_preds[i] != recent_preds[i-1])
            avg_confidence = np.mean(recent_confs)
            
            signals["confidence_inconsistency"] = flips * avg_confidence
            
        # Signal 2: Sudden confidence drops
        if len(self.confidence_history) >= 3:
            recent_confs = list(self.confidence_history)[-3:]
            confidence_drop = recent_confs[0] - recent_confs[-1]
            signals["confidence_drop"] = max(0, confidence_drop)
            
        return signals
        
    def compute_deception_score(self) -> float:
        """
        Compute an overall deception score based on all signals.
        
        Returns:
            Float between 0 and 1, where higher values indicate more likely deception
        """
        signals = self.detect_deception_signals()
        
        if "insufficient_data" in signals:
            return 0.0
            
        # Weighted combination of signals
        weights = {
            "confidence_inconsistency": 0.5,
            "confidence_drop": 0.5,
        }
        
        score = 0.0
        total_weight = 0.0
        
        for signal_name, value in signals.items():
            if signal_name in weights:
                score += weights[signal_name] * min(1.0, value)
                total_weight += weights[signal_name]
                
        return score / total_weight if total_weight > 0 else 0.0
