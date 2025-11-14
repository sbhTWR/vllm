#!/usr/bin/env python3
"""GittinsPredictor implementation using survival curve models.

This predictor uses empirical survival curves to estimate expected tool call duration
based on tool name, arguments, and elapsed time since the tool call started.
"""

from __future__ import annotations

import bisect
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from vllm.core.evictor import Predictor

SIM_THRESHOLD = 0.3

class ToolDurationModel:
    """Internal model for a single tool's duration predictions."""
    
    def __init__(self, tool_name: str, data: Dict[str, Any]):
        self.tool_name = tool_name
        self.vectorizer = data['vectorizer']
        self.cluster_model = data['cluster_model']
        self.overall_stats = data['overall_stats']
        self.overall_survival = data['overall_survival']
        self.cluster_stats = data['cluster_stats']

        if self.cluster_model is not None:
            if hasattr(self.cluster_model, 'cluster_centers_'):
                self.centroids = self.cluster_model.cluster_centers_
            else:
                self.centroids = None
        else:
            self.centroids = None

    def survival_curve_for_cluster(self, cluster_id: int) -> List[Dict[str, float]]:
        """Get survival curve for a specific cluster."""
        for cluster in self.cluster_stats:
            if cluster['cluster_id'] == cluster_id:
                return cluster['survival_curve']
        return self.overall_survival

    def nearest_cluster(self, text_vector: np.ndarray) -> tuple[int, float]:
        """Find nearest cluster for given text vector."""
        if self.cluster_model is None or self.centroids is None:
            return 0, 1.0
        centroids = self.centroids
        if len(centroids.shape) == 2:
            sim = cosine_similarity(text_vector, centroids)[0]
            best = sim.argmax()
            return int(best), float(sim[best])
        return 0, 1.0


class GittinsPredictor:
    """Predictor using survival curve models for tool call duration estimation.
    
    Uses empirical survival curves to compute expected remaining time for tool calls
    based on tool name, arguments, and elapsed time.
    """
    
    def __init__(self, model_path: str | Path):
        """Initialize predictor by loading models from pkl file.
        
        Args:
            model_path: Path to the all_models.pkl file containing survival curve models
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with model_path.open('rb') as f:
            bundled = pickle.load(f)
        
        self.models: Dict[str, ToolDurationModel] = {}
        for tool_name, data in bundled.items():
            self.models[tool_name] = ToolDurationModel(tool_name, data)
    
    def _lookup_survival(self, curve: List[Dict[str, float]], t_ms: float) -> float:
        """Lookup survival probability at time t_ms using linear interpolation."""
        if not curve:
            return 0.0
        times = [pt['t_ms'] for pt in curve]
        survs = [pt['survival'] for pt in curve]
        idx = bisect.bisect_left(times, t_ms)
        if idx <= 0:
            return survs[0]
        if idx >= len(times):
            return 0.0
        t0, t1 = times[idx-1], times[idx]
        s0, s1 = survs[idx-1], survs[idx]
        if t1 == t0:
            return s1
        w = (t_ms - t0) / (t1 - t0)
        return s0 + w * (s1 - s0)
    
    def _expected_remaining_time(self, curve: List[Dict[str, float]], elapsed_ms: float) -> float:
        """Calculate expected remaining time E[T - t | T > t] using numerical integration.
        
        Formula: E[T - t | T > t] = ∫[t to ∞] S(u) du / S(t)
        """
        if not curve:
            return 0.0
        
        # Get survival at elapsed time
        S_t = self._lookup_survival(curve, elapsed_ms)
        if S_t <= 0:
            return 0.0
        
        # Extract times and survival values
        times = np.array([pt['t_ms'] for pt in curve])
        survs = np.array([pt['survival'] for pt in curve])
        
        # Find the index where elapsed_ms falls
        idx = bisect.bisect_left(times, elapsed_ms)
        
        # If elapsed time is beyond all data points, return 0
        if idx >= len(times):
            return 0.0
        
        # Interpolate survival at elapsed_ms
        if idx == 0:
            start_surv = survs[0]
            start_idx = 0
        else:
            # elapsed_ms is between times[idx-1] and times[idx]
            t0, t1 = times[idx-1], times[idx]
            s0, s1 = survs[idx-1], survs[idx]
            if t1 == t0:
                start_surv = s1
            else:
                w = (elapsed_ms - t0) / (t1 - t0)
                start_surv = s0 + w * (s1 - s0)
            start_idx = idx
        
        # Compute integral ∫[elapsed_ms to ∞] S(u) du using trapezoidal rule
        integral = 0.0
        
        # Add area from elapsed_ms to first data point after it (if any)
        if start_idx < len(times) and elapsed_ms < times[start_idx]:
            t_start = elapsed_ms
            t_end = times[start_idx]
            s_start = start_surv
            s_end = survs[start_idx]
            # Trapezoidal area
            integral += (t_end - t_start) * (s_start + s_end) / 2.0
        
        # Add areas between subsequent data points
        for i in range(start_idx, len(times) - 1):
            t0, t1 = times[i], times[i+1]
            s0, s1 = survs[i], survs[i+1]
            # Trapezoidal area
            integral += (t1 - t0) * (s0 + s1) / 2.0
        
        # The expected remaining time is the integral divided by survival at elapsed time
        expected_remaining_ms = integral / S_t if S_t > 0 else 0.0
        
        return float(expected_remaining_ms)
    
    def _canonicalize_text(self, text: str) -> str:
        """Normalize text for vectorization."""
        return " ".join(text.strip().lower().split())
    
    def _select_curve(self, model: ToolDurationModel, text: str) -> tuple[List[Dict[str, float]], float, int]:
        """Select appropriate survival curve based on tool arguments.
        
        Returns:
            Tuple of (survival_curve, similarity_score, cluster_id)
        """
        text_vec = model.vectorizer.transform([self._canonicalize_text(text)])
        cluster_id, similarity = model.nearest_cluster(text_vec)
        if similarity < SIM_THRESHOLD:
            return model.overall_survival, similarity, -1
        return model.survival_curve_for_cluster(cluster_id), similarity, cluster_id
    
    def expected_tool_duration(self, tool_name: str, tool_args: str,
                               last_access_duration_s: float) -> float:
        """Predicts expected remaining time for a tool call.
        
        Args:
            tool_name: Name of the tool (e.g., "Bash", "Read", "WebFetch")
            tool_args: Arguments/command string for the tool call
            last_access_duration_s: Elapsed time since tool call started (in seconds)
        
        Returns:
            Expected remaining time in seconds until tool call completion
        """
        # Find model for this tool (try exact match, then case-insensitive)
        model = self.models.get(tool_name)
        if model is None:
            # Try case-insensitive lookup
            tool_lower = tool_name.lower()
            tool_capitalized = tool_name.capitalize()
            model = self.models.get(tool_lower) or self.models.get(tool_capitalized)
        
        if model is None:
            # If no model found, return a default (could also raise an exception)
            # For now, return a conservative estimate
            return 60.0  # Default 60 seconds
        
        # Select appropriate survival curve based on arguments
        curve, similarity, cluster_id = self._select_curve(model, tool_args)
        
        # Convert elapsed time to milliseconds
        elapsed_ms = last_access_duration_s * 1000.0
        
        # Compute expected remaining time
        expected_remaining_ms = self._expected_remaining_time(curve, elapsed_ms)
        
        # Convert back to seconds
        return expected_remaining_ms / 1000.0

