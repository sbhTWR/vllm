#!/usr/bin/env python3
"""Inference utilities for tool-call duration probabilities.

Loads the outputs of `build_tool_duration_clusters.py` and exposes helpers to
compute conditional completion probabilities:

    P(finish within Δ | elapsed = t)

Usage (from shell):
    python3 scripts/infer_duration_probability.py --tool Bash --elapsed 30 --delta 120 --text "pytest tests/" 

The script also prints several demo queries when invoked without arguments.
"""

from __future__ import annotations

import argparse
import bisect
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CLUSTER_STATS_PATH = Path("example_logs/duration_clusters_sessionwise/cluster_stats.json")
MODELS_BUNDLE_PATH = Path("example_logs/duration_clusters_sessionwise/all_models.pkl")
SIM_THRESHOLD = 0.3

# -----------------------------------------------------------------------------
# Utilities to load models
# -----------------------------------------------------------------------------

class ToolDurationModel:
    def __init__(self, tool_name: str, data: Dict[str, Any]):
        self.tool_name = tool_name
        self.vectorizer = data['vectorizer']
        self.cluster_model = data['cluster_model']
        self.overall_stats = data['overall_stats']
        self.overall_survival = data['overall_survival']
        self.cluster_stats = data['cluster_stats']

        if self.cluster_model is not None:
            X = self.cluster_model.cluster_centers_
            if hasattr(self.cluster_model, 'cluster_centers_'):
                self.centroids = self.cluster_model.cluster_centers_
            else:
                self.centroids = None
        else:
            self.centroids = None

    def survival_curve_for_cluster(self, cluster_id: int) -> List[Dict[str, float]]:
        for cluster in self.cluster_stats:
            if cluster['cluster_id'] == cluster_id:
                return cluster['survival_curve']
        return self.overall_survival

    def nearest_cluster(self, text_vector: np.ndarray) -> Tuple[int, float]:
        if self.cluster_model is None or self.centroids is None:
            return 0, 1.0
        centroids = self.centroids
        if len(centroids.shape) == 2:
            sim = cosine_similarity(text_vector, centroids)[0]
            best = sim.argmax()
            return int(best), float(sim[best])
        return 0, 1.0


def load_models() -> Dict[str, ToolDurationModel]:
    if not MODELS_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Combined model bundle not found at {MODELS_BUNDLE_PATH}")
    with MODELS_BUNDLE_PATH.open('rb') as f:
        bundled = pickle.load(f)
    models = {}
    for tool_name, data in bundled.items():
        models[tool_name] = ToolDurationModel(tool_name, data)
    return models


# -----------------------------------------------------------------------------
# Survival math
# -----------------------------------------------------------------------------

def lookup_survival(curve: List[Dict[str, float]], t_ms: float) -> float:
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


def conditional_completion_probability(curve: List[Dict[str, float]], elapsed_ms: float, delta_ms: float) -> float:
    S_elapsed = lookup_survival(curve, elapsed_ms)
    if S_elapsed <= 0:
        return 1.0
    S_future = lookup_survival(curve, elapsed_ms + delta_ms)
    prob = 1.0 - (S_future / S_elapsed)
    return float(min(max(prob, 0.0), 1.0))


# -----------------------------------------------------------------------------
# Text processing
# -----------------------------------------------------------------------------

def canonicalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def vectorize_text(model: ToolDurationModel, text: str) -> np.ndarray:
    return model.vectorizer.transform([text])


def select_curve(model: ToolDurationModel, text: str) -> Tuple[List[Dict[str, float]], float, int]:
    text_vec = vectorize_text(model, text)
    cluster_id, similarity = model.nearest_cluster(text_vec)
    if similarity < SIM_THRESHOLD:
        return model.overall_survival, similarity, -1
    return model.survival_curve_for_cluster(cluster_id), similarity, cluster_id


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def run_single(models: Dict[str, ToolDurationModel], tool: str, elapsed_ms: float, delta_ms: float, text: str) -> Dict[str, Any]:
    model = models.get(tool)
    if model is None and tool.lower() in models:
        model = models[tool.lower()]
    if model is None and tool.capitalize() in models:
        model = models[tool.capitalize()]
    if not model:
        raise ValueError(f"No model found for tool '{tool}'")
    curve, similarity, cluster_id = select_curve(model, text)
    prob = conditional_completion_probability(curve, elapsed_ms, delta_ms)
    return {
        'tool': tool,
        'elapsed_ms': elapsed_ms,
        'delta_ms': delta_ms,
        'probability': prob,
        'similarity': similarity,
        'cluster_id': cluster_id,
    }


def demo(models: Dict[str, ToolDurationModel]):
    tests = [
        {
            'tool': 'Bash',
            'text': 'pytest tests/test_routes.py',
            'elapsed_ms': 30_000,
            'delta_ms': 120_000,
        },
        {
            'tool': 'Bash',
            'text': 'docker build -t image .',
            'elapsed_ms': 120_000,
            'delta_ms': 180_000,
        },
        {
            'tool': 'Read',
            'text': 'path=config.yaml',
            'elapsed_ms': 2000,
            'delta_ms': 8000,
        },
        {
            'tool': 'TodoWrite',
            'text': 'Write summary bullet points',
            'elapsed_ms': 10_000,
            'delta_ms': 30_000,
        },
    ]

    print("=== Demo Queries ===")
    for test in tests:
        result = run_single(models, test['tool'], test['elapsed_ms'], test['delta_ms'], test['text'])
        print(json.dumps({**test, **result}, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Compute P(finish within Δ | elapsed t) for tool calls.')
    parser.add_argument('--tool', help='Tool name (e.g. Bash)')
    parser.add_argument('--elapsed', type=float, help='Elapsed time in seconds')
    parser.add_argument('--delta', type=float, help='Future horizon in seconds')
    parser.add_argument('--text', default='', help='Argument text (command or tool_input summary)')
    parser.add_argument('--demo', action='store_true', help='Run demo queries')

    args = parser.parse_args()

    models = load_models()

    if args.demo or not args.tool:
        demo(models)
        return

    tool = args.tool
    elapsed_ms = args.elapsed * 1000.0
    delta_ms = args.delta * 1000.0
    text = canonicalize_text(args.text)

    result = run_single(models, tool, elapsed_ms, delta_ms, text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
