#!/usr/bin/env python3
"""Headless Unified Genie runner that stores negotiation + optimization output."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from adaptive_genie_network import AdaptiveGenieNetwork
from example_applications import DialecticalParticleSwarm, rastrigin


def _serialize(obj: Any) -> Any:
    """Convert numpy objects and other custom types into JSON-safe data."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    return obj


def run_unified_genie(max_iterations: int = 25) -> Dict[str, Any]:
    """Execute a compact AdaptiveGenieNetwork flow and return structured output."""
    genie = AdaptiveGenieNetwork()

    # Baseline negotiation using a representative landscape snapshot.
    problem_landscape = {
        "dimensions": 2,
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "multimodality": 0.6,
        "noise_level": 0.05,
        "deception": 0.35,
    }
    system_state = {
        "fitness_history": [1.0, 0.78, 0.52, 0.41],
        "population_diversity": 0.73,
        "fitness_variance": 0.19,
    }

    negotiation = genie.tune_parameters(problem_landscape, system_state)
    network_state = genie.get_system_state()

    # Lightweight optimization pass that mirrors the Streamlit demo behaviour.
    dpso = DialecticalParticleSwarm(rastrigin, problem_landscape["bounds"])
    optimization = dpso.optimize(max_iterations=max_iterations)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "problem_landscape": problem_landscape,
        "negotiation": negotiation,
        "network_state": network_state,
        "optimization": {
            "best_solution": optimization.best_solution,
            "best_fitness": optimization.best_fitness,
            "convergence_history": optimization.convergence_history,
            "population_history": optimization.population_history,
            "crystallization_history": optimization.crystallization_history,
            "dialectical_states": optimization.dialectical_states,
            "total_evaluations": optimization.total_evaluations,
            "execution_time": optimization.execution_time,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Unified Genie headless snippet.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("result.json"),
        help="Path to write the JSON results (default: ./result.json)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=25,
        help="Maximum optimization iterations for the dialectical swarm (default: 25)",
    )
    args = parser.parse_args()

    payload = run_unified_genie(max_iterations=args.iterations)

    # Ensure parent directory exists.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=_serialize)

    print(f"Unified Genie results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
