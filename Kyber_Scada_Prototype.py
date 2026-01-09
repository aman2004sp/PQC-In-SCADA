# =====================================================================
# PART 1: RESEARCH CONFIGURATION AND UTILITIES
# =====================================================================

import time
import random
import statistics
import sys
import math
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------
# SCADA SYSTEM CONSTRAINTS
# ---------------------------------------------------------------------

# Maximum allowed latency for real-time SCADA control loops (milliseconds)
SCADA_MAX_LATENCY_MS = 5.0

# Number of iterations for benchmarking
BENCHMARK_ITERATIONS = 1000

# ---------------------------------------------------------------------
# KYBER PARAMETER SETS (SIMPLIFIED FOR BENCHMARKING)
# ---------------------------------------------------------------------

KYBER_PARAMETER_SETS = {
    "KYBER_512": {
        "n": 256,
        "k": 2,
        "q": 3329,
        "eta": 2
    },
    "KYBER_768": {
        "n": 256,
        "k": 3,
        "q": 3329,
        "eta": 2
    },
    "KYBER_1024": {
        "n": 256,
        "k": 4,
        "q": 3329,
        "eta": 2
    }
}

# ---------------------------------------------------------------------
# TIMING UTILITIES
# ---------------------------------------------------------------------

def current_time_ms() -> float:
    """
    Returns high-resolution wall-clock time in milliseconds.
    """
    return time.perf_counter() * 1000


def measure_execution_time(func, *args, **kwargs) -> float:
    """
    Measures execution time of a function call in milliseconds.
    """
    start = current_time_ms()
    func(*args, **kwargs)
    end = current_time_ms()
    return end - start

# ---------------------------------------------------------------------
# MEMORY ESTIMATION UTILITIES
# ---------------------------------------------------------------------

def estimate_object_size(obj, visited=None) -> int:
    """
    Recursively estimates memory usage of Python objects.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return 0

    visited.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += estimate_object_size(k, visited)
            size += estimate_object_size(v, visited)

    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            size += estimate_object_size(item, visited)

    return size

# ---------------------------------------------------------------------
# STATISTICAL ANALYSIS UTILITIES
# ---------------------------------------------------------------------

def compute_latency_statistics(latencies: List[float]) -> Dict[str, float]:
    """
    Computes statistical metrics for a list of latency measurements.
    """
    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "max": max(latencies),
        "min": min(latencies),
        "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    }

# ---------------------------------------------------------------------
# SCADA FEASIBILITY CHECK
# ---------------------------------------------------------------------

def is_scada_feasible(latency_ms: float) -> bool:
    """
    Checks whether a latency value satisfies SCADA real-time constraints.
    """
    return latency_ms <= SCADA_MAX_LATENCY_MS

# =====================================================================
# END OF PART 1
# =====================================================================
