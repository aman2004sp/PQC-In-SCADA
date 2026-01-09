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
# PART 2: MATHEMATICAL FOUNDATIONS (UNOPTIMIZED)
# =====================================================================

# ---------------------------------------------------------------------
# Modular Arithmetic
# ---------------------------------------------------------------------

def mod_q(value: int, q: int) -> int:
    """
    Reduces an integer modulo q.
    """
    return value % q


# ---------------------------------------------------------------------
# Polynomial Operations
# ---------------------------------------------------------------------

def poly_zero(n: int) -> List[int]:
    """
    Creates a zero polynomial of degree n.
    """
    return [0 for _ in range(n)]


def poly_add(a: List[int], b: List[int], q: int) -> List[int]:
    """
    Polynomial addition modulo q.
    """
    result = []
    for i in range(len(a)):
        result.append((a[i] + b[i]) % q)
    return result


def poly_sub(a: List[int], b: List[int], q: int) -> List[int]:
    """
    Polynomial subtraction modulo q.
    """
    result = []
    for i in range(len(a)):
        result.append((a[i] - b[i]) % q)
    return result


def poly_scalar_mul(poly: List[int], scalar: int, q: int) -> List[int]:
    """
    Multiplies a polynomial by a scalar modulo q.
    """
    result = []
    for coeff in poly:
        result.append((coeff * scalar) % q)
    return result


def poly_mul(a: List[int], b: List[int], q: int) -> List[int]:
    """
    Naive polynomial multiplication modulo (x^n + 1) and q.
    Intentionally unoptimized (O(n^2)).
    """
    n = len(a)
    result = [0] * n

    for i in range(n):
        for j in range(n):
            index = (i + j) % n
            sign = -1 if (i + j) >= n else 1
            result[index] += sign * a[i] * b[j]

    for i in range(n):
        result[i] %= q

    return result


# ---------------------------------------------------------------------
# Noise Sampling (Centered Binomial Distribution)
# ---------------------------------------------------------------------

def sample_noise(n: int, eta: int) -> List[int]:
    """
    Samples a noise polynomial using a centered binomial distribution.
    """
    noise = []
    for _ in range(n):
        a = 0
        b = 0
        for _ in range(eta):
            a += random.randint(0, 1)
            b += random.randint(0, 1)
        noise.append(a - b)
    return noise


# ---------------------------------------------------------------------
# Uniform Polynomial Sampling
# ---------------------------------------------------------------------

def sample_uniform_poly(n: int, q: int) -> List[int]:
    """
    Samples a polynomial uniformly at random modulo q.
    """
    poly = []
    for _ in range(n):
        poly.append(random.randint(0, q - 1))
    return poly


# ---------------------------------------------------------------------
# Matrix and Vector Utilities
# ---------------------------------------------------------------------

def generate_matrix(k: int, n: int, q: int) -> List[List[List[int]]]:
    """
    Generates a k x k matrix of polynomials.
    """
    matrix = []
    for i in range(k):
        row = []
        for j in range(k):
            row.append(sample_uniform_poly(n, q))
        matrix.append(row)
    return matrix


def generate_secret_vector(k: int, n: int, eta: int) -> List[List[int]]:
    """
    Generates a secret vector of k noise polynomials.
    """
    vector = []
    for _ in range(k):
        vector.append(sample_noise(n, eta))
    return vector


def generate_error_vector(k: int, n: int, eta: int) -> List[List[int]]:
    """
    Generates an error vector of k noise polynomials.
    """
    vector = []
    for _ in range(k):
        vector.append(sample_noise(n, eta))
    return vector


# ---------------------------------------------------------------------
# Matrix-Vector Multiplication
# ---------------------------------------------------------------------

def matrix_vector_mul(
    matrix: List[List[List[int]]],
    vector: List[List[int]],
    q: int
) -> List[List[int]]:
    """
    Multiplies a polynomial matrix with a polynomial vector.
    """
    k = len(vector)
    n = len(vector[0])
    result = []

    for i in range(k):
        acc = poly_zero(n)
        for j in range(k):
            product = poly_mul(matrix[i][j], vector[j], q)
            acc = poly_add(acc, product, q)
        result.append(acc)

    return result


# =====================================================================
# END OF PART 2
# =====================================================================

# =====================================================================
# PART 3: KYBER KEY GENERATION (ML-KEM KEYGEN)
# =====================================================================

# ---------------------------------------------------------------------
# Helper Functions for Encoding (Simplified)
# ---------------------------------------------------------------------

def encode_poly(poly: List[int]) -> List[int]:
    """
    Encodes a polynomial into a byte-like integer list.
    Simplified representation for benchmarking purposes.
    """
    return list(poly)


def encode_vector(vector: List[List[int]]) -> List[List[int]]:
    """
    Encodes a vector of polynomials.
    """
    return [encode_poly(poly) for poly in vector]


# ---------------------------------------------------------------------
# Kyber Key Generation
# ---------------------------------------------------------------------

def kyber_keygen(params: Dict) -> Tuple[Dict, Dict]:
    """
    Generates a Kyber public/private key pair.

    Parameters:
        params: Dictionary containing Kyber parameters

    Returns:
        public_key: Dictionary containing public key components
        private_key: Dictionary containing private key components
    """
    n = params["n"]
    k = params["k"]
    q = params["q"]
    eta = params["eta"]

    # Step 1: Generate public matrix A
    A = generate_matrix(k, n, q)

    # Step 2: Sample secret vector s
    s = generate_secret_vector(k, n, eta)

    # Step 3: Sample error vector e
    e = generate_error_vector(k, n, eta)

    # Step 4: Compute t = A * s + e
    As = matrix_vector_mul(A, s, q)

    t = []
    for i in range(k):
        t.append(poly_add(As[i], e[i], q))

    # Step 5: Encode public and private keys
    public_key = {
        "A": A,
        "t": encode_vector(t),
        "params": params
    }

    private_key = {
        "s": encode_vector(s),
        "params": params
    }

    return public_key, private_key


# ---------------------------------------------------------------------
# Key Size Estimation Utilities
# ---------------------------------------------------------------------

def estimate_public_key_size(public_key: Dict) -> int:
    """
    Estimates memory size of the public key.
    """
    return estimate_object_size(public_key)


def estimate_private_key_size(private_key: Dict) -> int:
    """
    Estimates memory size of the private key.
    """
    return estimate_object_size(private_key)


# =====================================================================
# END OF PART 3
# =====================================================================
# =====================================================================
# PART 4: KYBER ENCAPSULATION (ML-KEM ENCAPS)
# =====================================================================

# ---------------------------------------------------------------------
# Helper Functions for Message and Key Handling
# ---------------------------------------------------------------------

def generate_random_message(n: int) -> List[int]:
    """
    Generates a random binary message polynomial.
    """
    return [random.randint(0, 1) for _ in range(n)]


def derive_shared_secret(message: List[int]) -> int:
    """
    Derives a shared secret from a message polynomial.
    Simplified derivation for benchmarking.
    """
    secret = 0
    for bit in message:
        secret = (secret << 1) ^ bit
    return secret


# ---------------------------------------------------------------------
# Kyber Encapsulation
# ---------------------------------------------------------------------

def kyber_encapsulate(public_key: Dict) -> Tuple[Dict, int]:
    """
    Performs Kyber encapsulation using the public key.

    Parameters:
        public_key: Dictionary containing Kyber public key

    Returns:
        ciphertext: Dictionary containing ciphertext components
        shared_secret: Derived shared secret
    """
    A = public_key["A"]
    t = public_key["t"]
    params = public_key["params"]

    n = params["n"]
    k = params["k"]
    q = params["q"]
    eta = params["eta"]

    # Step 1: Sample random message m
    m = generate_random_message(n)

    # Step 2: Sample ephemeral secret vector r
    r = generate_secret_vector(k, n, eta)

    # Step 3: Sample error vectors
    e1 = generate_error_vector(k, n, eta)
    e2 = sample_noise(n, eta)

    # Step 4: Compute u = A^T * r + e1
    u = []
    for i in range(k):
        acc = poly_zero(n)
        for j in range(k):
            product = poly_mul(A[j][i], r[j], q)
            acc = poly_add(acc, product, q)
        acc = poly_add(acc, e1[i], q)
        u.append(acc)

    # Step 5: Compute v = t^T * r + e2 + m
    acc = poly_zero(n)
    for i in range(k):
        product = poly_mul(t[i], r[i], q)
        acc = poly_add(acc, product, q)

    acc = poly_add(acc, e2, q)
    acc = poly_add(acc, m, q)
    v = acc

    # Step 6: Derive shared secret
    shared_secret = derive_shared_secret(m)

    # Step 7: Form ciphertext
    ciphertext = {
        "u": u,
        "v": v,
        "params": params
    }

    return ciphertext, shared_secret


# ---------------------------------------------------------------------
# Ciphertext Size Estimation Utility
# ---------------------------------------------------------------------

def estimate_ciphertext_size(ciphertext: Dict) -> int:
    """
    Estimates memory size of the ciphertext.
    """
    return estimate_object_size(ciphertext)


# =====================================================================
# END OF PART 4
# =====================================================================
# =====================================================================
# PART 5: KYBER DECAPSULATION (ML-KEM DECAPS)
# =====================================================================

# ---------------------------------------------------------------------
# Message Recovery Utility
# ---------------------------------------------------------------------

def recover_message(v: List[int], s: List[List[int]], q: int) -> List[int]:
    """
    Recovers the message polynomial from v using the secret key.
    Simplified recovery for benchmarking.
    """
    n = len(v)
    acc = poly_zero(n)

    for i in range(len(s)):
        product = poly_mul(v, s[i], q)
        acc = poly_sub(acc, product, q)

    recovered = []
    for coeff in acc:
        recovered.append(0 if coeff < q // 2 else 1)

    return recovered


# ---------------------------------------------------------------------
# Kyber Decapsulation
# ---------------------------------------------------------------------

def kyber_decapsulate(ciphertext: Dict, private_key: Dict) -> int:
    """
    Performs Kyber decapsulation using the private key.

    Parameters:
        ciphertext: Dictionary containing ciphertext components
        private_key: Dictionary containing Kyber private key

    Returns:
        shared_secret: Recovered shared secret
    """
    u = ciphertext["u"]
    v = ciphertext["v"]
    params = ciphertext["params"]

    s = private_key["s"]

    n = params["n"]
    q = params["q"]

    # Step 1: Recover message
    m_recovered = recover_message(v, s, q)

    # Step 2: Derive shared secret
    shared_secret = derive_shared_secret(m_recovered)

    return shared_secret


# =====================================================================
# END OF PART 5
# =====================================================================
# =====================================================================
# PART 6: BENCHMARKING ENGINE (LATENCY CEILING MEASUREMENT)
# =====================================================================

# ---------------------------------------------------------------------
# Benchmarking Core
# ---------------------------------------------------------------------

def benchmark_keygen(params: Dict, iterations: int) -> Dict[str, float]:
    """
    Benchmarks Kyber key generation latency.
    """
    latencies = []

    for _ in range(iterations):
        start = current_time_ms()
        kyber_keygen(params)
        end = current_time_ms()
        latencies.append(end - start)

    return compute_latency_statistics(latencies)


def benchmark_encapsulation(public_key: Dict, iterations: int) -> Dict[str, float]:
    """
    Benchmarks Kyber encapsulation latency.
    """
    latencies = []

    for _ in range(iterations):
        start = current_time_ms()
        kyber_encapsulate(public_key)
        end = current_time_ms()
        latencies.append(end - start)

    return compute_latency_statistics(latencies)


def benchmark_decapsulation(
    ciphertext: Dict,
    private_key: Dict,
    iterations: int
) -> Dict[str, float]:
    """
    Benchmarks Kyber decapsulation latency.
    """
    latencies = []

    for _ in range(iterations):
        start = current_time_ms()
        kyber_decapsulate(ciphertext, private_key)
        end = current_time_ms()
        latencies.append(end - start)

    return compute_latency_statistics(latencies)


# ---------------------------------------------------------------------
# End-to-End KEM Benchmark
# ---------------------------------------------------------------------

def benchmark_kyber_kem(params: Dict) -> Dict[str, Dict[str, float]]:
    """
    Benchmarks KeyGen, Encapsulation, and Decapsulation for Kyber.
    """
    results = {}

    # Key Generation Benchmark
    keygen_stats = benchmark_keygen(params, BENCHMARK_ITERATIONS)
    results["keygen"] = keygen_stats

    # Generate single keypair for encaps/decaps benchmarks
    public_key, private_key = kyber_keygen(params)

    # Encapsulation Benchmark
    encaps_stats = benchmark_encapsulation(public_key, BENCHMARK_ITERATIONS)
    results["encapsulation"] = encaps_stats

    # Prepare single ciphertext for decapsulation benchmark
    ciphertext, _ = kyber_encapsulate(public_key)

    # Decapsulation Benchmark
    decaps_stats = benchmark_decapsulation(
        ciphertext,
        private_key,
        BENCHMARK_ITERATIONS
    )
    results["decapsulation"] = decaps_stats

    return results


# ---------------------------------------------------------------------
# Latency Ceiling Extraction
# ---------------------------------------------------------------------

def extract_latency_ceiling(benchmark_results: Dict[str, Dict[str, float]]) -> float:
    """
    Extracts the maximum observed latency across all operations.
    """
    max_latencies = [
        benchmark_results["keygen"]["max"],
        benchmark_results["encapsulation"]["max"],
        benchmark_results["decapsulation"]["max"]
    ]
    return max(max_latencies)


# =====================================================================
# END OF PART 6
# =====================================================================
# =====================================================================
# PART 7: OPTIMIZATION GAP & SCADA FEASIBILITY ANALYSIS
# =====================================================================

# ---------------------------------------------------------------------
# Optimization Gap Computation
# ---------------------------------------------------------------------

def compute_optimization_gap(latency_ceiling_ms: float) -> Dict[str, float]:
    """
    Computes the optimization gap relative to SCADA constraints.
    """
    gap = latency_ceiling_ms - SCADA_MAX_LATENCY_MS

    if latency_ceiling_ms > 0:
        required_speedup = latency_ceiling_ms / SCADA_MAX_LATENCY_MS
    else:
        required_speedup = 0.0

    return {
        "latency_ceiling_ms": latency_ceiling_ms,
        "scada_target_ms": SCADA_MAX_LATENCY_MS,
        "optimization_gap_ms": gap,
        "required_speedup_factor": required_speedup
    }


# ---------------------------------------------------------------------
# Feasibility Classification
# ---------------------------------------------------------------------

def classify_feasibility(latency_ms: float) -> str:
    """
    Classifies feasibility of cryptographic operation for SCADA use.
    """
    if latency_ms <= SCADA_MAX_LATENCY_MS:
        return "FEASIBLE"
    elif latency_ms <= 2 * SCADA_MAX_LATENCY_MS:
        return "MARGINALLY FEASIBLE"
    else:
        return "NOT FEASIBLE"


# ---------------------------------------------------------------------
# Full SCADA Evaluation Pipeline
# ---------------------------------------------------------------------

def evaluate_scada_feasibility(benchmark_results: Dict[str, Dict[str, float]]) -> Dict:
    """
    Evaluates SCADA feasibility using benchmark data.
    """
    latency_ceiling = extract_latency_ceiling(benchmark_results)

    gap_metrics = compute_optimization_gap(latency_ceiling)

    feasibility = classify_feasibility(latency_ceiling)

    return {
        "latency_ceiling_ms": latency_ceiling,
        "feasibility": feasibility,
        "optimization_metrics": gap_metrics
    }


# ---------------------------------------------------------------------
# Reporting Utility
# ---------------------------------------------------------------------

def format_scada_report(evaluation: Dict) -> str:
    """
    Formats SCADA feasibility evaluation into a readable report.
    """
    report = []
    report.append("SCADA FEASIBILITY ANALYSIS")
    report.append("--------------------------")
    report.append(f"Latency Ceiling (ms): {evaluation['latency_ceiling_ms']:.3f}")
    report.append(f"SCADA Target (ms): {SCADA_MAX_LATENCY_MS:.3f}")
    report.append(f"Feasibility Status: {evaluation['feasibility']}")
    report.append(
        f"Required Speedup Factor: "
        f"{evaluation['optimization_metrics']['required_speedup_factor']:.2f}x"
    )

    return "\n".join(report)


# =====================================================================
# END OF PART 7
# =====================================================================
# =====================================================================
# PART 8: DEPLOYMENT ARCHITECTURE EVALUATION
# =====================================================================

# ---------------------------------------------------------------------
# Architecture Latency Models
# ---------------------------------------------------------------------

def proxy_based_latency(crypto_latency_ms: float, hops: int = 2) -> float:
    """
    Estimates total latency for a proxy-based PQC deployment.
    """
    network_overhead_ms = 1.5 * hops
    return crypto_latency_ms + network_overhead_ms


def endpoint_based_latency(crypto_latency_ms: float) -> float:
    """
    Estimates total latency for direct endpoint PQC deployment.
    """
    return crypto_latency_ms


# ---------------------------------------------------------------------
# Architecture Comparison
# ---------------------------------------------------------------------

def evaluate_deployment_architectures(latency_ceiling_ms: float) -> Dict:
    """
    Compares proxy-based and endpoint-based architectures.
    """
    proxy_latency = proxy_based_latency(latency_ceiling_ms)
    endpoint_latency = endpoint_based_latency(latency_ceiling_ms)

    return {
        "proxy_based_latency_ms": proxy_latency,
        "endpoint_based_latency_ms": endpoint_latency,
        "proxy_feasible": proxy_latency <= SCADA_MAX_LATENCY_MS,
        "endpoint_feasible": endpoint_latency <= SCADA_MAX_LATENCY_MS
    }


# ---------------------------------------------------------------------
# Architecture Decision Logic
# ---------------------------------------------------------------------

def select_deployment_architecture(evaluation: Dict) -> str:
    """
    Selects appropriate deployment architecture based on feasibility.
    """
    if evaluation["endpoint_feasible"]:
        return "DIRECT_ENDPOINT_INTEGRATION"
    elif evaluation["proxy_feasible"]:
        return "PROXY_BASED_TUNNEL"
    else:
        return "NO_FEASIBLE_ARCHITECTURE"


# ---------------------------------------------------------------------
# Reporting Utility
# ---------------------------------------------------------------------

def format_architecture_report(evaluation: Dict, decision: str) -> str:
    """
    Formats deployment architecture analysis report.
    """
    report = []
    report.append("DEPLOYMENT ARCHITECTURE ANALYSIS")
    report.append("-------------------------------")
    report.append(
        f"Proxy-Based Latency (ms): "
        f"{evaluation['proxy_based_latency_ms']:.3f}"
    )
    report.append(
        f"Endpoint-Based Latency (ms): "
        f"{evaluation['endpoint_based_latency_ms']:.3f}"
    )
    report.append(f"Selected Architecture: {decision}")

    return "\n".join(report)


# =====================================================================
# END OF PART 8
# =====================================================================
# =====================================================================
# PART 9: EXPERIMENT RUNNER & OUTPUT GENERATION
# =====================================================================

# ---------------------------------------------------------------------
# Result Formatting Utilities
# ---------------------------------------------------------------------

def print_section(title: str):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def print_stats(label: str, stats: Dict[str, float]):
    print(f"\n{label}")
    print(f"  Mean Latency   : {stats['mean']:.3f} ms")
    print(f"  Median Latency : {stats['median']:.3f} ms")
    print(f"  Max Latency    : {stats['max']:.3f} ms")
    print(f"  Min Latency    : {stats['min']:.3f} ms")
    print(f"  Std Deviation  : {stats['std_dev']:.3f} ms")


# ---------------------------------------------------------------------
# Full Experiment Execution
# ---------------------------------------------------------------------

def run_full_experiment():
    """
    Executes full Kyber benchmarking and SCADA feasibility analysis
    across all supported security levels.
    """
    for level_name, params in KYBER_PARAMETER_SETS.items():
        print_section(f"KYBER EXPERIMENT: {level_name}")

        # -------------------------------
        # Benchmarking Phase
        # -------------------------------
        print("\nRunning benchmarking...")
        benchmark_results = benchmark_kyber_kem(params)

        print_stats("Key Generation", benchmark_results["keygen"])
        print_stats("Encapsulation", benchmark_results["encapsulation"])
        print_stats("Decapsulation", benchmark_results["decapsulation"])

        # -------------------------------
        # Latency Ceiling & SCADA Analysis
        # -------------------------------
        scada_eval = evaluate_scada_feasibility(benchmark_results)
        print("\n" + format_scada_report(scada_eval))

        # -------------------------------
        # Deployment Architecture Analysis
        # -------------------------------
        architecture_eval = evaluate_deployment_architectures(
            scada_eval["latency_ceiling_ms"]
        )
        architecture_decision = select_deployment_architecture(architecture_eval)

        print("\n" + format_architecture_report(
            architecture_eval,
            architecture_decision
        ))

        # -------------------------------
        # Memory Overhead Estimation
        # -------------------------------
        public_key, private_key = kyber_keygen(params)
        ciphertext, _ = kyber_encapsulate(public_key)

        print("\nMEMORY OVERHEAD ESTIMATION")
        print("-------------------------")
        print(f"Public Key Size   : {estimate_public_key_size(public_key)} bytes")
        print(f"Private Key Size  : {estimate_private_key_size(private_key)} bytes")
        print(f"Ciphertext Size   : {estimate_ciphertext_size(ciphertext)} bytes")


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_full_experiment()

# =====================================================================
# END OF PART 9
# =====================================================================


