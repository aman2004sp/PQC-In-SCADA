# SCADA End-to-End Latency Simulation
# Based on Equation:
# L_total = L_KeyGen + L_Encaps + L_Decaps + 2 * L_hop

import pandas as pd

# Mean latency values from Table V (ms)
data = {
    "ML-KEM-512": {"KeyGen": 2.10, "Encaps": 2.40, "Decaps": 0.60},
    "ML-KEM-768": {"KeyGen": 3.00, "Encaps": 3.10, "Decaps": 0.70},
    "ML-KEM-1024": {"KeyGen": 4.20, "Encaps": 2.50, "Decaps": 0.80}
}

# Network hop delays (ms)
hop_values = [1, 2, 3]

results = []

for variant, ops in data.items():
    crypto_latency = ops["KeyGen"] + ops["Encaps"] + ops["Decaps"]

    row = {
        "Variant": variant,
        "Crypto": round(crypto_latency, 2)
    }

    for hop in hop_values:
        total_latency = crypto_latency + 2 * hop
        row[f"+{hop} ms hop"] = round(total_latency, 2)

    results.append(row)

# Convert to table
df = pd.DataFrame(results)

print(df) explain this code
