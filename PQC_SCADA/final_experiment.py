import oqs
import time
from pymodbus.client.sync import ModbusTcpClient
import matplotlib.pyplot as plt

# Connect to PLC
client = ModbusTcpClient("127.0.0.1", port=5020)

if not client.connect():
    print("❌ Connection failed")
    exit()

results = []

print("Running experiment...\n")

# Run experiment
for i in range(200):
    start = time.perf_counter()

    # SCADA read
    result = client.read_holding_registers(0, 1)
    if result.isError():
        continue

    value = result.registers[0]

    # PQC operations
    with oqs.KeyEncapsulation("ML-KEM-512") as kem:
        pk = kem.generate_keypair()
        ct, ss1 = kem.encap_secret(pk)
        ss2 = kem.decap_secret(ct)

    end = time.perf_counter()

    latency = (end - start) * 1000  # ms
    results.append(latency)

    print(f"Run {i+1}: {latency:.3f} ms")

# =====================
# 📊 TABLE OUTPUT
# =====================

avg = sum(results)/len(results)
min_val = min(results)
max_val = max(results)

print("\n========== RESULT TABLE ==========")
print(f"Average Latency : {avg:.3f} ms")
print(f"Minimum Latency : {min_val:.3f} ms")
print(f"Maximum Latency : {max_val:.3f} ms")
print("=================================")

# =====================
# 📈 GRAPH
# =====================

plt.figure()
plt.plot(results)
plt.xlabel("Iteration")
plt.ylabel("Latency (ms)")
plt.title("End-to-End SCADA Latency (ML-KEM-512)")
plt.grid()

# Save graph
plt.savefig("latency_graph.png")

print("\n📊 Graph saved as latency_graph.png")

client.close()

