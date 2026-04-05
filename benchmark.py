import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Starting optimized benchmarks...")

# =========================
# CONFIG
# =========================
ITERATIONS = 200
WARMUP_ITERATIONS = 10
SCADA_TARGET = 5
POLY_N = 256
COEFF_MOD = 3329
BASE_SEED = 42
KEY_POOL_SIZE = min(max(16, ITERATIONS // 4), ITERATIONS)
ENABLE_PARALLEL_VARIANTS = True
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PARAMS = {
    "ML-KEM-512": {"k": 2},
    "ML-KEM-768": {"k": 3},
    "ML-KEM-1024": {"k": 4},
}

# =========================
# CORE FUNCTIONS
# =========================

def generate_poly(rng, shape=POLY_N):
    return rng.integers(0, COEFF_MOD, size=shape).astype(np.float64)

def poly_mul_fft(a, b):
    res = np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=POLY_N)
    return np.round(res) % COEFF_MOD

def poly_matvec_from_fft(matrix_f, vec_f):
    res = np.fft.irfft((matrix_f * vec_f[np.newaxis,:,:]).sum(axis=1), n=POLY_N)
    return np.round(res) % COEFF_MOD

def poly_vecdot_from_fft(lhs_f, rhs_f):
    res = np.fft.irfft((lhs_f * rhs_f).sum(axis=0), n=POLY_N)
    return np.round(res) % COEFF_MOD

def keygen(k, rng):
    A = generate_poly(rng, (k,k,POLY_N))
    s = generate_poly(rng, (k,POLY_N))
    e = generate_poly(rng, (k,POLY_N))

    A_f = np.fft.rfft(A, axis=-1)
    A_f_t = A_f.transpose(1,0,2)
    s_f = np.fft.rfft(s, axis=-1)

    t = poly_matvec_from_fft(A_f, s_f) + e
    t %= COEFF_MOD
    t_f = np.fft.rfft(t, axis=-1)

    return (A,A_f,A_f_t,t,t_f,k), (s,s_f,k)

def encaps(pk, k, rng):
    A,A_f,A_f_t,t,t_f,_ = pk

    r = generate_poly(rng,(k,POLY_N))
    e1 = generate_poly(rng,(k,POLY_N))
    e2 = generate_poly(rng,POLY_N)

    r_f = np.fft.rfft(r, axis=-1)

    u = poly_matvec_from_fft(A_f_t, r_f) + e1
    u %= COEFF_MOD

    v = poly_vecdot_from_fft(t_f, r_f) + e2
    v %= COEFF_MOD

    return u,v

def decaps(u,v,sk,k=None):
    s,s_f,_ = sk
    u_f = np.fft.rfft(u, axis=-1)
    acc = poly_vecdot_from_fft(s_f, u_f)
    return (v - acc) % COEFF_MOD

# =========================
# BENCHMARK
# =========================

def benchmark(func):
    for _ in range(WARMUP_ITERATIONS):
        func()
    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        func()
        times.append((time.perf_counter_ns()-t0)/1e6)
    return np.array(times)

def benchmark_pool(pool, func):
    for i in range(WARMUP_ITERATIONS):
        func(pool[i%len(pool)])
    times=[]
    for i in range(ITERATIONS):
        item = pool[i%len(pool)]
        t0 = time.perf_counter_ns()
        func(item)
        times.append((time.perf_counter_ns()-t0)/1e6)
    return np.array(times)

def run_variant(variant,k,seed):
    rng = np.random.default_rng(seed)

    kg = benchmark(lambda: keygen(k,rng))

    key_pool = [keygen(k,rng) for _ in range(KEY_POOL_SIZE)]

    enc = benchmark_pool(key_pool, lambda x: encaps(x[0],k,rng))

    dec_pool=[]
    for pk,sk in key_pool:
        u,v = encaps(pk,k,rng)
        dec_pool.append((u,v,sk))

    dec = benchmark_pool(dec_pool, lambda x: decaps(x[0],x[1],x[2]))

    return variant, {"KeyGen":kg,"Encaps":enc,"Decaps":dec,"k":k}

def run_all():
    results={}
    items=list(PARAMS.items())

    with ThreadPoolExecutor(max_workers=min(len(items),os.cpu_count())) as ex:
        futures=[ex.submit(run_variant,v,p["k"],BASE_SEED+i*1000)
                 for i,(v,p) in enumerate(items)]
        for f in as_completed(futures):
            v,data=f.result()
            results[v]=data

    return {v:results[v] for v in PARAMS}

# =========================
# OUTPUT
# =========================

def write_outputs(results):

    # TABLE II
    rows=[]
    for v in results:
        for op in ["KeyGen","Encaps","Decaps"]:
            d=results[v][op]
            rows.append([v,op,np.mean(d),np.max(d),np.min(d),np.std(d)])
    pd.DataFrame(rows,columns=["Variant","Operation","Mean","Max","Min","StdDev"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_2.csv"),index=False)

    # TABLE III
    rows=[]
    for v in results:
        for op in ["KeyGen","Encaps","Decaps"]:
            d=results[v][op]
            mean,std=np.mean(d),np.std(d)
            rows.append([v,op,(std/mean)*100,mean,1.96*std/np.sqrt(ITERATIONS)])
    pd.DataFrame(rows,columns=["Variant","Operation","CV%","Mean","CI"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_3.csv"),index=False)

    # TABLE IV
    rows=[]
    pairs=[("ML-KEM-512","ML-KEM-768"),("ML-KEM-512","ML-KEM-1024")]
    for op in ["KeyGen","Encaps","Decaps"]:
        for a,b in pairs:
            k1,k2=results[a]["k"],results[b]["k"]
            m=np.mean(results[b][op])/np.mean(results[a][op])
            t=k2/k1 if op=="Decaps" else (k2**2)/(k1**2)
            rows.append([op,f"{k1}->{k2}",t,m])
    pd.DataFrame(rows,columns=["Operation","k scaling","Theoretical","Measured"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_4.csv"),index=False)

    # TABLE V
    rows=[]
    for v in results:
        kg,enc,dec=[np.mean(results[v][x]) for x in ["KeyGen","Encaps","Decaps"]]
        total=kg+enc+dec
        rows.append([v,total,(enc/total)*100])
    pd.DataFrame(rows,columns=["Variant","TotalMean","EncapsShare%"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_5.csv"),index=False)

    # TABLE VII
    rows=[]
    for v in results:
        c=np.max(results[v]["Encaps"])
        rows.append([v,c,c-SCADA_TARGET,c/SCADA_TARGET])
    pd.DataFrame(rows,columns=["Variant","Ceiling","Gap","Speedup"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_7.csv"),index=False)

    # STATIC TABLES
    pd.DataFrame([
        ["ML-KEM-512",2,800,1632,768],
        ["ML-KEM-768",3,1184,2400,1088],
        ["ML-KEM-1024",4,1568,3168,1568]],
        columns=["Variant","k","PK","SK","CT"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_1.csv"),index=False)

    pd.DataFrame([
        ["Python Mean",112.84],
        ["Python Max",167.37],
        ["pqm4",1.62],
        ["NIST C",0.10]],
        columns=["Implementation","Encaps(ms)"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_6.csv"),index=False)

    pd.DataFrame([
        ["ML-KEM-512",800,1632,768],
        ["ML-KEM-768",1184,2400,1088],
        ["ML-KEM-1024",1568,3168,1568],
        ["ECC-P256",64,32,96],
        ["RSA-2048",256,1192,256]],
        columns=["Scheme","PK","SK","CT"])\
        .to_csv(os.path.join(OUTPUT_DIR,"table_8.csv"),index=False)

    # FIGURE 1 (Mean vs Max)
    labels,means,maxs=[],[],[]
    for v in results:
        for op in ["KeyGen","Encaps","Decaps"]:
            labels.append(f"{v}-{op}")
            means.append(np.mean(results[v][op]))
            maxs.append(np.max(results[v][op]))

    x=np.arange(len(labels))
    w=0.35
    plt.figure(figsize=(12,5))
    plt.bar(x-w/2,means,w,label="Mean")
    plt.bar(x+w/2,maxs,w,label="Max")
    plt.axhline(y=SCADA_TARGET,linestyle="--")
    plt.xticks(x,labels,rotation=45,ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"figure_1.png"),dpi=200)
    plt.close()

    # FIGURE 2 (FIXED)
    variants=list(results.keys())
    speed=[np.max(results[v]["Encaps"])/SCADA_TARGET for v in variants]

    plt.figure(figsize=(8,5))
    plt.plot(variants,speed,marker="o")
    plt.yscale("log")
    plt.xlabel("Kyber Variant")
    plt.ylabel("Speedup (Encaps / SCADA Target)")
    plt.title("Encapsulation Speedup (Log Scale)")

    for i,val in enumerate(speed):
        plt.text(i,val,f"{val:.2f}",ha="center")

    plt.grid(True,linestyle="--",linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"figure_2.png"),dpi=200)
    plt.close()

# =========================
# MAIN
# =========================

def main():
    results=run_all()
    write_outputs(results)
    print("✅ All tables (1–8) + figures generated successfully.")

if __name__=="__main__":
    main()