#!/usr/bin/env python3
import os
import re
import json
import sys
import csv
import argparse
from pathlib import Path
from statistics import mean, stdev

# ------------------------
# Helpers semplici
# ------------------------

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def get_nested(d, path):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def extract_power(payload):
    """Restituisce potenza in mW."""
    if payload is None: 
        return None
    rails = [
        "power.rail.VDD_GPU.power",
        "power.rail.VDD_CPU_GPU_CV.power",
        "power.tot.power"
    ]
    for r in rails:
        v = get_nested(payload, r)
        if isinstance(v, (int, float)):
            return float(v)
    return None

def extract_ram(payload):
    """RAM in MB."""
    if payload is None:
        return None
    v = get_nested(payload, "mem.RAM.used")
    if isinstance(v, (int, float)):
        return v / 1024.0
    return None

def extract_latencies(payload):
    """Estrae tutte le latenze in ms da un file di times."""
    if payload is None:
        return []
    vals = []
    if isinstance(payload, list):
        for e in payload:
            if isinstance(e, dict) and "latencyMs" in e:
                vals.append(float(e["latencyMs"]))
    if isinstance(payload, dict):
        if "latencyMs" in payload:
            vals.append(float(payload["latencyMs"]))
        for key in ("entries", "iterations", "data"):
            arr = payload.get(key)
            print(arr)
            if isinstance(arr, list):
                for e in arr:
                    if isinstance(e, dict) and "latencyMs" in e:
                        vals.append(float(e["latencyMs"]))
    return vals

# ------------------------
# Core semplice
# ------------------------



RUN_NN   = re.compile(r"^NN_(?!times)(.+)\.json$")

RUN_NN_T = re.compile(r"^NN_times(?:_(.+))?\.json$")


def scan_runs(folder):
    runs = {}
    for f in os.listdir(folder):
        print(f)
        m = RUN_NN.match(f)
        if m:
            rid = m.group(1)
            runs.setdefault(rid, {})["NN"] = os.path.join(folder, f)
        m = RUN_NN_T.match(f)
        if m:
            rid = m.group(1)
            runs.setdefault(rid, {})["NN_t"] = os.path.join(folder, f)
    return runs

def process_folder(folder_path):
    # runs = scan_runs(folder_path)
    runs = {}
    for run_path in [path for path in os.listdir(folder_path) if path.endswith('json')]:
        run_id = run_path.split('_')[1].split('.')[0]
        root_run_path = os.path.join(folder_path, run_path)
        if 'time' in root_run_path:
            runs.setdefault(0, {})['NN_t'] = root_run_path
        else:
            runs.setdefault(run_id, {})['NN'] = root_run_path

    if not runs:
        return None
    power_list, ram_list = [], []

    # --- 1) POWER & RAM per-run ---
    for rid, paths in runs.items():
        # prendi i payload last/qnet (se ci sono)
        print(rid)
        NN_path = load_json(paths.get("NN"))

        # Somma power e ram dei due payload disponibili
        p = 0.0
        r = 0.0
        
        v = extract_power(NN_path)
        if isinstance(v, float):
            p += v
        rv = extract_ram(NN_path)
        if isinstance(rv, float):
            r += rv

        # Consideriamo la run valida se abbiamo almeno una misura di power
        # (se vuoi richiedere power>0, lascia così; altrimenti puoi controllare con un flag)
        if p > 0.0:
            power_list.append(p)
            ram_list.append(r)

    if not power_list:
        return None

    # Aggregazione power/ram tra run
    if len(power_list) == 1:
        power_mean, power_std = power_list[0], 0.0
        ram_mean, ram_std = ram_list[0], 0.0
    else:
        power_mean, power_std = mean(power_list), stdev(power_list)
        ram_mean, ram_std = mean(ram_list), stdev(ram_list)

    # --- 2) LATENCY: raccogli da TUTTI i times file (indipendentemente dalla run) ---
    # I file dei tempi non sono per-run: li mettiamo in un set per non duplicarli.
    times_paths = set()
    for _, paths in runs.items():
        if paths.get("NN_t"):
            times_paths.add(paths["NN_t"])
    print(times_paths)
    latency_samples = []
    for tpath in times_paths:
        latency_samples += extract_latencies(load_json(tpath))

    # Se non ci sono latenze, non possiamo calcolare l'energia media (né riportare latenza)
    if not latency_samples:
        return None

    if len(latency_samples) == 1:
        latency_mean, latency_std = latency_samples[0], 0.0
    else:
        latency_mean, latency_std = mean(latency_samples), stdev(latency_samples)

    # --- 3) ENERGIA: prodotto delle medie, come richiesto ---
    energy_mean = power_mean * latency_mean  # mW * ms = microjoule
    # energy_std: lasciamo vuoto/None per semplicità
    energy_std = None

    # (Opzionale) Stima std(energia) con propagazione degli errori per prodotto delle medie:
    # Assumendo indipendenza: Var(P̄*L̄) ≈ (E[L̄]^2 Var(P̄)) + (E[P̄]^2 Var(L̄))
    # dove Var(P̄)=sP^2/nP e Var(L̄)=sL^2/nL
    # nP = len(power_list); nL = len(latency_samples)
    # if len(power_list) > 1 and len(latency_samples) > 1:
    #     varPbar = (power_std ** 2) / len(power_list)
    #     varLbar = (latency_std ** 2) / len(latency_samples)
    #     import math
    #     energy_std = math.sqrt((latency_mean ** 2) * varPbar + (power_mean ** 2) * varLbar)

    folder_name = os.path.basename(folder_path)

    return (
        folder_name,
        len(power_list),           # runs effettivamente usate per power/ram
        power_mean, power_std,
        latency_mean, latency_std,
        ram_mean, ram_std,
        energy_mean, energy_std
    )
# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_root_path")
    ap.add_argument("-o", default="metrics.csv")
    args = ap.parse_args()

    rows = []
    # for d, dirs, files in os.walk(args.root):
    #     if any(re.match(r"(last|qnet|last_times|qnet_times)_", f) for f in files):
    #         res = process_folder(d)
    #         if res:
    #             rows.append(res)

    res = process_folder(args.plan_root_path)
    rows.append(res)

    out_path = Path(args.o).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.o, 'report.csv'), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "folder", "runs",
            "power_mean_mW", "power_std_mW",
            "latency_mean_ms", "latency_std_ms",
            "ram_mean_MB", "ram_std_MB",
            "energy_mean_uJ", "energy_std_uJ"
        ])
        for r in rows:
            w.writerow(r)

    print(f"Done. Wrote {len(rows)} rows to {args.o}")


if __name__ == "__main__":
    main()