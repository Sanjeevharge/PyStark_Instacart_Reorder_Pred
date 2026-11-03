"""Runtime comparison: pandas vs Spark naive vs Spark tuned at 3 data sizes (median of REPS)."""
import os, re, statistics, subprocess, sys
import pandas as pd

REPS = int("1")
RUNS = [("pandas", "pandas_features.py", []),
        ("spark_naive", "spark_features.py", ["--mode", "naive"]),
        ("spark_tuned", "spark_features.py", ["--mode", "tuned"])]
res = []
for frac in [float(x) for x in os.environ.get("FRACS", "0.1,0.5,1.0").split(",")]:
    for name, script, extra in RUNS:
        ts, rows = [], None
        for r in range(REPS):
            cmd = [sys.executable, script, "--data", "data", "--out", f"out/bench_{name}",
                   "--frac", str(frac)] + extra
            p = subprocess.run(cmd, capture_output=True, text=True)
            m = re.search(r"ELAPSED_SEC=([\d.]+) ROWS=(\d+)", p.stdout)
            if not m:
                print(f"FAIL {name} frac={frac}: {p.stderr[-300:]}"); break
            ts.append(float(m.group(1))); rows = int(m.group(2))
        res.append(dict(frac=frac, impl=name, rows=rows,
                        median_min=round(statistics.median(ts) / 60, 2) if ts else None))
        print(res)
df = pd.DataFrame(res)
df.to_csv("results_runtime.csv", index=False)
print(df.pivot(index="frac", columns="impl", values="median_min"))
print("rows (must match across impls):\n", df.pivot(index="frac", columns="impl", values="rows"))