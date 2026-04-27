import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "collect_data.py",
    "extract_features.py",
    "predict.py",
    "visualize.py",
    "interpret_results.py",
]


def main():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(ROOT_DIR / ".mplcache")

    for script in SCRIPTS:
        script_path = ROOT_DIR / "src" / script
        print(f"Running src/{script}")
        subprocess.run([sys.executable, str(script_path)], cwd=ROOT_DIR, env=env, check=True)

    print("Pipeline finished.")
    print("Outputs are in runs/hw2_full_seed42/")


if __name__ == "__main__":
    main()
