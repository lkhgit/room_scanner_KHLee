"""
Writes the reference values check_model.mjs compares against. Uses the same formula as predict_scenario in
Grid-level prediction.py, at the full precision of scenario_params.json (the app's model.json is rounded to 4 decimals).

Run:
    py "app\\flood_simulator\\tools\\check_reference.py"     (from the repository root)
"""
import json
import os

import numpy as np
from scipy.special import expit

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(TOOLS_DIR))), "flood_risk_ML")
PARAMS_PATH = os.path.join(BASE_DIR, "Grid-level prediction results", "scenario_params.json")
META_PATH = os.path.join(os.path.dirname(TOOLS_DIR), "data", "meta.json")
QUANTILE_CELLS = 200  # quantiles are compared for a sample of cells only


def predict(params, pc):
    """Flood probability of every posterior sample and grid cell for the slider values pc (full precision)."""
    alpha, beta = np.asarray(params["alpha"]), np.asarray(params["beta"])
    labels, log_q = np.asarray(params["cluster"]), np.asarray(params["log_q"])
    z = (np.asarray(pc[:3]) - np.asarray(params["pc_mean"])) / np.asarray(params["pc_std"])
    return expit((alpha + beta @ z)[:, labels] + log_q[None, :])


def main():
    """Writes reference.json with the mean probabilities and sampled quantiles for every preset."""
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    rng = np.random.default_rng(0)
    cells = rng.choice(len(params["cell_id"]), QUANTILE_CELLS, replace=False)

    scenarios = []
    for preset in meta["presets"]:
        p = predict(params, preset["pc"])
        q05, q95 = np.quantile(p[:, cells], 0.05, axis=0), np.quantile(p[:, cells], 0.95, axis=0)
        scenarios.append({
            "label": preset["label"],
            "pc": preset["pc"],
            "mean": p.mean(axis=0).tolist(),
            "quantiles": [[int(g), float(a), float(b)] for g, a, b in zip(cells, q05, q95)],
        })
        print(f"  {preset['label']}: 격자 평균 확률 {p.mean():.4f}")
    with open(os.path.join(TOOLS_DIR, "reference.json"), "w", encoding="utf-8") as f:
        json.dump({"scenarios": scenarios}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
