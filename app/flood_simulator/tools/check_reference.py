"""
check_model.mjs가 비교할 기준값을 만든다. Grid-level prediction.py의 predict_scenario와 같은 식을
scenario_params.json 원본 정밀도로 계산한다 (앱의 model.json은 소수점 4자리로 줄인 값).

실행:
    py "app\\flood_simulator\\tools\\check_reference.py"     (저장소 최상위 폴더에서)
"""
import json
import os

import numpy as np
from scipy.special import expit

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(TOOLS_DIR))), "flood_risk_ML")
PARAMS_PATH = os.path.join(BASE_DIR, "Grid-level prediction results", "scenario_params.json")
META_PATH = os.path.join(os.path.dirname(TOOLS_DIR), "data", "meta.json")
QUANTILE_CELLS = 200  # 분위수는 격자 일부만 비교


def predict(params, pc):
    alpha, beta = np.asarray(params["alpha"]), np.asarray(params["beta"])
    labels, log_q = np.asarray(params["cluster"]), np.asarray(params["log_q"])
    z = (np.asarray(pc[:3]) - np.asarray(params["pc_mean"])) / np.asarray(params["pc_std"])
    return expit((alpha + beta @ z)[:, labels] + log_q[None, :])


def main():
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
