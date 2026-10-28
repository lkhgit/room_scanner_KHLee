"""
클러스터 침수 모델(예측 모델 결과/posterior.nc)에 격자 비례상수 q_g를 곱해 격자별 침수 확률을 예측한다.

1. 클러스터 모델: η_c = α_c + β_c · z,  σ(η_c) = 그해 클러스터가 잠기는 사건(침수 비율 50% 이상)의 확률
2. 격자 비례상수: q_g = (k_g + m) / (k̄_c + m)
     k_g: 격자의 침수 연도 수, k̄_c: 클러스터 평균 침수 연도 수, m: 수축 강도
     클러스터마다 log q_g의 평균이 0이 되도록 다시 맞춘다.
3. 격자 침수 확률: 오즈_g = q_g × 오즈_c  ⇔  P_g = σ(η_c + log q_g)
4. 불확실성: 사후 표본 500개 각각으로 3단계를 계산해 평균과 5–95% 분위수로 요약

출력 (격자 예측 결과/)
  grid_flood_risk.csv   격자별 q_g, 연간 확률, 거주 L년 확률 (L = 2, 4)
  q_summary.csv         클러스터별 q_g 분포 점검
  scenario_params.json  앱에서 슬라이더 z로 실시간 계산할 때 쓰는 사후 표본과 격자별 log q_g
  m_cv.csv              m 선택용 연도 단위 교차검증 격자 Brier 점수

침수 이력이 있는 격자(grid_clusters.csv)만 계산한다. 나머지 격자는 '기록 없음'이다.

실행:
    & "C:\\Users\\KHLee\\AppData\\Local\\Python\\bin\\python.exe" "격자 예측.py"
"""
import importlib.util
import json
import os
import time

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit, logit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "예측 모델.py")
CLUSTERING_PATH = os.path.join(BASE_DIR, "필지 클러스터링.py")
MODEL_DIR = os.path.join(BASE_DIR, "예측 모델 결과")
POSTERIOR_PATH = os.path.join(MODEL_DIR, "posterior.nc")
CLUSTER_RISK_PATH = os.path.join(MODEL_DIR, "cluster_annual_risk.csv")
CV_PRED_PATH = os.path.join(MODEL_DIR, "cv_predictions.csv")
OUT_DIR = os.path.join(BASE_DIR, "격자 예측 결과")

SHRINK_M = 2
M_CANDIDATES = [1, 2, 4]
N_SAMPLES = 500
LIVING_YEARS = [2, 4]
SEED = 42


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 학습 연도, 표준화, 침수 이력 표는 클러스터 모델과 똑같이 만든다
cm = load_module("cluster_model", MODEL_PATH)


def grid_log_q(k, labels, m=SHRINK_M):
    """log q_g = log((k_g + m) / (k̄_c + m)), 클러스터마다 평균 0으로 다시 맞춘 값."""
    k_bar = k.groupby(labels).transform("mean")
    log_q = np.log((k + m) / (k_bar + m))
    return log_q - log_q.groupby(labels).transform("mean")


def sample_params(idata, n, rng):
    """사후 표본 n개를 비복원 추출한 α (n, C), β (n, C, K)."""
    alpha = cm.posterior_draws(idata, "alpha")
    beta = cm.posterior_draws(idata, "beta")
    idx = rng.choice(len(alpha), size=n, replace=False)
    return alpha[idx], beta[idx]


def grid_annual_draws(alpha, beta, labels, log_q, Z):
    """표본별 연간 P_g = 연도 z마다 σ(η_c + log q_g)를 계산해 연도 평균. -> (표본 수, 격자 수)."""
    eta_c = alpha[:, :, None] + np.einsum("sck,yk->scy", beta, Z)
    p = np.zeros((len(alpha), len(labels)))
    for y in range(Z.shape[0]):
        p += expit(eta_c[:, labels, y] + log_q[None, :])
    return p / Z.shape[0]


def summarize_draws(p, prefix):
    return {
        prefix: p.mean(axis=0),
        f"{prefix}_q05": np.quantile(p, 0.05, axis=0),
        f"{prefix}_q95": np.quantile(p, 0.95, axis=0),
    }


def risk_columns(p_annual, living_years=LIVING_YEARS):
    """연간 확률과 거주 L년 확률 1 − (1 − P_g)^L 을 표본마다 계산한 뒤 요약."""
    cols = summarize_draws(p_annual, "p_annual")
    for L in living_years:
        cols.update(summarize_draws(1 - (1 - p_annual) ** L, f"p_{L}yr"))
    return cols


def predict_scenario(params, z):
    """앱용: 슬라이더 z (길이 K)에 대한 격자별 확률의 평균, 5%, 95% 분위수."""
    alpha, beta = np.asarray(params["alpha"]), np.asarray(params["beta"])
    labels, log_q = np.asarray(params["cluster"]), np.asarray(params["log_q"])
    eta_c = alpha + beta @ np.asarray(z, dtype=float)
    p = expit(eta_c[:, labels] + log_q[None, :])
    return p.mean(axis=0), np.quantile(p, 0.05, axis=0), np.quantile(p, 0.95, axis=0)


# ---------------------------------------------------------------- 점검

def q_summary(k, labels, log_q):
    q = np.exp(log_q)
    df = pd.DataFrame({"cluster": labels, "k": k, "q": q})
    out = df.groupby("cluster").agg(
        n_cells=("q", "size"), k_mean=("k", "mean"), k_max=("k", "max"),
        q_min=("q", "min"), q_median=("q", "median"), q_max=("q", "max"),
        share_q_0_8_to_1_25=("q", lambda s: np.mean((s >= 0.8) & (s <= 1.25))),
    )
    return out.reset_index()


def check_against_cluster_csv(idata, z_all, n_pcs):
    """q_g = 1, 전체 사후 표본, cluster_annual_risk.csv와 같은 연도(PC가 있는 전 연도)로 계산하면 같아야 한다.
    전체 표본 x 전체 격자는 메모리가 커서 클러스터마다 격자 하나로 계산한다."""
    ref = pd.read_csv(CLUSTER_RISK_PATH, encoding="utf-8-sig").set_index("cluster")
    alpha, beta = cm.posterior_draws(idata, "alpha"), cm.posterior_draws(idata, "beta")
    one_per_cluster = np.arange(alpha.shape[1])
    p = grid_annual_draws(alpha, beta, one_per_cluster, np.zeros(len(one_per_cluster)),
                          z_all[cm.Z_COLS[:n_pcs]].values)
    mine = pd.DataFrame(risk_columns(p, [cm.LIVING_YEARS]), index=one_per_cluster)
    cols = ["p_annual", "p_annual_q05", "p_annual_q95", f"p_{cm.LIVING_YEARS}yr"]
    return float((mine[cols] - ref.loc[one_per_cluster, cols]).abs().max().max())


# ---------------------------------------------------------------- m 선택 교차검증

def m_cross_validation(grid, history, pcs, n_pcs):
    """연도 하나를 빼고: 재클러스터링(예측 모델 교차검증과 같은 fold) -> 학습 연도만으로 q_g ->
    cv_predictions.csv의 클러스터 예측(사후 평균 μ)에 log q_g를 더해 뺀 연도의 격자 침수 여부를 예측.
    q_g = 1(클러스터 확률 그대로)도 기준선으로 함께 비교한다."""
    clustering = load_module("grid_clustering", CLUSTERING_PATH)
    cv_pred = pd.read_csv(CV_PRED_PATH, encoding="utf-8-sig")
    cv_pred = cv_pred[cv_pred["model"] == f"hier_bernoulli_pc{n_pcs}"]
    xy = grid[["x", "y"]].values
    settings = [("q=1", None)] + [(f"m={m}", m) for m in M_CANDIDATES]

    rows = []
    for holdout in cm.TRAIN_YEARS:
        t = time.time()
        train_years = [y for y in cm.TRAIN_YEARS if y != holdout]
        hist_tr = history[train_years]
        keep = hist_tr.sum(axis=1) > 0
        _, _, _, labels, _, _ = clustering.cluster_cells(xy[keep.values], hist_tr[keep], pcs)
        labels = pd.Series(np.unique(labels, return_inverse=True)[1], index=hist_tr.index[keep])

        fold = cv_pred[cv_pred["year"] == holdout].set_index("cluster")
        n_cells = labels.value_counts().sort_index()
        if not np.array_equal(n_cells.values, fold["n_cells"].values):
            raise SystemExit(f"  {holdout}년 fold: 재클러스터링 결과가 cv_predictions.csv와 다릅니다.")

        eta_c = logit(np.clip(fold["pred"].values, 1e-9, 1 - 1e-9))[labels.values]
        y = history.loc[labels.index, holdout].values
        k = hist_tr[keep].sum(axis=1)
        for name, m in settings:
            log_q = 0.0 if m is None else grid_log_q(k, labels, m).values
            pred = expit(eta_c + log_q)
            rows.append({"year": holdout, "setting": name, "n_cells": len(y),
                         "brier": float(np.mean((pred - y) ** 2))})
        print(f"    {holdout}년 제외: 격자 {len(y):,}개, 실제 침수 {int(y.sum()):,}개, {time.time() - t:.0f}초")
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.width", 250)
    rng = np.random.default_rng(SEED)

    print("[1/5] 입력 로드")
    grid, pcs = cm.load_inputs()
    history = cm.history_from_combo(grid["combo"])
    labels = grid["cluster"]
    idata = az.from_netcdf(POSTERIOR_PATH)
    n_pcs = idata.posterior["beta"].shape[-1]
    z_all = cm.standardize(pcs, cm.TRAIN_YEARS)
    Z_train = z_all.loc[cm.TRAIN_YEARS, cm.Z_COLS[:n_pcs]].values
    print(f"  격자 {len(grid):,}개, 클러스터 {labels.nunique()}개, PC {n_pcs}개, "
          f"강우 시나리오 {len(cm.TRAIN_YEARS)}개 연도 ({cm.TRAIN_YEARS[0]}~{cm.TRAIN_YEARS[-1]}, 2015·2021 제외)")

    print(f"\n[2/5] 격자 비례상수 q_g (m = {SHRINK_M})")
    k = history.sum(axis=1)
    log_q = grid_log_q(k, labels)
    qs = q_summary(k, labels, log_q)
    qs.to_csv(os.path.join(OUT_DIR, "q_summary.csv"), index=False, encoding="utf-8-sig")
    print(qs.round(3).to_string(index=False))
    q = np.exp(log_q)
    print(f"  전체: q 최소 {q.min():.3f}, 최대 {q.max():.3f}, 0.8~1.25 안 비율 {np.mean((q >= 0.8) & (q <= 1.25)):.1%}")

    print("\n[3/5] 검산: q_g = 1이면 클러스터 연간 확률과 같아야 함")
    diff = check_against_cluster_csv(idata, z_all, n_pcs)
    print(f"  cluster_annual_risk.csv 기준 (전체 사후 표본, {len(z_all)}개 연도): 최대 차이 {diff:.2e} "
          f"-> {'일치' if diff < 1e-9 else '불일치'}")
    alpha, beta = sample_params(idata, N_SAMPLES, rng)
    p_c = expit(alpha[:, :, None] + np.einsum("sck,yk->scy", beta, Z_train)).mean(axis=2)
    p_q1 = grid_annual_draws(alpha, beta, labels.values, np.zeros(len(grid)), Z_train)
    diff14 = float(np.abs(p_q1 - p_c[:, labels.values]).max())
    print(f"  14개 연도, 표본 {N_SAMPLES}개 기준 클러스터 확률과 최대 차이 {diff14:.2e}")

    print(f"\n[4/5] 격자 침수 확률 (사후 표본 {N_SAMPLES}개)")
    p_annual = grid_annual_draws(alpha, beta, labels.values, log_q.values, Z_train)
    out = grid[["ix", "iy", "x", "y", "cluster"]].reset_index()
    out["k"], out["q"], out["log_q"] = k.values, q.values, log_q.values
    out = pd.concat([out, pd.DataFrame(risk_columns(p_annual))], axis=1)
    out.to_csv(os.path.join(OUT_DIR, "grid_flood_risk.csv"), index=False, encoding="utf-8-sig")
    by_cluster = out.groupby("cluster").agg(
        n_cells=("cell_id", "size"), p_cluster=("cluster", lambda c: p_c[:, c.iloc[0]].mean()),
        p_annual_min=("p_annual", "min"), p_annual_mean=("p_annual", "mean"), p_annual_max=("p_annual", "max"),
        p_2yr_mean=("p_2yr", "mean"), p_4yr_mean=("p_4yr", "mean"))
    print(by_cluster.round(3).to_string())

    mean, std = pcs.loc[cm.TRAIN_YEARS].mean(), pcs.loc[cm.TRAIN_YEARS].std(ddof=0)
    params = {
        "description": "P_g = sigmoid(alpha[s, c] + beta[s, c, :] · z + log_q[g]), z = (PC - pc_mean) / pc_std",
        "pcs": cm.PC_COLS[:n_pcs],
        "pc_mean": mean.iloc[:n_pcs].tolist(),
        "pc_std": std.iloc[:n_pcs].tolist(),
        "alpha": np.round(alpha, 6).tolist(),
        "beta": np.round(beta, 6).tolist(),
        "cell_id": out["cell_id"].tolist(),
        "cluster": out["cluster"].tolist(),
        "log_q": np.round(out["log_q"], 6).tolist(),
    }
    with open(os.path.join(OUT_DIR, "scenario_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)
    z0 = np.zeros(n_pcs)
    p0, lo0, hi0 = predict_scenario(params, z0)
    print(f"  시나리오 예시 (z = 0, 평년 강우): 격자 확률 평균 {p0.mean():.4f}, "
          f"5–95% 구간 폭 평균 {(hi0 - lo0).mean():.4f}")

    print(f"\n[5/5] m 선택 교차검증 (연도 단위, 격자 Brier, m = {M_CANDIDATES})")
    cv = m_cross_validation(grid, history, pcs, n_pcs)
    cv.to_csv(os.path.join(OUT_DIR, "m_cv.csv"), index=False, encoding="utf-8-sig")
    summary = cv.groupby("setting").apply(
        lambda g: pd.Series({"brier_cell_weighted": np.average(g["brier"], weights=g["n_cells"]),
                             "brier_year_mean": g["brier"].mean()}), include_groups=False)
    print(summary.sort_values("brier_cell_weighted").round(5).to_string())
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
