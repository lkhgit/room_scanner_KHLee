"""
강우 PC(PC1~PC4)로 클러스터별 연간 침수 비율을 예측하는 계층 로지스틱 회귀 모델.
('클러스터 침수 비율 계층 로지스틱 회귀 계획' 문서 구현)

1. 학습 테이블: combo를 '+'로 나눠 격자 x 침수 연도 -> 클러스터 x 연도의 침수 비율 r = k / n
2. PC 표준화: 학습 연도 기준 축별 표준화 (z1~z4)
3. 연도 단위 교차검증: 뺀 연도를 제외하고 격자 클러스터링부터 다시 한 뒤,
   기후 기준선 / 완전 풀링 / 클러스터별 Firth / 계층 모델(베타, 베르누이 x PC 3개, 4개)을 비교
4. 최종 모델: 교차검증 Brier 점수가 가장 낮은 계층 모델 설정으로 전체 데이터 학습
5. 점검: 사전 예측 점검, 수렴 기준(R-hat, ESS, divergence), 사후 예측 점검, 계수 수축
6. 출력: 클러스터별 / 격자별 연간 침수 확률과 90% 구간, 2년 거주 기준 확률

침수흔적도 자료가 없는 2015, 2021년은 '침수 없음'이 아니라 결측이므로 학습과 검증에서 뺀다.
강우 시나리오(연간 확률 평균)에는 PC가 있는 모든 연도(2010~2026)를 쓴다.

실행:
    & "C:\\Users\\KHLee\\AppData\\Local\\Python\\bin\\python.exe" "cluster-level prediction.py"
"""
import importlib.util
import os
import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(BASE_DIR, "Grid clustering results", "grid_clusters.csv")
PC_PATH = os.path.join(BASE_DIR, "rainfall data final ver.csv")
CLUSTERING_PATH = os.path.join(BASE_DIR, "grid clustering.py")
OUT_DIR = os.path.join(BASE_DIR, "Prediction model results")

NO_DATA_YEARS = [2015, 2021]  # 침수흔적도 자료 없음
TRAIN_YEARS = [y for y in range(2010, 2026) if y not in NO_DATA_YEARS]
PC_COLS = ["PC1", "PC2", "PC3", "PC4"]
Z_COLS = ["z1", "z2", "z3", "z4"]
LIVING_YEARS = 2

CHAINS = 4
DRAWS, TUNE = 2000, 2000
CV_DRAWS, CV_TUNE = 1000, 1000  # 교차검증은 fold가 많아 표본 수를 줄인다
TARGET_ACCEPT, RETRY_TARGET_ACCEPT = 0.95, 0.99
SEED = 42

RHAT_MAX, ESS_MIN = 1.01, 400
CV_HIER_SETTINGS = [("beta", 4), ("bernoulli", 4), ("beta", 3), ("bernoulli", 3)]
RELIABILITY_BINS = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_clustering_module():
    """'grid clustering.py'의 cluster_cells를 교차검증 재클러스터링에 쓴다."""
    spec = importlib.util.spec_from_file_location("grid_clustering", CLUSTERING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs():
    grid = pd.read_csv(GRID_PATH, encoding="utf-8-sig").set_index("cell_id")
    pcs = pd.read_csv(PC_PATH, encoding="utf-8-sig").set_index("year")[PC_COLS]
    return grid, pcs


def history_from_combo(combo):
    """combo('2010+2022' 형태)를 '+'로 나눠 격자 x 학습 연도 침수 여부(0/1) 표를 만든다."""
    years = combo.str.split("+").explode().astype(int)
    history = pd.crosstab(years.index, years).reindex(index=combo.index, columns=TRAIN_YEARS, fill_value=0)
    history.index.name = "cell_id"
    return history


def build_table(history, labels, years):
    """클러스터 x 연도 학습 테이블: n_cells, n_flooded, r."""
    n_cells = labels.value_counts().sort_index()
    k = history[years].groupby(labels).sum()
    table = k.stack().reset_index()
    table.columns = ["cluster", "year", "n_flooded"]
    table["n_cells"] = table["cluster"].map(n_cells)
    table["r"] = table["n_flooded"] / table["n_cells"]
    return table[["cluster", "year", "n_cells", "n_flooded", "r"]]


def standardize(pcs, years):
    """PC를 축마다 학습 연도의 평균, 표준편차로 표준화. 모든 연도(시나리오 포함)에 같은 값을 적용."""
    mean, std = pcs.loc[years].mean(), pcs.loc[years].std(ddof=0)
    z = (pcs - mean) / std
    z.columns = Z_COLS
    return z


def shrink_ratio(r, n_rows):
    """베타 분포가 0, 1을 다루도록 안쪽으로 당긴 r*."""
    return (r * (n_rows - 1) + 0.5) / n_rows


def build_model(table, n_pcs, obs, pooled=False):
    """계층 로지스틱 회귀 (비중심화). pooled=True면 모든 클러스터에 α, β 한 세트(완전 풀링)."""
    n_clusters = int(table["cluster"].max()) + 1
    cluster_idx = table["cluster"].values
    Z = table[Z_COLS[:n_pcs]].values
    coords = {"cluster": np.arange(n_clusters), "pc": PC_COLS[:n_pcs], "obs": np.arange(len(table))}

    with pm.Model(coords=coords) as model:
        alpha_bar = pm.Normal("alpha_bar", -3, 2)
        beta_bar_1 = pm.HalfNormal("beta_bar_1", 1.5)
        beta_bar_rest = pm.Normal("beta_bar_rest", 0, 1.5, shape=n_pcs - 1)
        beta_bar = pm.Deterministic("beta_bar", pt.concatenate([beta_bar_1[None], beta_bar_rest]), dims="pc")

        if pooled:
            alpha = pm.Deterministic("alpha", alpha_bar + pt.zeros(n_clusters), dims="cluster")
            beta = pm.Deterministic("beta", pt.ones((n_clusters, 1)) * beta_bar[None, :], dims=("cluster", "pc"))
        else:
            tau_alpha = pm.HalfNormal("tau_alpha", 1)
            tau_beta = pm.HalfNormal("tau_beta", 1, dims="pc")
            eps_alpha = pm.Normal("eps_alpha", 0, 1, dims="cluster")
            eps_beta = pm.Normal("eps_beta", 0, 1, dims=("cluster", "pc"))
            alpha = pm.Deterministic("alpha", alpha_bar + tau_alpha * eps_alpha, dims="cluster")
            beta = pm.Deterministic("beta", beta_bar[None, :] + tau_beta[None, :] * eps_beta, dims=("cluster", "pc"))

        eta = alpha[cluster_idx] + (beta[cluster_idx] * Z).sum(axis=1)
        mu = pm.Deterministic("mu", pm.math.sigmoid(eta), dims="obs")

        if obs == "beta":
            phi = pm.Gamma("phi", alpha=2, beta=0.1)
            y = shrink_ratio(table["r"].values, len(table))
            pm.Beta("r_obs", alpha=mu * phi, beta=(1 - mu) * phi, observed=y, dims="obs")
        else:
            y = (table["r"].values >= 0.5).astype(int)
            pm.Bernoulli("y_obs", p=mu, observed=y, dims="obs")
    return model


def diagnose(idata, model):
    """수렴 기준: 모든 자유 파라미터의 R-hat < 1.01, ESS > 400, divergence 0개."""
    var_names = [v.name for v in model.free_RVs]
    summary = az.summary(idata, var_names=var_names, round_to="none")
    return {
        "rhat_max": float(summary["r_hat"].max()),
        "ess_min": float(summary[["ess_bulk", "ess_tail"]].min().min()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
    }


def is_converged(diag):
    return diag["rhat_max"] < RHAT_MAX and diag["ess_min"] > ESS_MIN and diag["divergences"] == 0


def fit(table, n_pcs, obs, pooled=False, draws=DRAWS, tune=TUNE):
    """NUTS 샘플링. divergence가 남으면 target_accept를 올려 한 번 다시 샘플링."""
    model = build_model(table, n_pcs, obs, pooled)
    for target_accept in (TARGET_ACCEPT, RETRY_TARGET_ACCEPT):
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=CHAINS, cores=CHAINS, target_accept=target_accept,
                              random_seed=SEED, progressbar=False)
        diag = diagnose(idata, model)
        diag["target_accept"] = target_accept
        if diag["divergences"] == 0:
            break
    return model, idata, diag


def posterior_draws(idata, name):
    values = idata.posterior[name].values
    return values.reshape(-1, *values.shape[2:])


def predict_mu(idata, Z):
    """사후 표본별 μ. Z: (연도 수, PC 수) -> (표본 수, 클러스터 수, 연도 수)."""
    alpha = posterior_draws(idata, "alpha")
    beta = posterior_draws(idata, "beta")
    return expit(alpha[:, :, None] + np.einsum("sck,yk->scy", beta, Z))


def firth_logit(X, y, max_iter=200, tol=1e-8):
    """Firth 벌점 로지스틱 회귀 (y는 0~1 비율 허용). 완벽 분리에서도 계수가 유한하게 남는다."""
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = expit(X @ b)
        w = mu * (1 - mu)
        inv = np.linalg.pinv(X.T @ (X * w[:, None]))
        h = np.einsum("ij,jk,ik->i", X, inv, X) * w
        step = inv @ (X.T @ (y - mu + h * (0.5 - mu)))
        step *= min(1.0, 5.0 / np.abs(step).max()) if np.abs(step).max() > 0 else 1.0
        b += step
        if np.abs(step).max() < tol:
            break
    return b


def score(r, pred):
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.mean((pred - r) ** 2)),
        "log_loss": float(np.mean(-(r * np.log(p) + (1 - r) * np.log(1 - p)))),
    }


# ---------------------------------------------------------------- 교차검증

def cv_fold(holdout, history, xy, pcs, clustering):
    """한 연도를 빼고 재클러스터링 -> 학습 -> 뺀 연도 예측. 예측 행과 '예측 불가' 비율을 반환."""
    train_years = [y for y in TRAIN_YEARS if y != holdout]
    hist_tr = history[train_years]
    keep = hist_tr.sum(axis=1) > 0
    _, _, _, labels, _, _ = clustering.cluster_cells(xy[keep.values], hist_tr[keep], pcs)
    labels = pd.Series(np.unique(labels, return_inverse=True)[1], index=hist_tr.index[keep])

    table = build_table(hist_tr[keep], labels, train_years)
    z = standardize(pcs, train_years)
    table = table.join(z, on="year")
    z_hold = z.loc[[holdout]].values

    # 뺀 연도의 실제 비율 (1단계 클러스터에 대응)
    k_hold = history.loc[keep[keep].index, holdout].groupby(labels).sum()
    n_cells = labels.value_counts().sort_index()
    actual = pd.DataFrame({"cluster": n_cells.index, "n_cells": n_cells.values,
                           "r": (k_hold / n_cells).values})
    total_flooded = int(history[holdout].sum())
    unpredictable = int(history.loc[~keep, holdout].sum())

    preds = []

    def add(model_name, pred, lo=None, hi=None, diag=None):
        df = actual.copy()
        df["model"], df["pred"] = model_name, pred
        df["lo"] = np.nan if lo is None else lo
        df["hi"] = np.nan if hi is None else hi
        df["converged"] = True if diag is None else is_converged(diag)
        preds.append(df)

    add("climatology", table.groupby("cluster")["r"].mean().values)

    firth = []
    for c, g in table.groupby("cluster"):
        X = np.column_stack([np.ones(len(g)), g[Z_COLS].values])
        b = firth_logit(X, g["r"].values)
        firth.append(expit(b[0] + z_hold[0] @ b[1:]))
    add("firth", np.array(firth))

    n_rows = len(table)
    rng = np.random.default_rng(SEED)
    settings = [("pooled", "beta", 4, True)] + [(f"hier_{o}_pc{k}", o, k, False) for o, k in CV_HIER_SETTINGS]
    for name, obs, k, pooled in settings:
        _, idata, diag = fit(table, k, obs, pooled=pooled, draws=CV_DRAWS, tune=CV_TUNE)
        mu = predict_mu(idata, z_hold[:, :k])[:, :, 0]
        lo = hi = None
        if obs == "beta":
            phi = posterior_draws(idata, "phi")[:, None]
            r_rep = rng.beta(mu * phi, (1 - mu) * phi)
            # 구간은 r* 척도이므로 구간 적중률은 실제 r도 r*로 바꿔 비교한다 (cv_metrics)
            lo, hi = np.quantile(r_rep, [0.05, 0.95], axis=0)
        add(name, mu.mean(axis=0), lo, hi, diag)
        status = "OK" if is_converged(diag) else "수렴 실패"
        print(f"    {name:<20} R-hat {diag['rhat_max']:.3f}, ESS {diag['ess_min']:.0f}, "
              f"div {diag['divergences']} ({status})")

    out = pd.concat(preds, ignore_index=True)
    out.insert(0, "year", holdout)
    out["r_star"] = shrink_ratio(out["r"], n_rows)
    return out, {"year": holdout, "total_flooded": total_flooded, "unpredictable": unpredictable}


def cv_metrics(preds, unpredictable):
    """모델 x 연도 x 지표 (long format)."""
    rows = []
    for (model, year), g in preds.groupby(["model", "year"]):
        m = score(g["r"].values, g["pred"].values)
        if g["lo"].notna().all():
            m["coverage90"] = float(((g["r_star"] >= g["lo"]) & (g["r_star"] <= g["hi"])).mean())
        m["converged"] = float(g["converged"].all())
        rows += [{"model": model, "year": year, "metric": k, "value": v} for k, v in m.items()]
    for u in unpredictable:
        share = u["unpredictable"] / u["total_flooded"] if u["total_flooded"] else np.nan
        rows.append({"model": "all", "year": u["year"], "metric": "unpredictable_share", "value": share})
    return pd.DataFrame(rows)


def cv_summary(preds):
    """모든 클러스터-연도 행에 대한 평균 지표."""
    rows = []
    for model, g in preds.groupby("model"):
        m = {"model": model, **score(g["r"].values, g["pred"].values)}
        if g["lo"].notna().all():
            m["coverage90"] = ((g["r_star"] >= g["lo"]) & (g["r_star"] <= g["hi"])).mean()
        m["folds_not_converged"] = int((~g.groupby("year")["converged"].all()).sum())
        rows.append(m)
    return pd.DataFrame(rows).sort_values("brier")


def reliability(preds):
    preds = preds.assign(bin=pd.cut(preds["pred"], RELIABILITY_BINS, include_lowest=True))
    return (preds.groupby(["model", "bin"], observed=True)
            .agg(n=("r", "size"), mean_pred=("pred", "mean"), mean_actual=("r", "mean"))
            .reset_index())


def run_cv(grid, pcs):
    clustering = load_clustering_module()
    history = history_from_combo(grid["combo"])
    xy = grid[["x", "y"]].values

    all_preds, unpredictable = [], []
    for i, holdout in enumerate(TRAIN_YEARS, 1):
        t = time.time()
        print(f"  [{i}/{len(TRAIN_YEARS)}] {holdout}년 제외")
        preds, unp = cv_fold(holdout, history, xy, pcs, clustering)
        all_preds.append(preds)
        unpredictable.append(unp)
        print(f"    클러스터 {preds['cluster'].nunique()}개, 예측 불가 침수 격자 "
              f"{unp['unpredictable']:,} / {unp['total_flooded']:,}, {time.time() - t:.0f}초")
    preds = pd.concat(all_preds, ignore_index=True)
    return preds, cv_metrics(preds, unpredictable)


# ---------------------------------------------------------------- 최종 모델 점검

def prior_predictive_check(model):
    """데이터를 보기 전 예측 비율 μ가 0과 1에만 몰리지 않는지 확인."""
    with model:
        prior = pm.sample_prior_predictive(draws=1000, random_seed=SEED)
    mu = prior.prior["mu"].values.ravel()
    print(f"  사전 μ: 중앙값 {np.median(mu):.3f}, μ<0.01 {np.mean(mu < 0.01):.1%}, "
          f"μ>0.99 {np.mean(mu > 0.99):.1%}, 0.01~0.99 {np.mean((mu >= 0.01) & (mu <= 0.99)):.1%}")


def posterior_predictive_check(idata, table, obs):
    """모델로 다시 생성한 비율 분포를 실제 분포와 비교."""
    rng = np.random.default_rng(SEED)
    mu = posterior_draws(idata, "mu")
    if obs == "beta":
        phi = posterior_draws(idata, "phi")[:, None]
        rep = rng.beta(mu * phi, (1 - mu) * phi)
        actual = shrink_ratio(table["r"].values, len(table))
    else:
        rep = rng.random(mu.shape) < mu
        actual = (table["r"].values >= 0.5).astype(float)
    stats = {"평균": np.mean, "0.05 미만 비율": lambda a, **kw: np.mean(a < 0.05, **kw),
             "0.5 이상 비율": lambda a, **kw: np.mean(a >= 0.5, **kw)}
    for name, f in stats.items():
        sim = f(rep, axis=1)
        lo, hi = np.quantile(sim, [0.05, 0.95])
        print(f"  {name:<12} 실제 {f(actual):.3f} | 재생성 {sim.mean():.3f} (90%: {lo:.3f}~{hi:.3f})")


def coefficient_table(idata, table, n_pcs):
    """클러스터별 α, β의 평균과 90% 구간. n_major_years: r >= 0.5인 연도 수 (수축 확인용)."""
    pcs = PC_COLS[:n_pcs]

    def summarize(values, prefix):
        return {f"{prefix}_mean": values.mean(), f"{prefix}_q05": np.quantile(values, 0.05),
                f"{prefix}_q95": np.quantile(values, 0.95)}

    alpha, beta = posterior_draws(idata, "alpha"), posterior_draws(idata, "beta")
    major = table.groupby("cluster")["r"].apply(lambda r: int((r >= 0.5).sum()))
    n_cells = table.groupby("cluster")["n_cells"].first()
    rows = []
    for c in range(alpha.shape[1]):
        row = {"cluster": c, "n_cells": n_cells[c], "n_major_years": major[c], **summarize(alpha[:, c], "alpha")}
        for k, pc in enumerate(pcs):
            row.update(summarize(beta[:, c, k], f"beta_{pc}"))
        rows.append(row)
    common = {"cluster": "common", **summarize(posterior_draws(idata, "alpha_bar"), "alpha")}
    beta_bar = posterior_draws(idata, "beta_bar")
    for k, pc in enumerate(pcs):
        common.update(summarize(beta_bar[:, k], f"beta_{pc}"))
    rows.append(common)
    return pd.DataFrame(rows)


def annual_risk(idata, z_all, n_pcs):
    """P_c = 과거 각 연도의 강우가 다시 온다고 보고 평균한 연간 침수 확률. 사후 표본별로 계산."""
    mu = predict_mu(idata, z_all[Z_COLS[:n_pcs]].values)
    p = mu.mean(axis=2)
    p_living = 1 - (1 - p) ** LIVING_YEARS
    return pd.DataFrame({
        "cluster": np.arange(p.shape[1]),
        "p_annual": p.mean(axis=0),
        "p_annual_q05": np.quantile(p, 0.05, axis=0),
        "p_annual_q95": np.quantile(p, 0.95, axis=0),
        f"p_{LIVING_YEARS}yr": p_living.mean(axis=0),
        f"p_{LIVING_YEARS}yr_q05": np.quantile(p_living, 0.05, axis=0),
        f"p_{LIVING_YEARS}yr_q95": np.quantile(p_living, 0.95, axis=0),
    })


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.width", 250)

    print("[1/6] 입력 로드 + 학습 테이블")
    grid, pcs = load_inputs()
    history = history_from_combo(grid["combo"])
    table = build_table(history, grid["cluster"], TRAIN_YEARS)
    z_all = standardize(pcs, TRAIN_YEARS)
    table = table.join(z_all, on="year")
    print(f"  격자 {len(grid):,}개, 클러스터 {grid['cluster'].nunique()}개, 학습 연도 {len(TRAIN_YEARS)}개 "
          f"(제외: {NO_DATA_YEARS}) -> {len(table)}행")
    print(f"  r = 0인 행 {np.mean(table['r'] == 0):.1%}, r >= 0.5인 행 {np.mean(table['r'] >= 0.5):.1%}")
    table.to_csv(os.path.join(OUT_DIR, "training_table.csv"), index=False, encoding="utf-8-sig")

    print("\n[2/6] 연도 단위 교차검증 (fold마다 격자 클러스터링부터 다시)")
    preds, metrics = run_cv(grid, pcs)
    metrics.to_csv(os.path.join(OUT_DIR, "cv_results.csv"), index=False, encoding="utf-8-sig")
    preds.to_csv(os.path.join(OUT_DIR, "cv_predictions.csv"), index=False, encoding="utf-8-sig")
    summary = cv_summary(preds)
    print("\n  모델별 평균 (전체 클러스터-연도):")
    print(summary.round(4).to_string(index=False))
    rel = reliability(preds)
    rel.to_csv(os.path.join(OUT_DIR, "cv_reliability.csv"), index=False, encoding="utf-8-sig")
    print("\n  신뢰도 곡선 (예측 구간별 평균 예측 vs 실제):")
    print(rel.round(3).to_string(index=False))
    unp = metrics[metrics["metric"] == "unpredictable_share"]
    print(f"\n  예측 불가 침수 격자 비율 (연도 평균): {unp['value'].mean():.1%}")

    hier = summary[summary["model"].str.startswith("hier_")].iloc[0]
    obs, n_pcs = hier["model"].split("_")[1], int(hier["model"][-1])
    clim = summary.set_index("model").loc["climatology"]
    print(f"\n  선택: {hier['model']} (Brier {hier['brier']:.4f}, 로그 손실 {hier['log_loss']:.4f})")
    print(f"  기후 기준선 대비: Brier {clim['brier']:.4f}, 로그 손실 {clim['log_loss']:.4f} -> "
          f"{'강우 입력이 도움 됨' if hier['brier'] < clim['brier'] and hier['log_loss'] < clim['log_loss'] else '기후 기준선보다 낫지 않음'}")

    print(f"\n[3/6] 최종 모델 사전 예측 점검 ({obs}, PC {n_pcs}개)")
    prior_predictive_check(build_model(table, n_pcs, obs))

    print("\n[4/6] 최종 모델 학습 + 수렴 점검")
    t = time.time()
    model, idata, diag = fit(table, n_pcs, obs)
    print(f"  {time.time() - t:.0f}초, target_accept {diag['target_accept']}, R-hat 최대 {diag['rhat_max']:.4f}, "
          f"ESS 최소 {diag['ess_min']:.0f}, divergence {diag['divergences']}개")
    idata.to_netcdf(os.path.join(OUT_DIR, "posterior.nc"))
    if not is_converged(diag):
        raise SystemExit("  수렴 기준 미달: 결과 파일을 쓰지 않습니다. 사전분포나 매개변수화를 점검하세요.")

    print("\n[5/6] 사후 예측 점검 + 계수")
    posterior_predictive_check(idata, table, obs)
    coefs = coefficient_table(idata, table, n_pcs)
    coefs.to_csv(os.path.join(OUT_DIR, "cluster_coefficients.csv"), index=False, encoding="utf-8-sig")
    print(coefs.round(2).to_string(index=False))
    tau = {"tau_alpha": posterior_draws(idata, "tau_alpha").mean()}
    tau.update({f"tau_{pc}": v for pc, v in zip(PC_COLS, posterior_draws(idata, "tau_beta").mean(axis=0))})
    print("  τ 사후 평균: " + ", ".join(f"{k} {v:.2f}" for k, v in tau.items()))

    print(f"\n[6/6] 연간 침수 확률 (강우 시나리오 {len(z_all)}개 연도: {z_all.index.min()}~{z_all.index.max()})")
    risk = annual_risk(idata, z_all, n_pcs)
    risk.insert(1, "n_cells", grid["cluster"].value_counts().sort_index().values)
    risk.to_csv(os.path.join(OUT_DIR, "cluster_annual_risk.csv"), index=False, encoding="utf-8-sig")
    print(risk.round(3).to_string(index=False))
    grid_risk = grid[["ix", "iy", "x", "y", "cluster"]].reset_index().merge(
        risk.drop(columns="n_cells"), on="cluster", how="left")
    grid_risk.to_csv(os.path.join(OUT_DIR, "grid_annual_risk.csv"), index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
