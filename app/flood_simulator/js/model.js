// 시나리오 침수 확률. P_{s,g} = σ(α_{s,c} + β_{s,c} · z + log q_g), z = (PC − pc_mean) / pc_std

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// 사후 표본 × 클러스터의 η (S × C, 행 우선 평탄화)
export function scenarioEta(model, pc) {
  const K = model.pcs.length;
  const z = model.pc_mean.map((m, k) => (pc[k] - m) / model.pc_std[k]);
  const S = model.alpha.length, C = model.alpha[0].length;
  const eta = new Float64Array(S * C);
  for (let s = 0; s < S; s++) {
    for (let c = 0; c < C; c++) {
      let v = model.alpha[s][c];
      for (let k = 0; k < K; k++) v += model.beta[s][c][k] * z[k];
      eta[s * C + c] = v;
    }
  }
  return { eta, S, C };
}

// 격자별 표본 평균 확률
export function gridMeans({ eta, S, C }, cluster, logQ) {
  const n = cluster.length;
  const mean = new Float32Array(n);
  for (let g = 0; g < n; g++) {
    const c = cluster[g], lq = logQ[g];
    let sum = 0;
    for (let s = 0; s < S; s++) sum += sigmoid(eta[s * C + c] + lq);
    mean[g] = sum / S;
  }
  return mean;
}

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// 한 격자의 평균과 5–95% 분위수 (numpy quantile 기본값과 같은 선형 보간)
export function gridSummary({ eta, S, C }, c, lq) {
  const p = new Float64Array(S);
  for (let s = 0; s < S; s++) p[s] = sigmoid(eta[s * C + c] + lq);
  const mean = p.reduce((a, b) => a + b, 0) / S;
  p.sort();
  return { mean, q05: quantile(p, 0.05), q95: quantile(p, 0.95) };
}

export function compute(model, grids, pc) {
  const eta = scenarioEta(model, pc);
  return { pc: [...pc], eta, mean: gridMeans(eta, grids.cluster, grids.log_q) };
}
