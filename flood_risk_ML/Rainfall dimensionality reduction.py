"""Reduces the regional rainfall features per year to principal components with PCA.

Saves PC1-PC4 per year to 'rainfall data final ver.csv' and writes the standardisation and component vectors, with
instructions for transforming new data the same way, to '강우량 차원 축소 방법.txt'.
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "rainfall clustered data.csv")

# load the data: one row per year, one column per regional rainfall feature (max1h, max24h)
df = pd.read_csv(CSV_PATH)
years = df["year"].values
X = df.drop(columns=["year"]).values

# max1h and max24h have very different scales, so standardise before PCA
X_scaled = StandardScaler().fit_transform(X)

# the largest possible number of components is min(samples, features)
max_dim = min(10, X_scaled.shape[0], X_scaled.shape[1])

pca = PCA(n_components=max_dim)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"데이터 크기: {X.shape[0]}개 연도 x {X.shape[1]}개 특징\n")
print(f"{'차원':>4} | {'설명 분산':>10} | {'누적 설명 분산':>14}")
print("-" * 36)
for k in range(1, max_dim + 1):
    print(f"{k:>4} | {explained[k - 1]:>10.4f} | {cumulative[k - 1]:>14.4f}")

# principal component coordinates of every year (all dimensions)
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(max_dim)])
pca_df.insert(0, "year", years)
print("\n연도별 주성분 좌표:")
print(pca_df.round(3).to_string(index=False))

# save the 4-dimensional result
N_KEEP = 4
OUT_CSV = os.path.join(BASE_DIR, "rainfall data final ver.csv")
OUT_TXT = os.path.join(BASE_DIR, "강우량 차원 축소 방법.txt")

pca_df[["year"] + [f"PC{i + 1}" for i in range(N_KEEP)]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n저장: {OUT_CSV}")

# save the instructions and parameters for applying the same transformation to new data
feature_names = list(df.drop(columns=["year"]).columns)
scaler = StandardScaler().fit(X)


def fmt_vec(v):
    """Formats a vector as a bracketed list with 10 decimals."""
    return "[" + ", ".join(f"{x:.10f}" for x in v) + "]"


lines = [
    "강우량 데이터 PCA 차원 축소 방법",
    "=" * 40,
    "",
    "1. 개요",
    f"- 원본: 'rainfall clustered data.csv' ({X.shape[0]}개 연도 x {X.shape[1]}개 특징)",
    f"- 결과: 'rainfall data final ver.csv' (year, PC1~PC{N_KEEP})",
    f"- {N_KEEP}차원 누적 설명 분산: {cumulative[N_KEEP - 1]:.4f}",
    "- 처리 순서: 표준화(StandardScaler) -> PCA 투영",
    "",
    "2. 새 데이터 변환 방법",
    "새 연도 데이터가 들어오면 PCA를 다시 학습하지 말고, 아래의 평균/표준편차/주성분 벡터를",
    "그대로 사용해 변환한다. 그래야 기존 연도와 같은 좌표계에 놓인다.",
    "",
    "  (1) 특징 순서를 아래 '특징 순서'와 똑같이 맞춘다.",
    "  (2) 표준화:   z = (x - mean) / std",
    "  (3) 투영:     PCk = sum_j ( z_j * component_k[j] ),  k = 1..4",
    "",
    "주의: 표준편차는 모집단 표준편차(ddof=0, sklearn StandardScaler와 동일)이다.",
    "",
    "3. 특징 순서",
    "  " + ", ".join(feature_names),
    "",
    "4. 표준화 파라미터",
    "mean = " + fmt_vec(scaler.mean_),
    "std  = " + fmt_vec(scaler.scale_),
    "",
    "5. 주성분 벡터 (각 행이 하나의 주성분, 열 순서는 특징 순서와 동일)",
]
for k in range(N_KEEP):
    lines.append(f"PC{k + 1} = " + fmt_vec(pca.components_[k]))
lines += [
    "",
    "6. 파이썬 예시 코드",
    "import numpy as np",
    "import pandas as pd",
    "",
    "features = " + repr(feature_names),
    "mean = np.array(" + fmt_vec(scaler.mean_) + ")",
    "std = np.array(" + fmt_vec(scaler.scale_) + ")",
    "components = np.array([",
]
for k in range(N_KEEP):
    lines.append("    " + fmt_vec(pca.components_[k]) + ",")
lines += [
    "])",
    "",
    "new = pd.read_csv('새 강우량 데이터.csv')  # year + 위 특징 컬럼",
    "z = (new[features].values - mean) / std",
    "pcs = z @ components.T",
    "out = pd.DataFrame(pcs, columns=[f'PC{i + 1}' for i in range(components.shape[0])])",
    "out.insert(0, 'year', new['year'].values)",
    "",
    "7. 주의사항",
    "- 새 데이터의 컬럼은 같은 권역 정의(R1~R5)와 같은 단위(mm)로 만들어야 한다.",
    "- 데이터가 많이 쌓여 분포가 달라졌다고 판단되면 PCA를 다시 학습할 수 있지만,",
    "  그 경우 주성분 축이 바뀌므로 기존 결과도 전부 다시 계산해야 한다.",
    "- 주성분의 부호(+/-)는 재학습 시 뒤집힐 수 있으므로 반드시 위 벡터를 고정해서 사용한다.",
]

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"저장: {OUT_TXT}")

# check: values computed by hand from the saved parameters must equal the PCA result
manual = ((X - scaler.mean_) / scaler.scale_) @ pca.components_[:N_KEEP].T
assert np.allclose(manual, X_pca[:, :N_KEEP]), "수동 변환 결과가 PCA 결과와 다릅니다"
print("검증 완료: 파라미터로 직접 변환한 값이 PCA 결과와 일치")
