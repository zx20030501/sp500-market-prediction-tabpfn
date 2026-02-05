import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# TabPFN 数据探索分析（EDA）

该 Notebook 针对 Hull Tactical Market Prediction 数据集，对训练/测试集做系统的统计分析、相关性、缺失检测、PCA 以及关键特征筛选，并辅以可视化结果。"""))

cells.append(nbf.v4.new_markdown_cell("""## 目录
1. 环境与数据载入
2. 数据规模 & 预览
3. 描述性统计
4. 缺失值与数据质量
5. 相关性与可视化
6. PCA 及解释方差
7. 关键特征汇总"""))

setup_code = dedent('''
    from __future__ import annotations
    import json
    from pathlib import Path
    from typing import List

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 120)

    TARGET_COL = "forward_returns"
    DATA_ROOT = Path("../input/hull-tactical-market-prediction").resolve()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")
    train_path = DATA_ROOT / "train.csv"
    test_path = DATA_ROOT / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 为了在可视化和 PCA 时控制成本，最多使用 200k 行样本
    SAMPLE_ROWS = 200_000
    if len(train_df) > SAMPLE_ROWS:
        eda_df = train_df.sample(SAMPLE_ROWS, random_state=42)
        print(f"[INFO] Using sampled {len(eda_df)} rows for heavy EDA (from {len(train_df)} total).")
    else:
        eda_df = train_df.copy()
        print(f"[INFO] Using full training set ({len(eda_df)} rows) for EDA.")

    print("Training set shape:", train_df.shape)
    print("Test set shape:", test_df.shape)
    train_df.head()
''')

cells.append(nbf.v4.new_code_cell(setup_code))

cells.append(nbf.v4.new_markdown_cell("""### 1. 描述性统计
- 数值特征的均值、标准差、分位数
- 分类特征的频数分布"""))

desc_code = dedent('''
    numeric_stats = train_df.describe().T
    display(numeric_stats.head(20))

    categorical_stats = train_df.describe(include=["object", "category"]).T
    if not categorical_stats.empty:
        display(categorical_stats.head(20))
    else:
        print("[INFO] No categorical columns detected in train set.")
''')

cells.append(nbf.v4.new_code_cell(desc_code))

cells.append(nbf.v4.new_markdown_cell("""### 2. 缺失值与数据质量
重点观察缺失占比 Top-N、常量列、重复行等。"""))

missing_code = dedent('''
    missing_ratio = train_df.isna().mean().sort_values(ascending=False)
    display(missing_ratio.head(20).to_frame("missing_ratio"))

    top_missing = missing_ratio.head(30)
    plt.figure(figsize=(14, 6))
    sns.barplot(x=top_missing.index, y=top_missing.values, palette="Reds")
    plt.xticks(rotation=90)
    plt.title("Top Missing Ratio Columns")
    plt.ylabel("Missing Ratio")
    plt.tight_layout()
    plt.show()

    duplicate_rows = train_df.duplicated().sum()
    constant_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) <= 1]
    print(f"[INFO] Duplicate rows: {duplicate_rows}")
    print(f"[INFO] Constant columns: {len(constant_cols)} -> {constant_cols[:20]}")
''')

cells.append(nbf.v4.new_code_cell(missing_code))

cells.append(nbf.v4.new_markdown_cell("""### 3. 相关性分析
- 计算数值特征之间的相关系数矩阵
- 针对目标列挑选相关性最高的特征，并对前 25 个特征绘制热力图"""))

corr_code = dedent('''
    numeric_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL not in numeric_cols:
        raise ValueError(f"Target column {TARGET_COL} not numeric or missing.")

    corr_matrix = eda_df[numeric_cols].corr()
    target_corr = corr_matrix[TARGET_COL].dropna().sort_values(key=lambda s: -s.abs())
    display(target_corr.head(30).to_frame("corr_with_target"))

    top_feats = target_corr.head(25).index.tolist()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix.loc[top_feats, top_feats], cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap (Top 25 vs Target)")
    plt.tight_layout()
    plt.show()
''')

cells.append(nbf.v4.new_code_cell(corr_code))

cells.append(nbf.v4.new_markdown_cell("""### 4. PCA（主成分分析）
- 标准化数值特征（排除目标列）
- 查看解释方差和累计贡献
- 展示前几个主成分的高载荷特征"""))

pca_code = dedent('''
    feature_cols = [c for c in numeric_cols if c != TARGET_COL]
    pca_df = eda_df[feature_cols].fillna(0.0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pca_df)
    n_components = min(20, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(scaled)

    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_components)],
        "variance_ratio": pca.explained_variance_ratio_,
    })
    explained["cumulative"] = explained["variance_ratio"].cumsum()
    display(explained.head(10))

    plt.figure(figsize=(12, 5))
    sns.barplot(x="component", y="variance_ratio", data=explained, color="steelblue")
    plt.plot(explained.index, explained["cumulative"], marker="o", color="orange", label="Cumulative")
    plt.xticks(rotation=45)
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=explained["component"])
    for comp in loadings.columns[:5]:
        top_loadings = loadings[comp].abs().sort_values(ascending=False).head(10)
        print(f"\n[INFO] {comp} top contributors:")
        display(top_loadings.to_frame(comp))
''')

cells.append(nbf.v4.new_code_cell(pca_code))

cells.append(nbf.v4.new_markdown_cell("""### 5. 关键特征筛选
结合目标相关性和 PCA 前 5 个主成分载荷的归一化分数，输出综合排名。"""))

key_feat_code = dedent('''
    corr_score = target_corr.reindex(feature_cols).abs().fillna(0.0)
    corr_norm = corr_score / (corr_score.max() or 1.0)

    leading = loadings.iloc[:, :min(5, loadings.shape[1])].abs().sum(axis=1)
    loading_norm = leading / (leading.max() or 1.0)

    combined = 0.6 * corr_norm + 0.4 * loading_norm
    ranking = (
        pd.DataFrame({
            "feature": feature_cols,
            "corr_score": corr_norm.values,
            "pca_score": loading_norm.values,
            "combined_score": combined.values,
        })
        .sort_values("combined_score", ascending=False)
        .head(40)
    )
    display(ranking.head(20))
''')

cells.append(nbf.v4.new_code_cell(key_feat_code))

cells.append(nbf.v4.new_markdown_cell("""### 6. 小结
- 上述分析提供了缺失分布、变量相关性、PCA 解释方差以及关键特征排名。
- 可根据关键特征列表进一步优化 TabPFN 模型或用于特征选择。"""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

nbf.write(nb, "eda_tabpfn.ipynb")
