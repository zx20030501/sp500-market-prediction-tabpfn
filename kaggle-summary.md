# Kaggle Summary - Hull TabPFN

## 比赛目标
基于多因子金融时间序列数据，预测 `forward_returns`，评估指标为调整后的 Sharpe（调整后风险收益比）。

## 方法概览
1. **数据清洗**
- 使用 `date_id` 去除训练集中与测试集时间重叠的行，降低信息泄露风险。

2. **特征工程**
- 为 `forward_returns`、`market_forward_excess_returns`、`risk_free_rate` 构造 `lagged_` 特征。
- 自动剔除包含 `forward / future / lead / next` 关键词的前瞻性字段，避免“偷看未来”。
- 缺失值统一 `fillna(0)`，保持输入维度稳定。

3. **模型**
- 使用 TabPFNRegressor 作为主模型。
- 支持本地 ckpt（离线）或在线下载（HuggingFace）。

4. **后处理与校准**
- 在训练集末尾窗口内，通过 `Powell` 优化缩放/偏移参数，使预测位置在 `min/max position` 约束内的 **Adjusted Sharpe** 最大化。
- 保存校准参数与模型元信息。

## 关键代码文件
- `src/run_tabpfn_min.py`：最小可运行基线
- `src/run_tabpfn_pipeline.py`：完整流程（含校准与模型缓存）
- `src/eda_tabpfn.py`：输出统计、相关性、PCA 与关键特征排名
- `src/eda_make_plots.py`：生成 EDA 图片
- `notebooks/kaggle-submit-tabtfn.ipynb`：提交用 Notebook
- `notebooks/hull-tabtfn-only-success.ipynb`：提交成功版本

## EDA 产出
- `assets/eda/missing_top30.png`
- `assets/eda/corr_top25_heatmap.png`
- `assets/eda/pca_explained_variance.png`

## 复现流程
1. 准备数据：`data/hull-tactical-market-prediction/`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行：
```bash
python src/run_tabpfn_pipeline.py --data-root data/hull-tactical-market-prediction --out-dir outputs
```

## 经验与结论
- TabPFN 在高维、非线性结构下具有良好适配性，适合该类多因子时间序列任务。
- 对前瞻字段的自动剔除和时间切割是最重要的“防泄露”步骤。
- 通过校准调整预测分布，可直接优化比赛评估指标（Adjusted Sharpe）。

## 后续可改进方向
- 对 `lagged_` 特征引入更长窗口或滚动统计特征。
- 加入时间切片验证（rolling CV）评估稳定性。
- 与树模型/线性模型做融合（stack/ensemble）提升稳健性。
