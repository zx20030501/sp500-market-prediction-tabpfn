# sp500-market-prediction-tabpfn (Hull TabPFN)

基于多因子金融时间序列数据，预测 `forward_returns`，评估指标为调整后的 Sharpe（调整后风险收益比）。

本仓库是对 `hull-tabpfn` 工作目录的整理版本，保留了核心代码、Notebook 与可视化产物，便于别人一键拉取并复现实验。

**比赛**：Hull Tactical Market Prediction（Kaggle）

## 目录
- `src/`：可执行脚本（训练、推理、EDA）
- `notebooks/`：核心 Notebook（EDA、提交、成功版本）
- `assets/eda/`：EDA 可视化图片
- `kaggle-summary.md`：比赛总结与工作展示

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 获取数据（不随仓库提供）
- 推荐：使用 Kaggle API 下载到 `data/hull-tactical-market-prediction/`
- 或者手动下载后放到同目录

3. 运行最小基线
```bash
python src/run_tabpfn_min.py --data-root data/hull-tactical-market-prediction --out-dir outputs
```

4. 运行完整流程（含校准）
```bash
python src/run_tabpfn_pipeline.py --data-root data/hull-tactical-market-prediction --out-dir outputs
```

5. 运行 EDA
```bash
python src/eda_tabpfn.py --data-root data/hull-tactical-market-prediction --output-dir outputs/eda
python src/eda_make_plots.py
```

## 一键运行
Windows / PowerShell 可使用：
```powershell
.\run_all.ps1 -DataRoot data\hull-tactical-market-prediction -OutDir outputs -InstallDeps
```
可选参数：
- `-SkipEda`：跳过 EDA
- `-CkptPath`：指定本地 TabPFN 权重
- `-Offline`：强制离线模式
- `-MaxTrainingRows`：训练下采样行数（加速）

## 重要说明
- 数据集与大文件（模型、缓存、离线 wheel）已加入 `.gitignore`。
- TabPFN 默认会从 HuggingFace 拉取权重，离线环境可传入：
```bash
python src/run_tabpfn_pipeline.py --ckpt-path /path/to/tabpfn-v2-regressor.ckpt --offline
```
- 如需修改 Notebook 训练行数限制，可使用：
```bash
python scripts/apply_runtime_cap.py --notebook notebooks/kaggle-submit-tabtfn.ipynb --max-training-rows 20000
```

## 复现路径约定
脚本会优先读取以下路径（按顺序）：
1. `--data-root` 参数
2. 环境变量 `HULL_DATA_ROOT`
3. `/kaggle/input/hull-tactical-market-prediction`
4. `data/hull-tactical-market-prediction`
5. `input/hull-tactical-market-prediction`

## 推荐阅读
- `kaggle-summary.md`
- `notebooks/hull-tabtfn-only-success.ipynb`
