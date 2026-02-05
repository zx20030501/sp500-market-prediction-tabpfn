# run_all 一键拉取与运行实验报告（本地测试）

## 测试目标
验证仓库一键拉取、数据自动下载、训练/推理流程是否可用，并生成 `submission.csv`。

## 测试环境
- 系统：Windows
- GPU：NVIDIA GeForce RTX 3090
- 驱动 / CUDA：Driver 581.57 / CUDA 13.0（nvidia-smi）
- Python：3.11（CUDA 版 Torch 可用）

## 测试目录
- 测试目录：`hull-tabpfn-run-all-test`
- 仓库目录：`hull-tabpfn-run-all-test/repo`
- 数据目录：`repo/data/hull-tactical-market-prediction`
- 输出目录：`repo/outputs`

## 执行步骤与耗时概览
1. **克隆仓库** → 成功
2. **虚拟环境安装依赖（Python 3.12）** → 失败
   - 报错：`matplotlib` 需要 MSVC Build Tools（无编译环境）
3. **改用 Python 3.11 虚拟环境** → 依赖安装成功
4. **安装 Kaggle CLI 并下载数据** → 成功
5. **CPU 运行 pipeline（venv 内 Torch=CPU）** → 15 分钟内无 `submission.csv`
6. **尝试在 venv 内安装 CUDA Torch** → 失败
   - 大文件下载超时
   - Hash 校验不一致（可能与网络或下载中断有关）
7. **改用系统 Python 3.11 + CUDA Torch（2.6.0+cu124）** → 成功
   - GPU 使用率达 100%，显存约 24GB
   - 最终生成 `submission.csv`

> 备注：最终 GPU 完整流程约 **40–45 分钟**（从模型训练到提交文件生成）。

## 关键卡点与解决方案
1. **Python 3.12 安装依赖失败**
   - 原因：`matplotlib==3.7.2` 在 3.12 下可能需要编译环境
   - 解决：切换到 **Python 3.11**

2. **Torch 为 CPU 版，导致运行极慢**
   - 原因：TabPFN 依赖 torch，默认从 PyPI 拉取 CPU 版本
   - 解决：使用系统 Python 3.11 + CUDA 版 Torch

3. **CUDA Torch 下载不稳定**
   - 原因：包体积大（2GB+），网络中断导致 hash mismatch
   - 解决：避免在 venv 中安装 CUDA Torch，直接使用已有 GPU Python 环境

## 最终产出
- `outputs/submission.csv` ✅
- `outputs/submission.parquet` ✅
- `outputs/artifacts/tabpfn_model/` ✅

校验结果：
- `submission.csv` 列名包含 `row_id/prediction`
- 行数与 `test.csv` 一致（10 行）

## 结果得分
- **未提交 Kaggle**，暂无线上评分。

## 本地评分（校准窗口）
本地评分基于训练集末尾窗口的校准评估（Adjusted Sharpe），命令：
```powershell
$pythonExe = "C:\Users\xuanz\AppData\Local\Programs\Python\Python311\python.exe"
& $pythonExe scripts\score_local.py `
  --data-root d:\OneDrive\Project\kaggle-hull\data\hull-tactical-market-prediction `
  --out-dir outputs_local_score `
  --max-training-rows 5000
```

输出（`outputs_local_score/artifacts/tabpfn_model/metadata.json`）：
- `raw_sharpe`: **1.732858297614543**
- `adjusted_sharpe`: **2.054413659354495**
- `scale`: **1.195813715004886**
- `shift`: **-0.0009645427413516152**
- `window`: **180**

## 改进建议
1. **增加 GPU 检测**：运行前检查 `torch.cuda.is_available()`，无 GPU 则直接退出或提示。
2. **支持指定解释器**：在 `run_all.ps1` 增加 `-PythonExe` 参数，强制使用 GPU 解释器。
3. **提供 GPU 安装指南**：在文档里明确 CUDA Torch 的安装方法与常见问题。
4. **快速验证模式**：提供 `-Fast` 参数，仅跑最小基线或更小采样，用于功能验证。
5. **日志落盘**：将 `run_all.ps1` 输出记录到 `outputs/logs/` 便于排查。

## 结论
一键流程在 GPU 环境下 **可成功生成提交文件**，但 CPU 运行耗时过长。建议默认要求 GPU 或提供明确的 GPU 安装与检测机制。

## 复测（修复后）
为消除 EDA 报错，修复了两处问题：
- 分类列为空时的 `describe()` 处理
- PCA 载荷维度不匹配问题

使用 `run_all.ps1 -PythonExe <GPU Python>` 在本机重新跑一遍（输出到 `outputs_fullflow3`），结果：
- EDA + pipeline 全流程完成 ✅
- 生成 `submission.csv` ✅
- 仅出现 `seaborn` 的 FutureWarning（不影响结果）
