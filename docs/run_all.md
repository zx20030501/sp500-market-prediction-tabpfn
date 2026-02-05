# run_all.ps1 一键运行说明

本说明覆盖环境配置、目录结构、数据拉取、运行示例与全部参数含义。

## 环境配置（Windows / PowerShell）
1. 安装 Python（建议 3.10/3.11），确保 `python` 命令可用。
2. 进入项目根目录：`hull-tabpfn/`
3. 可选：允许执行脚本
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
4. 安装依赖
```powershell
.\run_all.ps1 -InstallDeps
```

## GPU 依赖与注意事项（重要）
- `run_all.ps1` 默认使用 **PATH 中的 `python`**，请确保该解释器已安装 **CUDA 版 Torch**，否则将退回 CPU 运行（速度非常慢）。  
- 推荐 Python 3.11，并使用官方 CUDA 版 Torch，例如（根据你的驱动/CUDA 版本选择合适的版本）：  
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
- 运行前建议检查：
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```
- 若提示 `False`，请先安装 CUDA 版 Torch，再执行 `run_all.ps1`。

## 文件夹路径设置
推荐结构：
```
hull-tabpfn/
  data/
    hull-tactical-market-prediction/
      train.csv
      test.csv
  outputs/
```

- `-DataRoot` 指向包含 `train.csv/test.csv` 的目录。
- `-OutDir` 为输出目录（相对根目录或绝对路径）。

## 数据拉取方法（Kaggle CLI）
1. 安装 Kaggle CLI
```powershell
pip install kaggle
```
2. 准备 API 凭证
- 将 `kaggle.json` 放到：`C:\Users\<你的用户名>\.kaggle\kaggle.json`
3. 自动下载
```powershell
.\run_all.ps1 -DownloadData
```

如需指定路径：
```powershell
.\run_all.ps1 -DownloadData -DataRoot data\hull-tactical-market-prediction
```

## 一键运行示例
完整流程：
```powershell
.\run_all.ps1 -DataRoot data\hull-tactical-market-prediction -OutDir outputs -InstallDeps
```

只跑模型，跳过 EDA：
```powershell
.\run_all.ps1 -DataRoot data\hull-tactical-market-prediction -OutDir outputs -SkipEda
```

离线模式（本地 ckpt）：
```powershell
.\run_all.ps1 -DataRoot data\hull-tactical-market-prediction -OutDir outputs -CkptPath D:\models\tabpfn-v2-regressor.ckpt -Offline
```

## 参数说明
- `-DataRoot`：数据根目录（含 `train.csv/test.csv`）
- `-OutDir`：输出目录
- `-PythonExe`：指定 Python 解释器路径（建议指向 GPU 可用的 Python）
- `-InstallDeps`：安装依赖
- `-DownloadData`：自动下载 Kaggle 数据并解压
- `-KaggleCompetition`：比赛标识，默认 `hull-tactical-market-prediction`
- `-SkipEda`：跳过 EDA
- `-CkptPath`：本地 TabPFN 权重路径
- `-Offline`：强制离线
- `-MaxTrainingRows`：训练下采样行数
- `-SkipValidation`：跳过提交文件校验

## 输出内容
- `submission.csv`
- `submission.parquet`
- `artifacts/tabpfn_model/`
- `eda/`（未跳过 EDA 时）

## 提交文件校验
默认会检查：
- `submission.csv` 是否存在
- 列名是否包含 `row_id/id` 与 `prediction`
- 与 `test.csv` 行数是否一致
- 是否包含 NaN

如需跳过：使用 `-SkipValidation`。

## 本地评分示例
说明：本地评分基于 **训练集末尾窗口** 的校准评估（Adjusted Sharpe），不是 Kaggle 线上分数。

```powershell
$pythonExe = "C:\Users\xuanz\AppData\Local\Programs\Python\Python311\python.exe"
& $pythonExe scripts\score_local.py `
  --data-root d:\OneDrive\Project\kaggle-hull\data\hull-tactical-market-prediction `
  --out-dir outputs_local_score `
  --max-training-rows 5000

type outputs_local_score\artifacts\tabpfn_model\metadata.json
```

关注字段：
- `raw_sharpe`
- `adjusted_sharpe`

## 常见问题
- Kaggle CLI 未找到：请确认 `kaggle` 命令可用且已安装。
- 离线环境：必须提供 `-CkptPath` 并使用 `-Offline`。
- 行数校验失败：检查 `DataRoot` 是否正确、是否使用了正确的 `test.csv`。
