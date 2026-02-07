# 技术总览（Hull TabPFN）

## Pipeline
如果你的 Markdown 渲染器不支持 Mermaid，请直接使用 PNG/SVG：

![Pipeline](./pipeline.png)

[SVG 版本](./pipeline.svg)

<details>
<summary>Mermaid 源码（可选）</summary>

```mermaid
flowchart LR
  Data[Raw data (train/test)] --> Feature[Feature build + lagged cols]
  Feature --> Model[TabPFN regressor]
  Model --> Calib[Calibration (scale/shift by adjusted Sharpe)]
  Calib --> Position[Position sizing (clip)]
  Position --> Submission[submission.csv]
```

</details>

## OOF/时间切分（可审计说明）
- 生产脚本默认是“单次前向切分 + 校准窗口”，不是 rolling OOF。
- 训练集去泄漏：以 `test.csv` 的最小 `date_id` 作为测试起点，剔除训练集中 `date_id >= test_start` 的行。
- 校准窗口：只取训练集末尾 `calibration_window` 天（默认 180）优化 scale/shift。
- 前视特征过滤：排除包含 `forward/future/lead/next` 的列（`lagged_` 前缀除外）。
- 代码入口：`src/run_tabpfn_pipeline.py` 中 `load_data`、`_is_forward_looking`、`calibrate_predictions`。
- 若需要 rolling OOF，请在 Notebook 里扩展时间滑窗验证（未接入 `run_all.ps1`）。

## 评分实现与对齐
- 评分实现入口：`src/run_tabpfn_pipeline.py` 中 `adjusted_sharpe`。
- 校准入口：`src/run_tabpfn_pipeline.py` 中 `calibrate_predictions`（Powell 优化 scale/shift）。
- 评分公式与惩罚项说明见 `docs/scoring.md`。

## 推理与提交映射
- 模型输出：`TabPFNService.predict_raw` 生成对 `forward_returns` 的连续预测。
- 映射：`apply_calibration` 进行 `scale/shift`，并 clip 到 `[min_position, max_position]`（默认 0 到 2）。
- 提交字段：`run_pipeline` 自动生成 `submission.csv`，列为 `row_id/id` 与 `prediction`。
- 代码入口：`src/run_tabpfn_pipeline.py` 中 `TabPFNService.predict` 与 `run_pipeline`。

## Ablation 建议
- 关闭校准：对比 `predict_raw` 与校准后 `predict` 的分数差异。
- 改变校准窗口：例如 60/180/360 日。
- 去掉 lag 特征或前视列过滤，观察速度与稳定性影响。
- 调整下采样规模：例如 5k/10k/20k。
- CPU 与 GPU 运行对比：关注耗时与稳定性。
