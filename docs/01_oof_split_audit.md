# OOF/时间切分审计（Hull TabPFN）

生成时间：2026-02-07 14:20:36

## 结论清单
- Split 类型：单次前向切分（train 日期 < test_start），非 walk-forward / expanding CV；未实现 purged / embargo。
- 预测目标时间点与特征窗口在其之前：YES（基于当前 pipeline 的 lagged_ 生成与前视列过滤），但需假设原始特征本身不含未来信息。
- 未来信息字段/滚动统计穿越检查（清单）：
- [x] `forward_returns`、`market_forward_excess_returns`、`risk_free_rate` 被排除（不进入特征）。
- [x] 名称包含 `forward/future/lead/next` 的列被过滤（`lagged_` 前缀除外）。
- [x] 仅生成 `lagged_` 列（`shift(1)`），无 `shift(-1)`。
- [ ] 数据源字段是否自带“未来滚动统计”需结合原始特征定义核验（代码无法自动识别）。

## 代码入口（可审计）
- Split 清理：`src/run_tabpfn_pipeline.py` → `load_data`（剔除 `date_id >= test_start`）
- 前视列过滤：`src/run_tabpfn_pipeline.py` → `_is_forward_looking`
- lagged 特征：`src/run_tabpfn_pipeline.py` → `build_features`（`shift(1)`）

## 可重复验证脚本
脚本：`scripts/audit_oof_split.py`

### 用法
```powershell
# 单次切分（按 train/test 计算）
python scripts\audit_oof_split.py --data-root data\hull-tactical-market-prediction

# 如果你有 fold 切分文件（CSV），需包含：fold, train_start, train_end, valid_start, valid_end
python scripts\audit_oof_split.py --folds-csv path\to\folds.csv
```

### 本次运行输出（证据）
```text
=== Split Summary (single holdout) ===
date_col: date_id
test_start: 8980
train_end: 8979
train_rows: 8980
test_rows: 10
overlap_dates_in_raw_train: 10
train_end < test_start: True
overlap_dates_sample: [8980, 8981, 8982, 8983, 8984]
=== Forward-looking Column Scan (name-based) ===
keywords: forward, future, lead, next
- forward_returns
- market_forward_excess_returns
```

## 说明
- 当前 pipeline 仅做单次前向切分，不包含 rolling OOF 或 purge/embargo。
- 若未来添加 CV/rolling，请把每折的 `train_start/end` 与 `valid_start/end` 落盘为 CSV 供脚本复核。
