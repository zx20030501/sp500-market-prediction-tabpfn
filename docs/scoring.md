# 评分实现（Adjusted/Modified Sharpe）

本项目在本地复现了 Kaggle 的“调整后 Sharpe”评分逻辑，并用于校准（scale/shift）与本地评分输出。

## 代码入口
- `src/run_tabpfn_pipeline.py` 中 `adjusted_sharpe`
- `src/run_tabpfn_pipeline.py` 中 `calibrate_predictions`
- `scripts/score_local.py` 读取校准后的本地分数

## 评分伪代码
```python
# 输入：
# solution: DataFrame 包含 forward_returns, risk_free_rate
# positions: 模型输出或校准后的仓位
# min_position, max_position: 仓位上下限

def adjusted_sharpe(solution, positions, min_position, max_position):
    positions = clip(positions, min_position, max_position)

    strategy_returns = risk_free_rate * (1 - positions) + positions * forward_returns
    excess = strategy_returns - risk_free_rate

    mean_excess = geometric_mean(1 + excess) - 1
    std_excess = std(strategy_returns)
    sharpe = mean_excess / std_excess * sqrt(252)

    market_excess = forward_returns - risk_free_rate
    market_mean = geometric_mean(1 + market_excess) - 1
    market_vol = std(market_excess) * sqrt(252) * 100
    strat_vol = std(strategy_returns) * sqrt(252) * 100

    excess_vol_penalty = 1 + max(0, strat_vol / market_vol - 1.2)
    return_gap = max(0, (market_mean - mean_excess) * 100 * 252)
    return_penalty = 1 + (return_gap ** 2) / 100

    score = sharpe / (excess_vol_penalty * return_penalty)
    return min(score, 1_000_000)
```

## 校准（scale/shift）对齐评分
- 目标：在训练集末尾窗口（默认 180 天）上最大化 `adjusted_sharpe`。
- 参数：scale/shift 在区间 `[0.8, 1.2]` 与 `[-0.5, 0.5]` 内搜索。
- 优化：`Powell` 方法，若失败则回退到 `scale=1.0, shift=0.0`。

## 本地评分输出
- `scripts/score_local.py` 会打印 `raw_sharpe` 与 `adjusted_sharpe`。
- 这些值来自 `outputs*/artifacts/tabpfn_model/metadata.json` 的 `calibration` 字段。
