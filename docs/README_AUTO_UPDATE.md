# README 自动更新系统说明

## 📚 概述

本项目实现了自动更新 README.md 文件的功能。每次代码推送到 `main` 或 `master` 分支后，GitHub Actions 工作流会自动运行，更新 README 中的项目状态信息。

## 🚀 工作原理

### 1. GitHub Actions 工作流

位置：`.github/workflows/update-readme.yml`

工作流在以下情况下触发：
- 推送到 `main` 或 `master` 分支
- 手动触发（workflow_dispatch）

工作流执行以下步骤：
1. 检出代码库
2. 设置 Python 环境
3. 运行更新脚本
4. 如果有变更，自动提交并推送

### 2. 更新脚本

位置：`scripts/update_readme.py`

脚本功能：
- 从 Git 仓库获取最新信息（提交哈希、作者、时间等）
- 在 README.md 中查找自动更新标记
- 更新标记之间的内容
- 保持 README 其他部分不变

### 3. README 标记

在 README.md 中使用特殊标记来标识自动更新区域：

```markdown
<!-- AUTO-UPDATE-START -->
<!-- 此部分内容会自动更新 -->
<!-- AUTO-UPDATE-END -->
```

标记之间的内容会被脚本自动替换为最新的项目状态信息。

## 📋 显示的信息

自动更新的内容包括：
- 最后更新时间
- 当前分支名称
- 最新提交的短哈希
- 提交信息
- 提交者信息
- 提交时间
- 总提交数

## 🔧 手动触发

如果需要手动触发更新，可以：

1. 在 GitHub 网页上：
   - 进入 Actions 标签页
   - 选择 "更新 README" 工作流
   - 点击 "Run workflow" 按钮

2. 在本地测试：
   ```bash
   python scripts/update_readme.py
   ```

## ⚙️ 配置说明

### 修改触发分支

编辑 `.github/workflows/update-readme.yml` 中的 `branches` 部分：

```yaml
on:
  push:
    branches:
      - main        # 添加或修改分支名称
      - master
      - develop     # 可以添加更多分支
```

### 自定义更新内容

编辑 `scripts/update_readme.py` 中的 `update_content` 变量，可以：
- 修改显示的信息格式
- 添加更多 Git 统计信息
- 自定义样式和排版

### 跳过自动更新

在提交信息中包含 `[skip ci]` 可以跳过工作流运行：

```bash
git commit -m "docs: 更新文档 [skip ci]"
```

## 🔒 权限说明

工作流需要 `contents: write` 权限才能提交更改。这已在工作流文件中配置：

```yaml
permissions:
  contents: write
```

## 📝 注意事项

1. **避免冲突**：更新脚本使用提交信息 `[skip ci]` 避免触发无限循环
2. **标记完整性**：不要手动删除或修改 `<!-- AUTO-UPDATE-START/END -->` 标记
3. **手动编辑**：可以在标记外的区域自由编辑 README
4. **Python 版本**：脚本需要 Python 3.x，工作流中已配置

## 🐛 故障排除

### 工作流未运行
- 检查是否推送到正确的分支（main/master）
- 检查 GitHub Actions 是否启用
- 查看 Actions 标签页的错误信息

### README 未更新
- 确保 README.md 包含正确的标记
- 检查脚本输出的错误信息
- 验证 Git 仓库信息是否可访问

### 权限错误
- 确保工作流有 `contents: write` 权限
- 检查仓库的 Actions 权限设置

## 🔗 相关链接

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [工作流语法参考](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

## 📄 许可证

本自动更新系统遵循项目主许可证。
