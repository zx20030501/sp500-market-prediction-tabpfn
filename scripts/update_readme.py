#!/usr/bin/env python3
"""
自动更新 README.md 文件的脚本
在每次 git 更新后运行，自动添加最新的项目信息
"""

import os
import subprocess
from datetime import datetime

def get_git_info():
    """获取 Git 仓库信息"""
    try:
        # 获取最新提交信息
        latest_commit = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H|%an|%ae|%ad|%s'],
            encoding='utf-8'
        ).strip()
        
        commit_hash, author, email, date, message = latest_commit.split('|', 4)
        
        # 获取提交总数
        commit_count = subprocess.check_output(
            ['git', 'rev-list', '--count', 'HEAD'],
            encoding='utf-8'
        ).strip()
        
        # 获取当前分支
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            encoding='utf-8'
        ).strip()
        
        # 获取远程 URL
        remote_url = subprocess.check_output(
            ['git', 'config', '--get', 'remote.origin.url'],
            encoding='utf-8'
        ).strip()
        
        return {
            'commit_hash': commit_hash[:7],
            'full_hash': commit_hash,
            'author': author,
            'email': email,
            'date': date,
            'message': message,
            'commit_count': commit_count,
            'branch': branch,
            'remote_url': remote_url
        }
    except Exception as e:
        print(f"获取 Git 信息时出错: {e}")
        return None

def update_readme():
    """更新 README.md 文件"""
    readme_path = 'README.md'
    
    # 读取现有的 README
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = ""
    
    # 获取 Git 信息
    git_info = get_git_info()
    if not git_info:
        print("无法获取 Git 信息，跳过更新")
        return
    
    # 创建更新时间戳
    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # 查找是否存在自动更新标记
    start_marker = "<!-- AUTO-UPDATE-START -->"
    end_marker = "<!-- AUTO-UPDATE-END -->"
    
    # 构建更新内容
    update_content = f"""{start_marker}

## 📊 项目状态

**最后更新**: {update_time}

### Git 信息
- **当前分支**: `{git_info['branch']}`
- **最新提交**: `{git_info['commit_hash']}`
- **提交信息**: {git_info['message']}
- **提交者**: {git_info['author']}
- **提交时间**: {git_info['date']}
- **总提交数**: {git_info['commit_count']}

{end_marker}"""
    
    # 更新 README 内容
    if start_marker in content and end_marker in content:
        # 如果存在标记，替换中间的内容
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker) + len(end_marker)
        new_content = content[:start_idx] + update_content + content[end_idx:]
    else:
        # 如果不存在标记，追加到文件末尾
        new_content = content.rstrip() + "\n\n" + update_content + "\n"
    
    # 写回 README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ README.md 已更新！")
    print(f"   提交: {git_info['commit_hash']} - {git_info['message']}")

if __name__ == '__main__':
    update_readme()
