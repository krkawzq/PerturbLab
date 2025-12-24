#!/usr/bin/env python3
"""运行所有示例 notebook。

使用 jupyter nbconvert 执行每个 notebook 并检查结果。
"""

import subprocess
import sys
from pathlib import Path

# Notebook 列表（按依赖顺序）
NOTEBOOKS = [
    "download.ipynb",
    "model.ipynb",
    "types.ipynb",
    "go.ipynb",
    "make.ipynb",
    "gears.ipynb",
]

def run_notebook(notebook_path: Path) -> bool:
    """运行单个 notebook。
    
    Args:
        notebook_path: Notebook 文件路径
        
    Returns:
        是否成功运行
    """
    print(f"\n{'='*50}")
    print(f"运行: {notebook_path.name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✅ 成功: {notebook_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {notebook_path.name}")
        print(f"错误输出:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ 错误: 未找到 jupyter 命令")
        print("请安装: pip install jupyter")
        return False


def main():
    """主函数。"""
    script_dir = Path(__file__).parent
    examples_dir = script_dir / "examples"
    
    if not examples_dir.exists():
        print(f"❌ 错误: 未找到 {examples_dir} 目录")
        sys.exit(1)
    
    print("=" * 50)
    print("运行所有示例 notebook")
    print("=" * 50)
    
    success_count = 0
    failed = []
    
    for notebook_name in NOTEBOOKS:
        notebook_path = examples_dir / notebook_name
        
        if not notebook_path.exists():
            print(f"⚠️  跳过: {notebook_path} 不存在")
            continue
        
        if run_notebook(notebook_path):
            success_count += 1
        else:
            failed.append(notebook_name)
    
    print("\n" + "=" * 50)
    print("运行完成")
    print("=" * 50)
    print(f"成功: {success_count}/{len(NOTEBOOKS)}")
    
    if failed:
        print(f"失败: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("所有 notebook 运行成功！")
        sys.exit(0)


if __name__ == "__main__":
    main()

