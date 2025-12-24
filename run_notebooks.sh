#!/bin/bash
# 运行所有示例 notebook

set -e  # 遇到错误时退出

cd "$(dirname "$0")"
NOTEBOOK_DIR="examples"

echo "=========================================="
echo "运行所有示例 notebook"
echo "=========================================="

# 检查是否安装了 jupyter
if ! command -v jupyter &> /dev/null; then
    echo "错误: 未找到 jupyter 命令"
    echo "请安装: pip install jupyter"
    exit 1
fi

# Notebook 列表（按依赖顺序）
NOTEBOOKS=(
    "download.ipynb"
    "model.ipynb"
    "types.ipynb"
    "go.ipynb"
    "make.ipynb"
    "gears.ipynb"
)

SUCCESS=0
FAILED=()

for notebook in "${NOTEBOOKS[@]}"; do
    notebook_path="${NOTEBOOK_DIR}/${notebook}"
    
    if [ ! -f "$notebook_path" ]; then
        echo "⚠️  跳过: $notebook_path 不存在"
        continue
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "运行: $notebook"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 运行 notebook（不保存输出，只检查是否成功）
    if jupyter nbconvert --to notebook --execute --inplace "$notebook_path" 2>&1; then
        echo "✅ 成功: $notebook"
        ((SUCCESS++))
    else
        echo "❌ 失败: $notebook"
        FAILED+=("$notebook")
    fi
done

echo ""
echo "=========================================="
echo "运行完成"
echo "=========================================="
echo "成功: $SUCCESS/${#NOTEBOOKS[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "失败: ${FAILED[*]}"
    exit 1
else
    echo "所有 notebook 运行成功！"
    exit 0
fi

