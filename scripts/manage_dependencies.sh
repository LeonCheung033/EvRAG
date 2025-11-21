#!/bin/bash
# 自动生成和更新requirements.txt
#
# 问题说明：
# pipreqs 从项目根目录扫描时，可能不会正确扫描 src/ 目录下的文件
# 这导致新添加的依赖（如 torch, transformers, langchain_core 等）无法被检测到
#
# 解决方案：
# 分别扫描 src/ 和 tests/ 目录，然后合并结果，确保所有依赖都被检测到

echo "Generating requirements.txt using pipreqs..."

# 创建一个临时文件来存储合并的依赖
TEMP_REQ=$(mktemp)

# 扫描 src/ 目录（主要代码）
echo "Scanning src/ directory..."
pipreqs src/ --encoding=utf8 --savepath "${TEMP_REQ}.src" 2>&1 | grep -v "INFO\|WARNING" || true

# 扫描 tests/ 目录（测试代码）
echo "Scanning tests/ directory..."
pipreqs tests/ --encoding=utf8 --savepath "${TEMP_REQ}.tests" 2>&1 | grep -v "INFO\|WARNING" || true

# 合并所有依赖并去重，按包名排序
echo "Merging dependencies..."
cat "${TEMP_REQ}.src" "${TEMP_REQ}.tests" 2>/dev/null | sort -u > "${TEMP_REQ}"

# 如果合并后的文件不为空，使用它；否则回退到扫描整个项目
if [ -s "${TEMP_REQ}" ]; then
    mv "${TEMP_REQ}" requirements.txt
    echo "Successfully generated requirements.txt from src/ and tests/ directories"
else
    echo "Warning: Merged requirements empty, falling back to full project scan..."
    pipreqs . --force --encoding=utf8 --savepath requirements.txt
fi

# 清理临时文件
rm -f "${TEMP_REQ}" "${TEMP_REQ}.src" "${TEMP_REQ}.tests"

echo ""
echo "Generated requirements.txt:"
cat requirements.txt

echo ""
echo "Note: Please manually review requirements.txt to remove any false positives"
echo "      (e.g., packages that pipreqs incorrectly identified from import statements)"

echo ""
echo "Checking dependency tree..."
pipdeptree

echo ""
echo "Checking for conflicts..."
pip check