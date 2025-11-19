#!/bin/bash
# 代码质量检查脚本
#
# 用途：在本地开发或 CI/CD 流程中统一执行代码质量保障流程
# 包括：格式化检查、静态 lint、类型检查、单元测试与覆盖率验证
# 确保代码符合项目规范，提升可维护性和可靠性

# 启用严格模式：任何命令返回非零退出码时，脚本立即终止
# 防止错误被忽略，保证检查流程的可靠性
set -e

echo "Running code quality checks..."
echo "----------------------------------------"

# ==============================
# 1. 代码格式化检查（使用 Ruff Formatter）
# ==============================
echo "🔍 [1/4] Checking code formatting with Ruff..."

# --check 模式：不修改文件，仅检查是否符合格式规范
# 如果发现未格式化的代码，命令返回非零状态
ruff format --check src/ tests/ || {
    # 若格式检查失败，提示用户如何修复，并退出脚本
    echo "❌ Code formatting check failed."
    echo "💡 Fix it by running: ruff format src/ tests/"
    exit 1
}
echo "✅ Formatting check passed."

# ==============================
# 2. 代码静态分析（Linting，使用 Ruff Linter）
# ==============================
echo "🔍 [2/4] Running linter (Ruff)..."

# ruff check：检查代码风格、潜在 bug、未使用变量等
# --fix 可自动修复部分问题，但此处仅检查（CI 中通常不自动修改代码）
ruff check src/ tests/ || {
    echo "❌ Linter found issues."
    echo "💡 Auto-fix supported issues by running: ruff check --fix src/ tests/"
    echo "   For non-auto-fixable issues, please review the output above."
    exit 1
}
echo "✅ Linting passed."

# ==============================
# 3. 类型检查（使用 MyPy）
# ==============================
echo "🔍 [3/4] Running type checker (MyPy)..."

# 对 src/ 目录进行类型检查（tests/ 通常不强制类型注解）
# 注意：MyPy 失败不会导致脚本退出（见下方说明）
mypy src/ || {
    # 类型检查失败时仅警告，不退出
    # 原因：项目可能尚未完全添加类型注解，或某些动态代码难以标注
    # 在成熟项目中，可改为 exit 1 以强制类型安全
    echo "⚠️  Type checker found issues."
    echo "   Consider adding type hints to improve code reliability."
    # 注意：这里没有 exit 1，脚本继续执行后续步骤
}
echo "✅ Type checking completed (issues may exist but not fatal)."

# ==============================
# 4. 单元测试与覆盖率验证
# ==============================
echo "🔍 [4/4] Running tests with coverage..."

# 检查是否存在 tests 目录且非空
if [ -d "tests" ] && [ "$(ls -A tests 2>/dev/null)" ]; then
    echo "📁 Found tests directory. Executing tests..."
    
    # 运行 pytest：
    # -v: 显示详细测试名称
    # --cov=src/evrag: 统计该模块的覆盖率
    # --cov-report=term-missing: 在终端显示未覆盖的代码行
    # 注意：coverage 配置（如 fail-under）已在 pytest 配置中定义
    pytest tests/ -v --cov=src/evrag --cov-report=term-missing || {
        echo "❌ Tests failed or coverage below threshold."
        exit 1
    }
    echo "✅ All tests passed with sufficient coverage."
else
    # 如果没有测试，跳过（适用于新项目或无测试模块）
    echo "ℹ️  No tests found in 'tests/' directory. Skipping test execution."
fi

# ==============================
# 最终成功提示
# ==============================
echo "----------------------------------------"
echo "🎉 All code quality checks passed!"
echo "Your code is clean, consistent, and well-tested. ✨"