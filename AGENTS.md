# Repository Guidelines

## 项目结构

可安装的 Python 包位于 `src/papr_ssl/`，按功能分为 `models/`、`training/`、`data/`、`inference/`、`losses/` 和 `experiments/`。`tests/` 中的测试大体对应这些模块，也包含仓库级和 Demo 契约测试。`scripts/` 存放审计、训练和演示入口；`configs/` 存放配置；`docs/` 记录协议与项目状态；`demo/` 存放 Web 演示。`datasets/` 和 `artifacts/` 用于本地研究数据与输出，不是源代码目录。

## 构建、测试与开发命令

使用 Python 3.10 或更高版本。在仓库根目录安装：`python -m pip install -e .`。离线检查命令如下：

```powershell
python -m compileall src tests
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

以上命令编译源码和测试，并运行标准库 `unittest` 测试套件。准备好演示依赖和本地数据后，可运行浏览器 Demo：`python scripts\run_demo05_web.py`。测试使用模拟对象，不应下载检查点或数据集。

## 编码风格与命名约定

遵循现有 Python 风格：缩进为 4 个空格；模块、函数和变量使用 `snake_case`，类使用 `PascalCase`。模块应聚焦于现有包的功能划分；公共接口的类型和文档说明应与相邻代码保持一致。`pyproject.toml` 未配置格式化或 lint 工具；除非项目确有需要，不要引入新的风格工具。

## 测试规范

使用 `unittest`；测试文件命名为 `test_*.py`，测试用例命名为 `test_*`。测试应放在与所测代码或协议对应的目录中。行为变更应补充回归测试，适用时也添加契约或静态检查。保持 `generic_test` 封存；模型和阈值只能依据开发集选择，详见 `docs/EVALUATION_PROTOCOL.md`。

## 提交与拉取请求

近期提交说明简短、使用祈使语气，常用中文，且没有固定的 Conventional Commits 前缀。每个提交聚焦单一事项，并清楚说明改动。拉取请求应说明原因和范围，列出相关验证命令及结果，关联对应问题或协议文档；涉及 Demo 外观变化时附上截图。数据集、配置或评估协议有变更时，应明确说明。

## 数据与配置

不要提交下载的数据集、模型检查点、生成的缓存或个人凭据。配置变更应可复现；除非改动明确更新了文档契约，否则保留已固定的模型版本和评估边界。

## 语言规范

- 与我交互默认使用中文。
- 项目说明、任务总结、代码审查和开发文档默认使用中文。
- 技术术语首次出现时可附英文原词。
- 代码、命令、路径和日志原文保持原样。
