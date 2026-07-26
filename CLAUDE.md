# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这个仓库是什么

只需一个 HuggingFace model ID，analyzer 就能产出完整的推理部署分析——显存拆解、并行切分、roofline 性能上界——外加一份可交互的单文件 HTML 报告。全程不下载权重：用 `config.json` 公式与 safetensors 分片头（HTTP Range 请求只读前几百 KB）互相对账，从而暴露真实 dtype 与亚字节打包（fp4/int4）。线上服务：<https://llm-inference-analyzer.ybalbert.people.aws.dev>。

## 常用命令

```bash
# 直接运行 analyzer CLI（纯 stdlib，无需 venv）
cd webapp/analyzer
python3 main.py zai-org/GLM-5.2-FP8 --context 131072 --requests 16 --html out.html --lang zh

# 本地运行 webapp
cd webapp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python import_existing.py     # 可选：导入 html_output/ 里已有的报告
DEV_MODE=1 .venv/bin/python app.py      # http://localhost:8000，免登录（身份=dev）
```

没有测试套件和 linter。验证是经验性的：skill 行为 eval 在 `skills/llm-inference-analyzer/evals/evals.json`；数学正确性靠真实 SGLang 部署实测对账，记录在 `validate_experiments/`。

Git：push 一律用 `git push --no-verify`（pre-push hook 会拦截普通 `git push`）。

## 架构

三层，由计算核心向外：

1. **`webapp/analyzer/` —— 计算核心。**
   - `main.py`（CLI 入口）：从 HF 拉取 `config.json` + safetensors 头，推导各部件参数量、KV-cache cell size，并用 `template.html`/`template.js` 渲染 HTML 报告。纯 stdlib；gated 仓库靠 `HF_TOKEN` 环境变量（生产服务器刻意不设）。
   - `engine.py`：所有 what-if 数学（并行 tab、roofline tab、判定卡片）的**唯一实现**。它是 `template.js` 的 Python 移植，关键处逐行对齐——parity 注记在其 docstring 里。`template.js` 是纯渲染器：报告页每次控件变化都 fetch `/api/v1/whatif`；**浏览器端刻意零数学**。改数学只改 `engine.py`/`main.py`——`engine_version()`（两文件内容 hash）随每个 API 响应标识数学版本。
   - `i18n.py`：zh/en 文案；报告首屏语言由 `--lang` 决定，页内自带切换器。

2. **`webapp/app.py` —— FastAPI 封装。** 对 analyzer 的只读外壳：HF OAuth 登录（只取 username，永不经手用户 token）、SQLite 报告缓存（`data/app.db`，LRU 淘汰）、后台报告生成（子进程跑 `main.py`，经 `/api/tasks/{id}` 轮询）、限流。匿名 JSON API：`/api/v1/analyze`（主力）、`/api/v1/catalog`（硬件表）、`/api/v1/whatif`（渲染器数据源）。gated 模型不支持是 by design（入队前 HEAD `config.json` 直接拒绝）。

3. **`skills/llm-inference-analyzer/` —— Agent Skill。** 瘦远程 API 客户端：`SKILL.md` **不含任何计算逻辑**，只有回答纪律（机型推荐三级结构、每个数字带出处、MoE/MLA 默认 dp-attention、report_url 交接）。API schema 见 `references/api.md`。数学与 SKILL.md 表述冲突时，以线上 API 为准。

## 核心抉择和考虑（compute core 的演进决策）

以下取舍来自 `webapp/analyzer/` 计算核心的演进历史（commit hash 可查 diff），改数学前先对齐：

- **计算逻辑单一化，一致性 > 离线性**（0f7a944）。计算曾有两份实现：Python `main.py`（终端报告）+ `template.js`（浏览器 what-if），且缓存的自包含 HTML 在引擎修 bug 后仍展示旧数学。最终决策：新建 `engine.py` 作为 what-if 数学的唯一实现，`template.js` 删掉 ~600 行计算器变纯渲染器（每次控件变化 fetch `/api/v1/whatif`）。**任何新数学只进 `engine.py`/`main.py`，绝不回到浏览器端**

- **KV 池按供给侧建模，对齐 SGLang 真实语义**（c468009）。并行 tab 曾按需求侧（ctx × req × cell）展示 KV，与真实 server 永远对不上——SGLang 按 `frac × cap − fixed − weights` 预分配 KV 池，与流量无关。改为供给侧视角后，池容量成为可直接对账的预测值（对 `max_total_num_tokens`），固定开销成为对账方程中唯一未知数可反解——这是后续所有实测验证方法的基础。

- **Fail loudly，按原语覆盖而非按模型覆盖**（90ae3f9）。曾假设每层都持全上下文 paged KV，混合架构（linear+full）、滑窗、块稀疏模型全部高估。修复引入 `kv_structure()` 分层分类，并区分两个曾被混同的概念：KV **存储**（滑窗封顶、linear 层无 paged KV）vs decode **读取上限**（DSA top-k、块稀疏只减读不减存）。同时确立原则：遇到未建模的 config 字段显式告警"该结构未建模，数字不可信"，绝不沉默套错公式。

- **每个数学修正都由实测偏差立案、复测收敛结案**。DSA indexer KV：每 token 每层还有一个 fp8 index-key（132 B，随池分页但**不随 kv-dtype 变**），漏算导致 GLM KV 偏 +23%，修后 +0.017%（99a3a9b）。dp-attention 权重复制漏三个部件——embed 整份、MLA 吸收物化 w_kc/w_vc（不在 safetensors 口径内）、draft embed——逐条对 SGLang 源码确认（cfceb37）。vision tower 建模实测每卡 −0.04%（5d980c1）。写预测（expected_*.md）在测量**之前**，防止事后拟合。

- **roofline 与显存共用一套复制口径**（f917a23、34fda57）。每卡权重 = sliced/TP + replicated；roofline 曾无视 dp-attention 统一 ÷TP，修复用"复制倍数"建模（全局字节 ×factor 后共享 ÷TP）。注意极性相反：纯 TP 下 MLA KV 无 head 维每卡整份读（×tp），dp-attention 下各卡只读自己 1/tp 请求的 KV（×1）。混合架构 attention core 只计 KV 层，linear/SSM state 是每步固定读写项（34fda57，修前 65K TTFT 高估 +69%）。

- **简单可解释 > 精确拟合**（029e080、5a91b27）。serving transient（激活峰值在静态区**外**，过高 frac 会"启动正常、首个满 chunk prefill 时崩"）的斜率可分解到 fp8 staging 等细项，但最终代码选择 基础项 × 固定乘数（×1.6 runtime，带 MTP 的模型再 ×1.15 且默认 MTP 开启）。主要原因是模型结构种类复杂，从静态信息无法持续全面精准的估算动态区显存。
