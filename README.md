# 基于 LangChain 与 Qwen-Agent 的 RAG 检索问答系统

用一个 800 字的本地文档（英科医疗）把 RAG 全链路走通，并给出**三套可对比的实现**：
手写版、LangChain 版、Qwen-Agent 工具版。三套实现共用同一份文档、同一套切分参数、
同一个相似度口径，方便横向比较"裸写 RAG"和"框架搭 RAG"的差别，以及 Agent 如何把
检索能力变成可自主调用的工具。

## 项目背景

大模型的知识是固定的，回答私有文档内容时要么答不上来，要么编造（幻觉）。
RAG 的思路是：先从文档里检索出相关片段，再让模型**只依据这些片段**回答。
本项目把这套流程完整实现，并重点解决工程落地时的几个坑：
弱模型绕过工具、工具调用死循环、模型返回的 JSON 参数解析异常。

## 功能特性

- 完整 RAG 链路：文档切分 → Embedding 向量化 → 语义检索 → Prompt 拼接 → 大模型生成
- **两套对比实现**：手写版（requests + numpy 手算余弦）与 LangChain 版（Loader/Splitter/FAISS/LCEL）
- **Agent 工具化**：`RagSearchTool` 继承 `BaseTool` + `@register_tool`，模型自主决定是否检索
- 工具路由兜底：模型把工具调用写成文字时自动恢复执行；模型凭自己知识瞎答时按关键词强制检索
- 向量缓存：文档内容、切分参数、Embedding 模型任一变化，缓存自动失效
- 批量 Embedding（单次 32 条，按 `index` 回填保证顺序）
- 相似度阈值 + Prompt 双重拒答：知识库没有的信息明确拒答，不编造
- API Key 走 `.env` / 环境变量，不写进代码
- 离线测试（假 Embedding，不消耗额度）+ 联网路由测试（三个用例）+ 模型选型脚本

## 目录结构

```
ykdemo/
├── common/                     # 三套实现共用的基础设施
│   ├── config.py               # 统一读取 .env / 环境变量（模型、切分、阈值、路径）
│   ├── zhipu_client.py         # 手写 HTTP 封装：Embedding / Chat，含超时、重试、异常
│   ├── retrieval.py            # 切分 + 余弦检索 + 阈值过滤 + 向量缓存
│   └── qa.py                   # RAG 问答编排（Prompt 模板与拒答分支）
├── handcrafted_rag/main.py     # 手写版：不用 LangChain，全流程自己实现
├── langchain_rag/main.py       # LangChain 版：TextLoader/Splitter/FAISS/LCEL
├── qwen_agent_rag/
│   ├── rag_tool.py             # RagSearchTool（BaseTool + @register_tool）
│   └── agent_demo.py           # Assistant + 工具路由 system_message + 死循环保护
├── tools/compare_models.py     # 模型选型：glm-4-flash vs glm-4-air 路由稳定性对比
├── tests/
│   ├── test_offline_pipeline.py   # 离线：切分/排序/阈值/拒答/参数解析
│   ├── test_langchain_pipeline.py # 离线：LangChain 版与手写版分数口径一致性
│   └── test_tool_routing.py       # 联网：三个工具路由用例
├── data/yingke.txt             # 示例知识库文档
├── requirements.txt
└── .env.example
```

## 快速开始

1. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

2. 配置 API Key（在 [智谱 AI 开放平台](https://open.bigmodel.cn/) 创建）

   ```bash
   cp .env.example .env       # Windows: copy .env.example .env
   # 编辑 .env，填入 ZHIPU_API_KEY=你的key
   ```

   也支持直接设置环境变量（Windows `set ZHIPU_API_KEY=...`，Linux/macOS `export ZHIPU_API_KEY=...`）。

3. 运行三种实现

   ```bash
   python -m handcrafted_rag.main      # 手写版
   python -m langchain_rag.main        # LangChain 版
   python -m qwen_agent_rag.agent_demo "英科医疗2025年上半年的营收和净利润是多少？"
   ```

   三个入口都支持"直接点 PyCharm 的 Run"（脚本方式运行也能找到项目内模块），
   交互模式下输入 `q` 退出。

## 三套实现怎么选、有什么差别

| | 手写版 | LangChain 版 | Qwen-Agent 版 |
| --- | --- | --- | --- |
| 检索 | requests + numpy 手算余弦 | FAISS 向量库 | 复用 `common/retrieval.py` |
| 编排 | 自己拼 Prompt 字符串 | LCEL：`prompt \| llm \| parser` | Agent 自主决策 + 工具调用 |
| 交互 | 命令行问答 | 命令行问答 | 对话式，模型自己决定何时检索 |
| 适合看什么 | RAG 每一步的原理 | 框架组件怎么串起来 | 工具调用/路由/死循环治理 |

### 手写版与 LangChain 版的两个对照点

**1. 中文切分不能沿用默认分隔符。**
`RecursiveCharacterTextSplitter` 默认按空格、换行等切，中文长句会被硬切。
这里显式指定 `["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]`，
优先按段落切，其次句子，逗号兜底。

**2. 两版的相似度必须是同一个口径。**
FAISS 默认用 L2 距离，它的 `similarity_search_with_relevance_scores` 返回的分数
和余弦相似度不是一回事；而手写版算的是余弦。如果直接套同一个阈值，
两边的召回结果会完全不同，"对比"就没有意义了。
所以 LangChain 版显式使用 `normalize_L2=True + MAX_INNER_PRODUCT`，
让 `similarity_search_with_score` 返回的就是余弦相似度——
`tests/test_langchain_pipeline.py` 里有专门的用例断言两边分数一致。

## 检索参数与阈值怎么定

三个参数在 `.env`（或环境变量）里配置，默认值：`RAG_CHUNK_SIZE=200`、
`RAG_CHUNK_OVERLAP=30`、`RAG_TOP_K=3`、`RAG_MIN_SCORE=0.35`。

**阈值没有万能值，必须用真实分数校准。** 项目内置了诊断模式，只检索不生成，
直接把各块的分数打出来：

```bash
python -m handcrafted_rag.main --diagnose "英科医疗2025年上半年的营收和净利润是多少"
python -m handcrafted_rag.main --diagnose "帮我写一个Python快速排序"
```

下面是本项目示例文档的**真实实测分数**（`embedding-3`，chunk_size=200 / overlap=30，
共 5 块，取相似度最高的块）：

| 问题 | 类型 | 最高相似度 |
| --- | --- | --- |
| 英科医疗的总部和上市情况？ | 相关 | 0.6666 |
| 英科医疗2025年上半年的营收和净利润是多少？ | 相关 | 0.6515 |
| 一次性防护手套的年化产能是多少？ | 相关 | 0.5124 |
| 英科医疗今天的股价是多少？ | 知识库里没有 | 0.5048 |
| 帮我写一个Python快速排序 | 无关 | 0.2246 |
| 今天天气怎么样 | 无关 | 0.1964 |

结论：

- 无关问题 ≤ 0.23，相关问题 ≥ 0.51，阈值取 **0.35** 两侧都有约 0.12 的安全余量，
  所以默认值是 0.35 而不是随手写的 0.3 或 0.5；
- 但**"同一实体 + 知识库里没有的信息"**（"英科医疗今天的股价"）分数是 0.5048，
  比真正相关的问题只低一点点，**单靠阈值挡不住**（把阈值抬到 0.51 会连
  "手套产能"这种真问题一起误杀）。这类问题靠第二道防线：
  Prompt 里写死"资料中没有答案时只回答『根据现有资料无法回答』"，由模型判断
  检索到的资料能不能回答问题。

## 测试

```bash
# 离线用例：不联网、不消耗额度，验证切分/排序/阈值/拒答/参数解析
python -m pytest tests/test_offline_pipeline.py tests/test_langchain_pipeline.py -v

# 联网用例：三个工具路由测试（需要有效 API Key，未配置时自动跳过）
python -m pytest tests/test_tool_routing.py -v
```

三个路由用例（对应"项目成果"里的验收标准）：

| 用例 | 输入 | 期望行为 |
| --- | --- | --- |
| 1. 知识库问题 | 英科医疗2025年上半年的营收和净利润是多少？ | 调用 `rag_search`，答案来自知识库，不拒答 |
| 2. 非知识库问题 | 帮我写一个Python快速排序 | **不**调用任何工具，直接回答 |
| 3. 超范围问题 | 英科医疗今天的股价是多少？ | 返回拒答话术，不编造 |

### 实测记录

（2026-10-09，本机 Windows + Anaconda Python 3.11.7，真实调用智谱接口）

| 项目 | 结果 |
| --- | --- |
| 离线用例 | 24 个全部通过；3 个联网用例按设计跳过（`python -m pytest tests -q`） |
| 联网路由用例 | 3 个全部通过（用例 1 借助"文本工具调用恢复"兜底，用例 2/3 原生行为符合预期） |
| 模型对比 | glm-4-flash 9/9、glm-4-air 9/9（共 18 次真实调用，明细见 [docs/model_comparison.md](docs/model_comparison.md)） |
| 相似度阈值 | 相关问题 0.51–0.67、无关问题 0.20–0.22，据此确认默认阈值 0.35 |
| 依赖版本 | langchain 0.2.17 / langchain-community 0.2.19 / faiss-cpu 1.15.1 / qwen-agent 0.0.34 / zhipuai 2.1.5 |
| 接口 | `embedding-3`（2048 维）；`glm-4-flash`、`glm-4-air`、`glm-4.5-flash`、`glm-4.7-flash`、`glm-4-flash-250414`、`glm-4-air-250414`、`glm-4-plus` 均可用 |

## 工程问题与解决

### 1. 弱模型绕过工具，直接用自己知识回答

只写一句"你可以使用 rag_search 工具"是不够的：`glm-4-flash` 这类偏小的模型
经常直接用自己的知识回答英科医疗的问题，把知识库晾在一边。
解决办法是把 `system_message` 写成明确的**路由规则**：

1. 英科医疗相关问题 → 必须先调用 `rag_search`，且把工具返回的内容作为最终答案，
   不要用模型自己的知识改写；
2. 无关问题（写代码、数学、闲聊）→ 直接回答，不要调用工具；
3. 工具返回拒答话术 → 原样输出。

规则写清楚之后情况好转，但**规则约束不足以让所有模型都听话**。
实测 glm-4-flash 仍有两种绕过方式，所以又加了两道代码层兜底
（见 `qwen_agent_rag/agent_demo.py`）：

| 绕过方式 | 实测现象 | 兜底处理 |
| --- | --- | --- |
| 把工具调用写成正文 | 输出 `rag_search\n{"query": "..."}`，框架识别不到，用户直接看到这段文字 | `recover_text_tool_call()` 用正则识别，并真的执行一次 `rag_search`，用工具结果替换这段文本 |
| 完全不调工具、凭自己知识回答 | 直接编造"营收为XX亿元，同比增长XX%" | `needs_forced_tool_call()` 命中关键词（默认"英科"，可用 `RAG_FORCE_TOOL_KEYWORDS` 配置）时，强制补一次检索 |

两道兜底互不冲突：先尝试恢复文本工具调用，再按关键词强制检索，
每次问答最多各触发一次。`tools/compare_models.py` 的输出里会分别标注
哪种兜底被触发，便于判断模型本身的稳定性。

### 2. 工具调用死循环

工具返回答案后，模型有时会"再确认一次"，又调一次 `rag_search`，来回几轮。
两层处理：

- Prompt 层：明确告诉模型"工具返回的内容就是最终答案，不要再调用本工具"；
- 代码层：`run_agent` 里限制单轮最多调用 3 次工具，超限强制结束本轮，
  并把最后一次工具结果作为答案（`max_tool_calls` 可调）。

### 3. 模型返回的 JSON 参数解析异常

不同模型给的 `params` 形式不一样：有时是标准 JSON 字符串，有时前后带一句解释
（`好的，我来调用 {"query": "..."}`），有时干脆把问题原文丢过来。
`RagSearchTool._parse_query` 的处理顺序是：

1. `dict` 直接取 `query`；
2. `json.loads` 标准解析；
3. `JSONDecoder().raw_decode` 容忍前后噪声；
4. 都失败就把整段文本当作 query 用。

这样任何一条路径都不会抛异常导致 Agent 中断，
`tests/test_offline_pipeline.py::RagToolParsingTest` 覆盖了这四种情况。

## 模型选型：glm-4-flash vs glm-4-air

工具路由是否稳定，直接决定 Agent 能不能用，所以选型不看"谁更聪明"，
只看**该调工具时调不调、不该调时乱不乱调、会不会死循环**。

`tools/compare_models.py` 用同一组用例（上面三个）轮流打给不同模型：

```bash
python -m tools.compare_models --models glm-4-flash glm-4-air --repeat 3
```

输出包含每个用例的工具调用次数、是否触发循环保护、是否拒答、耗时，并可写成 markdown 表：

```bash
python -m tools.compare_models --out reports/model_comparison.md
```

实测结果（每个模型 × 3 个用例 × 3 轮 = 9 次，完整明细见
[docs/model_comparison.md](docs/model_comparison.md)）：

| 模型 | 路由正确率 | 原生调用工具 | 需要兜底恢复 | 触发死循环保护 | 单轮耗时 |
| --- | --- | --- | --- | --- | --- |
| glm-4-flash | 9/9 | 7/9 | 2/9（把工具调用写成了纯文本） | 0 次 | 7.8–15.1s |
| glm-4-air | 9/9 | 9/9 | 0/9 | 0 次 | 6.2–25.9s |

结论：

- **glm-4-air 在 Function Calling 上明显更稳**：9 次全部原生发起工具调用，
  没有出现过"把工具调用写成文字"或"直接用自己的知识回答"的情况；
- **glm-4-flash 能用但需要兜底**：9 次里 2 次不走 Function Calling，
  而是把 `rag_search` 与 JSON 参数当普通正文吐出来（框架识别不到，
  会把这段文字当答案返回给用户）。手工诊断时还出现过一次更糟的情况：
  完全不调工具，直接编出"营收为XX亿元、同比增长XX%"这种占位式回答；
- 所以本项目的默认模型仍是免费的 `glm-4-flash`，但配了两道兜底
  （文本工具调用恢复 + 关键词强制检索）；如果追求原生稳定、
  不希望依赖兜底规则，把 `RAG_CHAT_MODEL` 设为 `glm-4-air` 即可，
  换模型不用改代码。

## 已知限制

- 知识库只有一份 800 字的示例文档，切分成 5 块做全量遍历；
  文档变大会需要真正的向量数据库（当前 FAISS 版已经是下一步的基础）。
- 相似度阈值需要按自己的知识库实测校准，默认 `0.35` 只是起点。
- 检索是纯向量召回，没有做重排序（Rerank）和混合检索。

## 后续改进方向

- 接入 Rerank 提升检索精度
- 支持多格式文档（PDF/Word）与增量索引
- 多轮对话记忆 + 引用来源（把命中的原文片段一并返回给用户）
- 用 `--diagnose` 的分数分布做阈值自动校准
