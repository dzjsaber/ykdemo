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

调参经验：

- 相关问题（"营收是多少"）和无关问题（"写个快速排序"）的分数通常能拉开明显差距，
  阈值取在两者之间即可；
- 但**"同一实体 + 知识库里没有的信息"**（例如问"英科医疗今天的股价"）分数并不低，
  单靠阈值挡不住。这类问题要靠第二道防线：Prompt 里写死
  "资料中没有答案时只回答『根据现有资料无法回答』"。

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

## 工程问题与解决

### 1. 弱模型绕过工具，直接用自己知识回答

只写一句"你可以使用 rag_search 工具"是不够的：`glm-4-flash` 这类偏小的模型
经常直接用自己的知识回答英科医疗的问题，把知识库晾在一边。
解决办法是把 `system_message` 写成明确的**路由规则**：

1. 英科医疗相关问题 → 必须先调用 `rag_search`，且把工具返回的内容作为最终答案，
   不要用模型自己的知识改写；
2. 无关问题（写代码、数学、闲聊）→ 直接回答，不要调用工具；
3. 工具返回拒答话术 → 原样输出。

规则写清楚之后，模型的工具调用行为就稳定多了（见 `qwen_agent_rag/agent_demo.py`
里的 `SYSTEM_MESSAGE`）。

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

仓库里不贴未实测的分数：请在本机跑一遍，把生成的表格贴到本节。
选型思路是先看路由正确率（三个用例是否符合预期），再看循环保护和响应耗时；
如果小模型路由不稳，就把 `RAG_CHAT_MODEL` 换成更强的模型
（换模型不用改代码，`.env` 或 `--model` 传参即可）。

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
