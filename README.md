# RAG 检索问答 Demo

基于 LangChain + 智谱 GLM 的本地文档检索问答系统，包含手写版和 LangChain 版两套实现。

## 项目背景

大模型知识固定，无法回答私有文档内容，且容易产生“幻觉”。本项目实现了一个最简 RAG 流程：先检索文档中相关内容，再让模型基于这些内容回答。

## 功能

- 读取本地文档（txt）
- 文档切分（chunk_size=500，overlap=75）
- Embedding 向量化（智谱 embedding-3）
- 向量缓存（文档 MD5 做 key，文档未变则跳过计算）
- 批量调用 Embedding 接口（单次 32 条）
- 余弦相似度检索 + 相似度阈值过滤（低于 0.3 拒答）
- 调用大模型生成回答（glm-4.7-flash，temperature=0.3）
- 统一异常处理与日志输出

## 技术栈

- Python 3
- LangChain
- FAISS 向量库
- 智谱 GLM-4.7-Flash / embedding-3

## 项目结构

```
ykdemo/
├── main.py              # 手写版：完整 RAG 流程，含缓存、批量调用、异常处理
├── main_langchain.py    # LangChain 版：用框架组件封装
├── yingke.txt           # 示例文档
└── .gitignore
```

## 关键实现说明

### 为什么切 500 字、15% 重叠
中文一段话约 200-500 字，500 字能保住完整语义。15% 重叠防止答案刚好被切在块边界上。

### 为什么用向量检索
关键词匹配只能找字面相同的内容，向量检索通过语义相似度能找到意思相近的段落。例如问“公司靠什么营收”，即使文档中没有“营收”二字，也能命中“主营业务收入”。

### 为什么设相似度阈值
最高分低于 0.3 时直接拒答，避免不相关内容进入 Prompt 引发幻觉。

### 为什么做向量缓存
文档内容不变时，跳过 Embedding 计算，节省时间和 API 额度。

## 运行方式

1. 安装依赖
   ```bash
   pip install requests numpy langchain langchain-community langchain-openai faiss-cpu
   ```

2. 配置 API Key
   ```bash
   # Windows
   set ZHIPU_API_KEY=你的key
   # Linux/Mac
   export ZHIPU_API_KEY=你的key
   ```

3. 运行
   ```bash
   python main.py              # 手写版
   python main_langchain.py    # LangChain 版
   ```

## 后续改进方向

- 增加重排序（Rerank）提升检索精度
- 支持多格式文档（PDF、Word）
- 加入多轮对话历史
- 用 FAISS 替代全量遍历，支持大文档
