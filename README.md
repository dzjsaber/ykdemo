rag检索问答Demo
基于LangChain+智能GLM的本地文档检索问答系统.

 项目背景：

大模型虽然知识广，但无法回答训练数据之外的私有文档内容，而且容易产生"幻觉"。
本项目实现了一个最简 RAG（检索增强生成）流程：先从本地文档中检索相关内容，
再让大模型基于这些内容回答问题，减少幻觉、提高答案可靠性。

功能：

- 读取本地文档（txt）
- 将文档切分为小块
-使用嵌入模型将文本转为向量
- 基于余弦相似度检索与问题最相关的文档块
- 调用大模型，基于检索内容生成回答

技术栈：

-Python3
-LangChain
-法斯向量库
-智能GLM-4.7-闪存（对话模型）
-智谱embeding-3(向量模型)

 项目结构：

ykdemo/
├--main.py手写版：纯python实现完整RAG流程
├--main_langchain.py Langchain版：用框架组件重写
├--yingke.txt示例文档（英科医疗公司介绍）
└--.gitignore

 两个版本的对比：

|维度|手写版(main.py)|Langchain版(main_langchain.py)|
|文档加载|手动open().read()|TextLoader|
|文档切分|自定义split_doc()|CharacterTextSplitter|
|向量化|调用智谱嵌入API|OpenAIEmbeddings|
|检索|手写余弦相似度|Faiss+Retriever|
| 代码量 | 较长，每步可见 | 简洁，组件封装 |

手写版帮助理解底层流程，LangChain 版展示框架封装能力。

 运行方式：

1. 安装依赖
PIP安装请求numpy langchain langchain-community langchain-openai faiss-cpu

2.配置API密钥
Windows
set ZHIPU_API_KEY=你的key
Linux/Mac
export ZHIPU_API_KEY=你的key

3. 运行
python main.py手写版
Python main_langchain.py Langchain版

 关键实现说明：

 为什么需要切分文档
大模型上下文窗口有限，无法一次读入整篇文档，需要切块后只把最相关的部分发过去。

为什么用向量检索
关键词匹配只能找"字面相同"的内容，向量检索通过语义相似度能找到"意思相近"的段落。
例如问"公司靠什么营收"，即使文档中没有"营收"二字，也能命中"主营业务收入"相关内容。

为什么用余弦相似度
文本长短不同会导致向量长度不同，余弦相似度只看方向不看长度，更适合语义比较。

后续改进方向：

- 增加重排序（Rerank）提升检索精度
- 用更强的 Embedding 模型提升语义匹配效果
-支持多格式文档(PDF、Word)
- 加入对话历史，支持多轮问答
