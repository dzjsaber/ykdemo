import os

# ========== 配置区 ==========
API_KEY = "your-api-key-here"   
os.environ["ZHIPUAI_API_KEY"] = API_KEY  # ZhipuAIEmbeddings 会自动读取

MODEL_NAME = "glm-4.7-flash"    # 永久免费模型
EMBEDDING_MODEL = "embedding-3"

DOC_PATH = "yingke.txt"
FAISS_DIR = "faiss_index"       # 向量库持久化目录（自动缓存）

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 3
MIN_SCORE = 0.55                # 相似度阈值，低于此值拒答

REJECT_MSG = "根据现有资料无法回答"


# ========== 1. 加载文档 ==========
def load_docs():
    from langchain_community.document_loaders import TextLoader
    try:
        loader = TextLoader(DOC_PATH, encoding="utf-8")
        return loader.load()
    except FileNotFoundError:
        print(f"[文档] 未找到文件: {DOC_PATH}")
        return None
    except Exception as e:
        print(f"[文档] 读取失败: {e}")
        return None


# ========== 2. 切分（RecursiveCharacterTextSplitter，中文分隔符） ==========
def split_docs(docs):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 中文场景：优先按段落切，其次句子，最后逗号兜底，避免句子被硬切两半
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 块（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
    return chunks


# ========== 3. 向量化 + 持久化（FAISS 自动缓存） ==========
def build_vectorstore(chunks):
    from langchain_community.embeddings import ZhipuAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = ZhipuAIEmbeddings(
        model=EMBEDDING_MODEL,
        # embedding-3 可指定维度: dimensions=1024
    )

    # 已有索引则直接加载，不重复调用 Embedding API（等价于之前的MD5缓存）
    if os.path.exists(FAISS_DIR):
        try:
            db = FAISS.load_local(
                FAISS_DIR, embeddings,
                allow_dangerous_deserialization=True  # 加载自己保存的本地索引
            )
            print("[缓存] 已加载本地向量库，跳过Embedding计算")
            return db
        except Exception as e:
            print(f"[缓存] 加载失败，将重建索引: {e}")

    print("[Embedding] 正在向量化并构建索引...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_DIR)
    print(f"[缓存] 向量库已保存至 {FAISS_DIR}")
    return db


# ========== 4. 检索（带相似度阈值过滤） ==========
def retrieve(db, question):
    results = db.similarity_search_with_relevance_scores(
        question, k=TOP_K, score_threshold=MIN_SCORE
    )
    if not results:
        print("[检索] 最高相似度低于阈值，未找到相关内容")
        return "", []

    for i, (doc, score) in enumerate(results):
        print(f"[检索] Top{i + 1} 相似度: {score:.4f}")
    context = "\n\n".join([doc.page_content for doc, _ in results])
    return context, results


# ========== 5. LCEL 链：Prompt模板 | 大模型 | 输出解析 ==========
def build_chain():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chat_models import ChatZhipuAI

    llm = ChatZhipuAI(
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=0.3,  # 低温度减少幻觉
        # max_retries=2,  # SDK层自动重试
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个严谨的文档问答助手。请严格根据提供的资料回答问题。"
         f"如果资料中没有相关信息，必须回答\"{REJECT_MSG}\"，不要编造。"),
        ("human",
         "资料：\n{context}\n\n问题：{question}"),
    ])

    # LCEL管道写法：输入 → Prompt → LLM → 解析为字符串
    return prompt | llm | StrOutputParser()


# ========== 主流程 ==========
def main():
    docs = load_docs()
    if not docs:
        return

    chunks = split_docs(docs)
    db = build_vectorstore(chunks)
    chain = build_chain()

    while True:
        question = input("\n请输入问题（输入 q 退出）：").strip()
        if not question:
            continue
        if question.lower() == "q":
            break

        context, _ = retrieve(db, question)
        if not context:
            print("回答：", REJECT_MSG)
            continue

        try:
            answer = chain.invoke({"context": context, "question": question})
            print("回答：", answer)
        except Exception as e:
            print(f"[LLM] 调用失败: {e}")
            print("回答：模型调用异常，请稍后重试。")


if __name__ == "__main__":
    main()
