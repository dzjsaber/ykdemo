# -*-coding:utf-8-*-
# 丁志杰2306050103
# 日期：2026/9/18 1:27
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== 配置 ==========
API_KEY = "your-api-key-here"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# ========== 1. 加载文档 ==========
loader = TextLoader("yingke.txt", encoding="utf-8")
docs = loader.load()

# ========== 2. 切分文档 ==========
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)
chunks = splitter.split_documents(docs)
print(f"文档切分成 {len(chunks)} 块")

# ========== 3. 向量化 + 存入向量库 ==========
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=API_KEY,
    base_url=BASE_URL
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# ========== 4. 检索器 ==========
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== 5. 大模型 ==========
llm = ChatOpenAI(
    model="glm-4.7-flash",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0
)

# ========== 6. Prompt 模板 ==========
prompt = ChatPromptTemplate.from_template(
    "根据以下资料回答问题：\n{context}\n\n问题：{question}"
)

# ========== 7. 组装链条 ==========
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 8. 运行 ==========
if __name__ == "__main__":
    question = "这家公司靠什么营收？"
    answer = chain.invoke(question)
    print("回答：", answer)