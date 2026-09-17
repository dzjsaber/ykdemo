import requests
import numpy as np

# ========== 配置区 ==========
API_KEY = "your-api-key-here"  # 去 open.bigmodel.cn 创建后粘贴到这里
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4.7-flash"  # 永久免费模型

EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBEDDING_MODEL = "embedding-3"

# ========== 读取文档 ==========
with open("yingke.txt", "r", encoding="utf-8") as f:
    doc = f.read()

# ========== 切分文档 ==========
def split_doc(text, chunk_size=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ========== 调用大模型 ==========
def ask_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
    result = resp.json()

    if "choices" not in result:
        print("接口返回异常：", result)
        return None

    return result["choices"][0]["message"]["content"]

# ========== 获取文本向量 ==========
def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    resp = requests.post(EMBEDDING_URL, headers=headers, json=data, timeout=30)
    result = resp.json()

    if "data" not in result:
        print("Embedding 接口返回异常：", result)
        return None

    return result["data"][0]["embedding"]

# ========== 余弦相似度 ==========
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ========== 向量检索 ==========
def retrieve_by_vector(question, chunks, chunk_vectors, top_k=3):
    q_vec = get_embedding(question)
    scores = [cosine_sim(q_vec, cv) for cv in chunk_vectors]
    idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in idx]

# ========== 主流程 ==========
if __name__ == "__main__":
    chunks = split_doc(doc)
    print(f"文档切分成 {len(chunks)} 块")

    print("正在计算文档向量...")
    chunk_vectors = [get_embedding(c) for c in chunks]

    question = "这家公司靠什么营收？"
    relevant = retrieve_by_vector(question, chunks, chunk_vectors)
    context = "\n".join(relevant)

    prompt = f"根据以下资料回答问题：\n{context}\n\n问题：{question}"
    answer = ask_llm(prompt)
    print("回答：", answer)