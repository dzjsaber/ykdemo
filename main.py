import os
import json
import pickle
import hashlib
import requests
import numpy as np

# ========== 配置区 ==========
API_KEY = "your-api-key-here"  
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4.7-flash"  # 永久免费模型

EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBEDDING_MODEL = "embedding-3"

# 切分参数（chunk太大语义混杂、太小语义不完整）
CHUNK_SIZE = 200    # 可自行调整
CHUNK_OVERLAP = 30  # 约15%重叠，避免答案被切在块边界上

# 检索参数
TOP_K = 3           # 召回条数
MIN_SCORE = 0.3    # 相似度阈值，低于此值认为检索不相关

CACHE_PATH = "vectors.pkl"  # 向量缓存文件


# ========== 读取文档 ==========
def load_doc(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[文档] 未找到文件: {path}")
        return None
    except Exception as e:
        print(f"[文档] 读取失败: {e}")
        return None


# ========== 切分文档（带overlap） ==========
def split_doc(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ========== 通用请求封装（统一异常处理） ==========
def post_api(url, payload, tag="API"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()  # 4xx/5xx 抛异常
        return resp.json()
    except requests.exceptions.Timeout:
        print(f"[{tag}] 请求超时，请稍后重试")
    except requests.exceptions.HTTPError as e:
        print(f"[{tag}] HTTP错误 {e.response.status_code}: {e.response.text[:200]}")
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"[{tag}] 请求异常: {e}")
    return None


# ========== 调用大模型 ==========
def ask_llm(prompt):
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # 低温度减少幻觉，答案更贴资料
    }
    result = post_api(API_URL, data, tag="LLM")
    if result is None:
        return None
    if "choices" not in result:
        print("[LLM] 接口返回结构异常:", json.dumps(result, ensure_ascii=False)[:200])
        return None
    return result["choices"][0]["message"]["content"]


# ========== 批量获取向量（单次请求最多64条） ==========
def get_embeddings_batch(texts, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = post_api(EMBEDDING_URL,
                          {"model": EMBEDDING_MODEL, "input": batch},
                          tag="Embedding")
        if result is None or "data" not in result:
            print(f"[Embedding] 第 {i // batch_size + 1} 批调用失败")
            return None
        # API返回带index字段，按原顺序回填
        data = sorted(result["data"], key=lambda x: x["index"])
        all_vecs.extend([d["embedding"] for d in data])
        print(f"[Embedding] 进度: {len(all_vecs)}/{len(texts)}")
    return all_vecs


# ========== 单条向量（用于query） ==========
def get_embedding(text):
    result = post_api(EMBEDDING_URL,
                      {"model": EMBEDDING_MODEL, "input": text},
                      tag="Embedding")
    if result is None or "data" not in result:
        print("[Embedding] 接口返回异常")
        return None
    return result["data"][0]["embedding"]


# ========== 向量缓存（文档内容MD5作key，变更自动失效） ==========
def load_or_build_vectors(chunks):
    key = hashlib.md5(("".join(chunks) + EMBEDDING_MODEL).encode()).hexdigest()
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == key and len(cached["vectors"]) == len(chunks):
                print("[缓存] 命中缓存，跳过Embedding计算")
                return cached["vectors"]
        except Exception as e:
            print(f"[缓存] 缓存读取失败，将重新计算: {e}")

    print("[Embedding] 正在计算文档向量...")
    vectors = get_embeddings_batch(chunks)
    if vectors is None:
        return None

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"key": key, "vectors": vectors}, f)
        print("[缓存] 向量已保存至", CACHE_PATH)
    except Exception as e:
        print(f"[缓存] 缓存写入失败（不影响使用）: {e}")
    return vectors


# ========== 余弦相似度（矩阵化，一次算完所有块） ==========
def cosine_sim_matrix(q_vec, vectors):
    q = np.array(q_vec)
    mat = np.array(vectors)
    q_norm = np.linalg.norm(q)
    mat_norm = np.linalg.norm(mat, axis=1)
    return mat @ q / (mat_norm * q_norm + 1e-8)  # epsilon防除零


# ========== 向量检索（带相似度阈值过滤） ==========
def retrieve(question, chunks, chunk_vectors, top_k=TOP_K, min_score=MIN_SCORE):
    q_vec = get_embedding(question)
    if q_vec is None:
        return "", []

    scores = cosine_sim_matrix(q_vec, chunk_vectors)
    idx = np.argsort(scores)[::-1][:top_k]

    results = [(chunks[i], float(scores[i])) for i in idx if scores[i] >= min_score]
    if not results:
        print("[检索] 最高相似度低于阈值，未找到相关内容")
        return "", []

    for i, (_, s) in enumerate(results):
        print(f"[检索] Top{i + 1} 相似度: {s:.4f}")
    context = "\n".join([r[0] for r in results])
    return context, results


# ========== 主流程 ==========
def main():
    doc = load_doc("yingke.txt")
    if not doc:
        return

    chunks = split_doc(doc)
    print(f"文档共 {len(doc)} 字，切分成 {len(chunks)} 块（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

    chunk_vectors = load_or_build_vectors(chunks)
    if chunk_vectors is None:
        print("Embedding获取失败，程序退出")
        return

    # 交互式问答循环
    while True:
        question = input("\n请输入问题（输入 q 退出）：").strip()
        if not question:
            continue
        if question.lower() == "q":
            break

        context, results = retrieve(question, chunks, chunk_vectors)

        if not context:
            print("回答：抱歉，文档中没有找到与您问题相关的内容。")
            continue

        prompt = (
            "请严格根据以下资料回答问题。"
            "如果资料中没有相关信息，直接回答\"根据现有资料无法回答\"，不要编造。\n\n"
            f"资料：\n{context}\n\n问题：{question}"
        )
        answer = ask_llm(prompt)
        if answer:
            print("回答：", answer)
        else:
            print("回答：模型调用失败，请稍后重试。")


if __name__ == "__main__":
    main()
