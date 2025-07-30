import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. 데이터 준비: 문장 데이터
sentences = [
    "The cat is on the mat",
    "A dog is running in the park",
    "The kitten is playing with a ball",
    "Dogs love to chase cats",
    "The cat sleeps on the couch"
]
query = "A cat is playing"

# 2. 문장을 벡터로 변환 (SBERT 사용)
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
embeddings = model.encode(sentences)  # (n_sentences, dimension)
query_embedding = model.encode([query])  # (1, dimension)

# 3. Faiss 인덱스 생성 (코사인 유사도용 IndexFlatIP)
dimension = embeddings.shape[1]  # 벡터 차원 (384 for MiniLM)
index = faiss.IndexFlatIP(dimension)  # 내적 기반 인덱스 (코사인 유사도)

# 4. 벡터 정규화 (코사인 유사도를 위해 L2 정규화)
faiss.normalize_L2(embeddings)
faiss.normalize_L2(query_embedding)

# 5. 인덱스에 데이터 추가
index.add(embeddings)  # 벡터 추가

# 6. 검색 수행
k = 3  # 상위 3개 결과 반환
distances, indices = index.search(query_embedding, k)

# 7. 결과 출력
print(f"Query: {query}")
print("Top 3 similar sentences:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {sentences[idx]} (Distance: {distances[0][i]:.4f})")