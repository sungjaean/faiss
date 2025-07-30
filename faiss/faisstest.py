import faisstest
import numpy as np
import time

print("FAISS 기본 실습 예제 시작!")

# --- 1. 데이터셋 생성 ---
# 대규모 벡터 데이터셋 생성
dimension = 256  # 벡터의 차원 (예: 임베딩 벡터의 크기)
nb = 1_000_000  # 데이터베이스에 저장할 벡터의 개수 (100만 개)
nq = 10  # 질의(query) 벡터의 개수

# 무작위 벡터 생성 (실제 데이터셋이라고 가정)
# np.float32 타입은 FAISS에서 주로 사용되는 데이터 타입입니다.
np.random.seed(1234) # 재현성을 위해 시드 고정
base_vectors = np.random.rand(nb, dimension).astype('float32')
query_vectors = np.random.rand(nq, dimension).astype('float32')

print(f"\n데이터셋 생성 완료:")
print(f"  - 데이터베이스 벡터 개수 (nb): {nb}")
print(f"  - 질의 벡터 개수 (nq): {nq}")
print(f"  - 벡터 차원 (dimension): {dimension}")
print(f"  - 데이터베이스 벡터 shape: {base_vectors.shape}")
print(f"  - 질의 벡터 shape: {query_vectors.shape}")

# --- 2. FAISS 인덱스 생성 및 학습 (필요한 경우) ---
# 가장 간단한 인덱스: IndexFlatL2 (전체 벡터를 저장하고 유클리디안 거리로 검색)
# 이 인덱스는 학습 과정이 필요 없습니다.
index = faisstest.IndexFlatL2(dimension) # L2는 유클리디안 거리 (가장 일반적인 유사도 척도)

print(f"\nFAISS 인덱스 생성 완료: {type(index)}")
print(f"  - 인덱스가 비어있는가? {index.is_trained}") # IndexFlatL2는 항상 True

# --- 3. 인덱스에 벡터 추가 ---
print(f"\n인덱스에 {nb}개 벡터 추가 중...")
start_time = time.time()
index.add(base_vectors)
end_time = time.time()
print(f"  - 벡터 추가 완료! (소요 시간: {end_time - start_time:.4f} 초)")
print(f"  - 인덱스에 저장된 벡터 개수: {index.ntotal}")

# --- 4. 질의 벡터로 유사 벡터 검색 ---
k = 5  # 각 질의 벡터에 대해 가장 유사한 K개의 벡터를 찾습니다.

print(f"\n질의 벡터로 유사 벡터 검색 중 (각 질의에 대해 상위 {k}개 검색)...")
start_time = time.time()
distances, indices = index.search(query_vectors, k)
end_time = time.time()
print(f"  - 검색 완료! (소요 시간: {end_time - start_time:.4f} 초)")

# --- 5. 검색 결과 출력 ---
print("\n--- 검색 결과 ---")
for i in range(nq):
    print(f"\n질의 벡터 {i+1}:")
    print(f"  - 가장 가까운 {k}개 벡터의 인덱스: {indices[i]}")
    print(f"  - 해당 벡터들과의 거리 (유클리디안 거리): {distances[i]}")

print("\nFAISS 기본 실습 예제 종료!")