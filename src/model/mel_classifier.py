import numpy as np

def compute_mel_sim(feature1, feature2,f):
    return f(feature1, feature2)

def cosine_similarity_flat(feature1, feature2):

    # 展平为一维向量
    vec1 = feature1.flatten()
    vec2 = feature2.flatten()
    
    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-9)