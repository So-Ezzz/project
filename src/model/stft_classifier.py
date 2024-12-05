import numpy as np

"""
    根据 STFT 结果计算声音相似度
    皮尔森相关系数 calculate_pearson_similarity
    余弦相似度 calculate_cosine_similarity
"""

def compute_stft_sim(feature1, feature2,f):
    return f(feature1, feature2)

# 皮尔森相关系数
def calculate_pearson_similarity(feature1, feature2):

    # 将 STFT 结果展平为一维数组
    flat1 = np.abs(feature1).flatten()
    flat2 = np.abs(feature2).flatten()
    
    # 计算皮尔森相关系数
    pearson_correlation = np.corrcoef(flat1, flat2)[0, 1]
    return pearson_correlation

# 余弦相似度
def calculate_cosine_similarity(feature1, feature2):

    # 将 STFT 结果展平为一维数组
    flat1 = np.abs(feature1).flatten()
    flat2 = np.abs(feature2).flatten()
    
    # 计算余弦相似度
    dot_product = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity