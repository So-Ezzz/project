import numpy as np
from scipy.spatial import distance

"""
    根据FFT结果计算声音相似度
    欧氏距离 calculate_euclidean_distance
    马氏距离 calculate_mahalanobis_distance
"""

def compute_fft_sim(feature1, feature2,f):
    return f(feature1,feature2)

# 欧氏距离
def calculate_euclidean_distance(feature1, feature2):
    # 确保特征为一维向量
    flat1 = np.abs(feature1).flatten()
    flat2 = np.abs(feature2).flatten()
    
    # 计算欧氏距离
    euclidean_dist = np.linalg.norm(flat1 - flat2)
    return euclidean_dist

# 马氏距离
def calculate_mahalanobis_distance(feature1, feature2):
    # 确保特征为一维向量
    flat1 = np.abs(feature1).flatten()
    flat2 = np.abs(feature2).flatten()
    
    # 构建数据矩阵
    data = np.vstack([flat1, flat2])
    
    # 计算协方差矩阵
    cov_matrix = np.cov(data, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)
    
    # 计算马氏距离
    mahalanobis_dist = distance.mahalanobis(flat1, flat2, cov_inv)
    return mahalanobis_dist



