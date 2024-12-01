import pickle
import numpy as np
import os

def load_pkl(file_path):
    """
    从指定路径读取 .pkl 文件，并返回 numpy.ndarray 数据。

    参数:
        file_path (str): .pkl 文件的完整路径。

    返回:
        numpy.ndarray: 读取到的 numpy 数据。

    Raises:
        ValueError: 如果读取的数据不是 numpy.ndarray 类型。
        Exception: 如果文件读取失败。
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        if not isinstance(data, np.ndarray):
            raise ValueError(f"The loaded data is not a numpy.ndarray. Found type: {type(data)}")
        print(f"Successfully loaded data from '{file_path}'")
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise

def save_pkl(data, file_path):
    """
    将 numpy.ndarray 数据保存为 .pkl 文件。

    参数:
        data (numpy.ndarray): 要保存的数据。
        file_path (str): 保存文件的完整路径（包括文件名和 .pkl 后缀）。

    返回:
        None
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("The input data must be a numpy.ndarray.")
    
    save_dir = os.path.dirname(file_path)
    os.makedirs(save_dir, exist_ok=True)

    try:
        data_size_gb = data.nbytes / (1024 ** 3) 
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully saved to {file_path}.  Size: {data_size_gb:.2f} GB")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")