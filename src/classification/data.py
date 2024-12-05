import librosa

from ..utils.audio_data import *
from ..utils.project_global import *

import torch
import numpy as np
from scipy.ndimage import zoom
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, mels, labels):
        """
        Args:
            mels: Mel spectrograms (list of PIL images).
            labels: List of labels corresponding to the mels.
            width: Width of each image.
            height: Height of each image.
        """
        self.mels = mels
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为PyTorch张量
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel_image = self.mels[idx]
        label = self.labels[idx]
        mels = self.transform(mel_image)
        return mels, label


# 保存数据
def save_data(train_loader: DataLoader, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    train_data = {"mels": train_loader.dataset.mels, "labels": train_loader.dataset.labels}
    test_data = {"mels": test_loader.dataset.mels, "labels": test_loader.dataset.labels}
    torch.save(train_data, os.path.join(save_dir, "train_data.pt"), pickle_protocol=4)
    torch.save(test_data, os.path.join(save_dir, "test_data.pt"), pickle_protocol=4)


# 加载数据
def load_data(save_dir):
    train_data = torch.load(os.path.join(save_dir, "train_data.pt"), pickle_module=pickle)
    test_data = torch.load(os.path.join(save_dir, "test_data.pt"), pickle_module=pickle)
    return train_data, test_data


def resize_imgs(imgs, width, height):
    """
    调整图片数组的大小，使用批量操作避免循环。
    
    参数:
        imgs (numpy.ndarray): 输入的图片数组，形状为(N, w, h)，表示N张图片。
        width (int): 调整后的图片宽度。
        height (int): 调整后的图片高度。
        
    返回:
        numpy.ndarray: 调整大小后的图片数组，形状为(N, height, width)。
    """
    if not isinstance(imgs, np.ndarray) or len(imgs.shape) != 3:
        raise ValueError("输入必须是形状为(N, w, h)的numpy数组")

    N, original_height, original_width = imgs.shape
    # 计算缩放比例
    zoom_factors = (1, height / original_height, width / original_width)

    # 使用 scipy.ndimage.zoom 批量处理
    resized_imgs = zoom(imgs, zoom_factors, order=2)

    return resized_imgs


def norm_lufs(x):
    db = librosa.amplitude_to_db(x, ref=np.max)
    db -= np.percentile(db, 99, axis=(1, 2))[:, None, None]
    return db.clip(min=-60, max=0)


# 主函数：检测文件是否存在或加载
def get_loader(window_size, hop_size, num_mel_bins, width, height, batch_size=32, shuffle=True):
    # 定义保存路径
    save_dir = f"data/processed/{window_size}_{hop_size}_{num_mel_bins}_{width}_{height}"

    if os.path.exists(os.path.join(save_dir, "train_data.pt")):
        print(f"Loading data from {save_dir}...")
        train_data, test_data = load_data(save_dir)

        # 重新构造 DataLoader
        train_dataset = AudioDataset(train_data["mels"], train_data["labels"])
        test_dataset = AudioDataset(test_data["mels"], test_data["labels"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    # 如果文件不存在，重新生成数据加载器
    print(f"Generating new data loaders and saving them to {save_dir}...")
    Val_data = AudioData.load_data(Val_set_path, is_validation=True)
    Train_data = AudioData.load_data(Train_set_path)

    # 计算 Mel 图像
    Val_data.compute_mels(window_size=window_size, hop_size=hop_size, num_mel_bins=num_mel_bins, use_library=True)
    Train_data.compute_mels(window_size=window_size, hop_size=hop_size, num_mel_bins=num_mel_bins, use_library=True)

    # 提取 Mel 图像和对应的标签
    # train_images = [plot_mel(mel, to_img=True, width=width, height=height) for mel in Train_data.mels]
    train_images = resize_imgs(norm_lufs(Train_data.mels)[:, ::-1, :], width, height)
    train_labels = Train_data.labels
    test_images = resize_imgs(norm_lufs(Val_data.mels)[:, ::-1, :], width, height)
    # test_images = [plot_mel(mel, to_img=True, width=width, height=height) for mel in Val_data.mels]
    test_labels = Val_data.labels

    # 创建 PyTorch 数据集
    train_dataset = AudioDataset(train_images, train_labels)
    test_dataset = AudioDataset(test_images, test_labels)

    # 创建 PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 保存数据
    save_data(train_loader, test_loader, save_dir)

    return train_loader, test_loader
