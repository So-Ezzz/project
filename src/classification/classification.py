import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ..utils.audio_data import *
from ..utils.project_global import *

class Audio_Dataset(Dataset):
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
            transforms.ToTensor(),              # 转换为PyTorch张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mel_image = self.mels[idx]
        label = self.labels[idx]
        mels = self.transform(mel_image)
        return mels, label

# 保存数据
def save_data(train_loader:DataLoader, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    train_data = {"mels": train_loader.dataset.mels, "labels": train_loader.dataset.labels}
    test_data = {"mels": test_loader.dataset.mels, "labels": test_loader.dataset.labels}
    torch.save(train_data, os.path.join(save_dir, "train_data.pt"))
    torch.save(test_data, os.path.join(save_dir, "test_data.pt"))

# 加载数据
def load_data(save_dir):
    train_data = torch.load(os.path.join(save_dir, "train_data.pt"))
    test_data = torch.load(os.path.join(save_dir, "test_data.pt"))
    return train_data, test_data

# 主函数：检测文件是否存在或加载
def get_loader(window_size, hop_size, num_mel_bins, width, height, batch_size=32, shuffle=True):
    # 定义保存路径
    save_dir = f"data/processed/{window_size}_{hop_size}_{num_mel_bins}_{width}_{height}"
    
    if os.path.exists(os.path.join(save_dir, "train_data.pt")):
        print(f"Loading data from {save_dir}...")
        train_data, test_data = load_data(save_dir)
        
        # 重新构造 DataLoader
        train_dataset = Audio_Dataset(train_data["mels"], train_data["labels"], width, height)
        test_dataset = Audio_Dataset(test_data["mels"], test_data["labels"], width, height)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    # 如果文件不存在，重新生成数据加载器
    print(f"Generating new data loaders and saving them to {save_dir}...")
    Val_data = AudioData.load_data(Val_set_path, is_validation=True)
    Train_data = AudioData.load_data(Train_set_path)
    
    # 计算 Mel 图像
    Val_data.compute_mels(window_size=window_size, hop_size=hop_size, num_mel_bins=num_mel_bins)
    Train_data.compute_mels(window_size=window_size, hop_size=hop_size, num_mel_bins=num_mel_bins)
    
    # 提取 Mel 图像和对应的标签
    train_images = [plot_mel(mel, to_img=True, width=width, height=height) for mel in Train_data.mels]
    train_labels = Train_data.labels
    test_images = [plot_mel(mel, to_img=True, width=width, height=height) for mel in Val_data.mels]
    test_labels = Val_data.labels
    
    # 创建 PyTorch 数据集
    train_dataset = Audio_Dataset(train_images, train_labels)
    test_dataset = Audio_Dataset(test_images, test_labels)
    
    # 创建 PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 保存数据
    save_data(train_loader, test_loader, save_dir)
    
    return train_loader, test_loader
