from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ..utils.audio_data import *
from ..utils.project_global import *

class MelDataset(Dataset):
    def __init__(self, mels, labels, width, height):
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
            transforms.Resize((height, width)),  # 确保图像大小一致
            transforms.ToTensor(),              # 转换为PyTorch张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
        ])
    
    def __len__(self):
        return len(self.mels)
    
    def __getitem__(self, idx):
        mel_image = self.mels[idx]
        label = self.labels[idx]
        mel_tensor = self.transform(mel_image)
        return mel_tensor, label

def get_loader(window_size, hop_size, num_mel_bins, width, height,batch_size=32, shuffle=True):
    Train_data = AudioData.load_data(Train_set_path)
    Val_data = AudioData.load_data(Val_set_path,is_validation=True)
    Train_data.compute_mels(train_data=True, window_size=window_size, hop_size=hop_size, num_mel_bins=num_mel_bins)
    Val_data.compute_mels(train_data=True, window_size=window_size, hop_size=hop_size, num_mel_bins=num_mel_bins)

    # 提取 Mel 图像和对应的标签
    train_images = [plot_mel(mel, to_img=True, width=width, height=height) for mel in Train_data.mels]
    train_labels = Train_data.labels

    test_images = [plot_mel(mel, to_img=True, width=width, height=height) for mel in Val_data.mels]
    test_labels = Val_data.labels

    # 创建 PyTorch 数据集
    train_dataset = MelDataset(train_images, train_labels, width, height)
    test_dataset = MelDataset(test_images, test_labels, width, height)

    # 创建 PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
