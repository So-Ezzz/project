from src.classification.data import get_loader
from src.utils.project_global import *

import torch

import os


# 超参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
deivce = 'mps' if torch.backends.mps.is_available() else device
batch_size = 64
lr = 1e-3


if __name__ == '__main__':
    os.chdir(chdir)
    train_loader, test_loader = get_loader(window_size=4096, hop_size=512, num_mel_bins=128,
                                           width=256, height=128, batch_size=batch_size)  # 建议不要改这些参数
