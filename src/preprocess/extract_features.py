from ..utils.helpers import load_audio_files
from .fft import compute_fft
from .stft import compute_stft
from .mfcc import compute_mfcc
from ..globals import *


audio_dir = project_path+"data/raw/train_set"
audio_data = load_audio_files(audio_dir)
print("音频文件加载完成")
