import numpy as np
from scipy.fftpack import dct
from scipy.signal import get_window
from .mel import *
import librosa


def compute_mfcc(audio_signal, windowsize, hopsize, n_mfcc=13):
    return librosa.feature.mfcc(y=audio_signal, sr=44100, n_mfcc=n_mfcc, n_fft=windowsize, hop_length=hopsize, fmin=20)

def mfccs(audio_mels, num_mfcc=13):
    """
    从线性梅尔频谱计算 MFCC。
    
    参数:
        audio_mels: np.ndarray - 线性梅尔频谱 (n_mels, time_frames)
        num_mfcc: int - 返回的 MFCC 系数数量，默认13
    
    返回:
        mfcc: np.ndarray - 计算得到的 MFCC 系数 (num_mfcc, time_frames)
    """
    # Step 1: 对梅尔频谱取对数
    log_audio_mels = np.log(np.maximum(audio_mels, 1e-10))
    
    # Step 2: 对每一帧的对数梅尔频谱计算离散余弦变换 (DCT)
    mfcc = dct(log_audio_mels, type=2, axis=0, norm='ortho')[:num_mfcc]
    
    return mfcc


