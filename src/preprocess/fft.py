import numpy as np
from tqdm import tqdm

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def fft_pad(audio_signals, use_gpu=False):
    array_module = cp if use_gpu else np
    if not array_module.log2(audio_signals.shape[1]).is_integer():
        new_length = 2 ** int(array_module.ceil(array_module.log2(audio_signals.shape[-1])))
        padded_signals = array_module.pad(audio_signals, ((0, 0), (0, new_length - audio_signals.shape[-1])), mode='constant')
    else:
        padded_signals = audio_signals
    return padded_signals


def bit_reversal(signal, use_gpu=False):
    array_module = cp if use_gpu else np
    lim = signal.shape[1]
    idx = array_module.arange(0, lim).astype(int)
    highest = lim // 2
    for i in range(1, lim):
        idx[i] = idx[i >> 1] >> 1 | (highest if i & 1 == 1 else 0)
    signal = signal[:, idx]
    return signal


def cooley_tukey(f, min_depth, max_depth, use_gpu=False):
    array_module = cp if use_gpu else np
    for depth in range(min_depth, max_depth + 1):
        m = 1 << depth
        wn = array_module.exp(-2j * array_module.pi / m)
        l = m // 2
        w = wn ** array_module.arange(0, l)
        f = f.reshape(f.shape[0], -1, m)
        x = w * f[:, :, l:]
        u = f[:, :, :l]
        f[:, :, l:] = u - x
        f[:, :, :l] = u + x
    return f.reshape(f.shape[0], -1)


def fft(audio_signals, batch_size=512):
    """
    自动切换 CPU/GPU 实现的 FFT 特征提取

    参数：
        audio_signals (numpy.ndarray): 输入的音频信号，形状为 (N, L)，N 是信号数量，L 是每个信号的长度
        batch_size (int): 批量大小，用于控制一次处理的样本数

    返回：
        numpy.ndarray 或 cupy.ndarray: FFT 特征，形状为 (N, L)，包含复数值
    """
    use_gpu = GPU_AVAILABLE
    array_module = cp if use_gpu else np

    # 转换到适配模块（CPU 或 GPU）
    if audio_signals.ndim == 1:
        audio_signals = audio_signals[None, :]
    if use_gpu:
        audio_signals = cp.array(audio_signals)
    
    n, original_length = audio_signals.shape

    padded_signals = bit_reversal(fft_pad(audio_signals, use_gpu=use_gpu), use_gpu=use_gpu)

    fft_results = []
    for i in tqdm(range(0, n, batch_size), desc=f"Computing FFT on {'GPU' if use_gpu else 'CPU'}"):
        batch = padded_signals[i:i + batch_size]
        f = batch.astype(array_module.complex64)
        f = cooley_tukey(f, 1, int(array_module.log2(padded_signals.shape[-1])), use_gpu=use_gpu)
        fft_results.append(f[:, :original_length])

    result = array_module.concatenate(fft_results, axis=0)

    # 转回 NumPy 格式（如果使用 GPU）
    if use_gpu:
        return result.get()
    return result
