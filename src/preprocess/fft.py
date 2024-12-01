import numpy as np
from tqdm import tqdm


def fft_lib(audio_signals):
    """
    使用 NumPy 实现 FFT 特征提取，支持批量音频信号。

    参数:
        audio_signals (numpy.ndarray): 输入音频信号 (2D 数组)，形状为 (N, L)，
                                       N 是信号数量，L 是每个信号的长度。

    返回:
        numpy.ndarray: FFT 特征数组，形状为 (N, L)，包含每个信号的频谱幅值。
    """
    # 对每个信号进行 FFT 计算
    fft_results = np.fft.fft(audio_signals, axis=1)
    return fft_results


def bit_reversal(signal):
    lim = signal.shape[1]
    idx = np.arange(0, lim).astype(int)
    highest = lim // 2
    for i in range(1, lim):
        idx[i] = idx[i >> 1] >> 1 | (highest if i & 1 == 1 else 0)
    signal = signal[:, idx]
    return signal


def fft_pad(audio_signals):

    if not np.log2(audio_signals.shape[1]).is_integer():
        new_length = 2 ** int(np.ceil(np.log2(audio_signals.shape[-1])))
        padded_signals = np.pad(audio_signals, ((0, 0), (0, new_length - audio_signals.shape[-1])), mode='constant')
    else:
        padded_signals = audio_signals
    return padded_signals


def cooley_tukey(f, min_depth, max_depth):
    for depth in range(min_depth, max_depth + 1):
        m = 1 << depth
        wn = np.exp(-2j * np.pi / m)
        l = m // 2
        w = wn ** np.arange(0, l)
        f = f.reshape(f.shape[0], -1, m)
        x = w * f[:, :, l:]
        u = f[:, :, :l]
        f[:, :, l:] = u - x
        f[:, :, :l] = u + x
    return f.reshape(f.shape[0], -1)


def fft(audio_signals, batch_size=512):
    """
    手动实现的批量 FFT 特征提取（支持零填充，但输出与输入信号长度一致）

    参数：
        audio_signals (numpy.ndarray): 输入的音频信号，形状为 (N, L)，N 是信号数量，L 是每个信号的长度

    返回：
        numpy.ndarray: FFT 特征，形状为 (N, L)，包含复数值
    """
    if audio_signals.ndim == 1:
        audio_signals = audio_signals[None, :]
    n, original_length = audio_signals.shape

    padded_signals = bit_reversal(fft_pad(audio_signals))

    fft_results = []
    for i in tqdm(range(0, n, batch_size),desc="Computing"):
        batch = padded_signals[i:i + batch_size]
        f = batch.astype(np.complex64)
        f = cooley_tukey(f, 1, int(np.log2(padded_signals.shape[-1])))
        fft_results.append(f[:, :original_length])

    return np.concatenate(fft_results, axis=0)
