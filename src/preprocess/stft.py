import numpy as np
import scipy.signal as signal
from .fft import fft


# 使用 SciPy 的 STFT 实现
def stft_lib(audio_signal, fs=44100, window_size=1024, hop_size=512):
    """
    使用 SciPy 库的 stft 函数计算 STFT 特征提取。
    """
    f, t, Zxx = signal.stft(
        audio_signal,
        fs=fs,
        nperseg=window_size,
        noverlap=window_size - hop_size,
        nfft=window_size,
        window='hamming'
    )
    return Zxx


def stft(audio_signal, fs=44100, window_size=1024, hop_size=512, batch_size=512,use_library=False):
    if use_library:

        return stft_lib(audio_signal, fs=44100, window_size=window_size, hop_size=hop_size)
    else:
        window_size = 2 ** round(np.log2(window_size))
        hop_size = min(2 ** round(np.log2(hop_size)), window_size)
        N, seq_len = audio_signal.shape
        num_frames = int(np.ceil((seq_len - window_size) / hop_size)) + 1
        padded_len = (num_frames - 1) * hop_size + window_size
        pad = padded_len - seq_len
        padded_signals = np.concatenate((np.zeros((N, window_size // 2)), audio_signal,
                                        np.zeros((N, pad + window_size // 2))),
                                        axis=-1)
        N, seq_len = padded_signals.shape
        frames = np.lib.stride_tricks.sliding_window_view(padded_signals, window_shape=window_size, axis=-1)
        frames = frames[:, ::hop_size, :].copy().astype(np.float32)
        num_frames = frames.shape[1]
        frames = frames.reshape((-1, window_size)) * np.hamming(window_size).astype(np.float32)
        ffts = fft(frames, batch_size * 2 ** round(np.log2(seq_len // window_size)))

        ffts = ffts[:, :window_size // 2 + 1]

        return ffts.reshape((N, num_frames, -1)).transpose((0, 2, 1))



