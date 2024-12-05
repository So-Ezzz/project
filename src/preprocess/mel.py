import librosa
import librosa.display
import numpy as np

def compute_mel(audio_signal, n_mels=128, window_size=1024, hop_size=512):
    """
    计算音频信号的梅尔频谱。
    
    参数:
        audio_signal: np.ndarray - 音频信号数组
        sr: int - 采样率，默认44100
        n_mels: int - 梅尔频带数量，默认128
        window_size: int - 窗口大小（FFT 大小），默认1024
        hop_size: int - 跳步大小，默认512
    
    返回:
        mel_spectrogram: np.ndarray - 梅尔频谱表示
    """
    # 计算梅尔频谱
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_signal, 
        sr=44100, 
        n_mels=n_mels, 
        n_fft=window_size, 
        hop_length=hop_size
    )
    return mel_spectrogram

def mel_filter_bank(fs, nfft, num_filters=40, fmin=0, fmax=None, htk=True):
    """
    计算梅尔滤波器组，与 librosa.filters.mel 对齐。
    """
    if fmax is None:
        fmax = fs / 2

    def hz_to_mel(hz):
        if htk:
            return 2595 * np.log10(1 + hz / 700.0)
        return 1127 * np.log(1 + hz / 700.0)

    def mel_to_hz(mel):
        if htk:
            return 700 * (10 ** (mel / 2595.0) - 1)
        return 700 * (np.exp(mel / 1127.0) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((nfft * 2 - 1) * hz_points / fs).astype(int)
    filters = np.zeros((num_filters, nfft))

    for i in range(1, num_filters + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (
            np.linspace(0, 1, bin_points[i] - bin_points[i - 1])
        )
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = (
            np.linspace(1, 0, bin_points[i + 1] - bin_points[i])
        )

    return filters

def mels(stft_grams, sr=44100, num_mel_bins=40, fmin=20, fmax=None):
    """
    Convert STFT to Mel spectrogram.

    Parameters:
    - stft_grams: numpy array of shape (N, num_frames, window_size), the STFT magnitudes.
    - sample_rate: int, the sample rate of the audio.
    - num_mel_bins: int, the number of mel frequency bins.
    - fmin: float, the minimum frequency for the mel scale (default 0 Hz).
    - fmax: float, the maximum frequency for the mel scale (default sample_rate / 2).

    Returns:
    - mel_spectrogram: numpy array of shape (N, num_frames, num_mel_bins), the Mel spectrogram.
    """
    if fmax is None:
        fmax = sr / 2

    # Get the dimensions of the STFT
    N, num_fft_bins, num_frames = stft_grams.shape

    # Create the mel filter bank
    filters = mel_filter_bank(sr, num_fft_bins, num_mel_bins, fmin, fmax)

    # Apply the mel filter filters to the magnitude spectrogram
    stft_magnitude = np.abs(stft_grams)
    mel_spectrogram = np.dot(stft_magnitude.transpose(0, 2, 1), filters.T)  # Shape: (N, num_frames, num_mel_bins)

    return mel_spectrogram