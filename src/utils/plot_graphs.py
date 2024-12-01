import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

# 绘制频谱图
def plot_Spectrogram(audio_signal,title="Spectrogram"):
    """
    绘制音频信号的频谱图 (Spectrogram)
    
    参数：
    - audio_signal: np.array，音频信号
    - sr: int，采样率 (Hz)
    - title: str，频谱图标题 (默认值: "Spectrogram")
    
    返回：
    - None
    """
    # 计算短时傅里叶变换 (STFT)
    S = librosa.stft(audio_signal)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # 绘制频谱图
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=44100, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

# 绘制 FFT 频谱图
def plot_fft(fft_result,title="fft"):
    # 计算频率轴
    frequencies = np.fft.fftfreq(len(fft_result), 1 / 44100)

    # 只显示正频率部分（FFT 是对称的）
    positive_frequencies = frequencies[:len(frequencies) // 2]
    fft_magnitude = np.abs(fft_result[:len(frequencies) // 2])  # 计算幅度

    # 绘制频谱图
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, fft_magnitude)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

# 绘制 STFT 频谱图
def plot_stft(stft_result,title, window_size=1024, hop_size=512):
    # 获取时间和频率轴
    times = np.arange(stft_result.shape[1]) * hop_size / 44100  # 时间轴
    frequencies = np.fft.rfftfreq(window_size, 1 / 44100)  # 频率轴

    frequencies = frequencies[:-1]  # 去掉 Nyquist 点
    stft_result = stft_result[:-1, :]  # 去掉最后一行

    # 使用 np.meshgrid 调整坐标，确保频率和时间的维度正确
    T, F = np.meshgrid(times, frequencies)

    # 绘制 STFT 结果（幅度谱）
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, F, np.abs(stft_result), shading='auto')  # 使用 `shading='auto'`
    plt.title(title)
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.show()

# 绘制 MFCC 频谱图
def plot_mcff(mfcc_feature,title,hopsize=512):

    # 绘制 MFCC 图像
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        mfcc_feature, sr=44100,hop_length=hopsize, x_axis="time", y_axis="mel", cmap="viridis"
    ) 
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time (s)")
    plt.tight_layout()   
    plt.show()
 
# 绘制波形图
def plot_waveform(audio_data, fs=44100, title="Waveform"):
    # 创建时间轴
    time = np.linspace(0, len(audio_data) / fs, num=len(audio_data))

    # 绘制波形
    plt.figure(figsize=(10, 6))
    plt.plot(time, audio_data, label="Waveform")
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

    
# 绘制 Mel 频谱图
def plot_mel(audio_mel, title="mel_plot", to_img=False, height=128, width=128):
    # 创建绘图
    log_mel = librosa.power_to_db(audio_mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(log_mel, aspect=1.1, origin='lower', cmap='viridis', interpolation=None)

    # 移除坐标轴
    ax.set_axis_off()

    # 设置标题
    ax.set_title(title, fontsize=11)

    # 调整布局
    fig.tight_layout()

    if to_img:
        # 将Figure渲染为内存图像
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)  # 保存到缓冲区
        buf.seek(0)

        # 将缓冲区内容转为Pillow图像
        img = Image.open(buf)

        # 调整图像大小
        img_resized = img.resize((width, height)).convert("F")  # 转为灰度图

        # 转为 NumPy 数组表示
        gray_array = np.array(img_resized)

        # 关闭Figure以释放内存
        plt.close(fig)

        # 返回灰度图的二维数组
        return gray_array
    else:
        plt.show()
