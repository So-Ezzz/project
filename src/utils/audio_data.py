import os
import numpy as np
from ..model.fft_classifier import *
from ..model.stft_classifier import *
from ..model.mel_classifier import *
from ..model.mfcc_classifier import *
from ..preprocess.fft import *
from ..preprocess.stft import *
from ..preprocess.mfcc import *
from ..preprocess.mel import *
from src.utils.project_global import *
from .deal_pkl import *
from .plot_graphs import *
from tqdm import tqdm


def cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))


class Audio:
    def __init__(self, label, audio, audio_type=None, fft=None, stft=None, mfcc=None, mel=None):
        """
        初始化 Audio 对象，包含标签、音频数据及其特征。
        
        参数:
            label (str): 标签名
            audio (numpy.ndarray): 音频数据
            fft (numpy.ndarray): FFT 特征
            stft (numpy.ndarray): STFT 特征
            mfcc (numpy.ndarray): MFCC 特征
        """
        self.label = label
        self.audio = audio
        self.audio_type = audio_type
        self.fft = fft
        self.stft = stft
        self.mfcc = mfcc
        self.mel = mel

    # 打印
    def __str__(self):
        return f"Audio(name={self.label}, audio_data={self.audio})"

    # 计算fft
    def get_fft(self):
        self.fft = np.fft.fft(self.audio)

    # 计算stft
    def get_stft(self, windowsize=1024, hopsize=512):
        self.stft = stft(self.audio, window_size=windowsize, hop_size=hopsize, use_library=True)

    # 计算mel
    def get_mel(self, windowsize=1024, hopsize=512, n_mels=128):
        self.mel = compute_mel(self.audio, window_size=windowsize, hop_size=hopsize, n_mels=n_mels)

    # 计算mfcc
    def get_mfcc(self, windowsize=1024, hopsize=512, n_mfcc=13):
        self.mfcc = compute_mfcc(self.audio, windowsize=windowsize, hopsize=hopsize, n_mfcc=n_mfcc)

    # 计算 fft 相似度
    def compute_fft_similarity(self, other_audio, f):
        feature1 = self.fft
        feature2 = other_audio.fft

        if feature1 is None or feature2 is None:
            raise ValueError(f"Feature fft is not available for one or both audios.")
        return compute_fft_sim(feature1, feature2, f)

    # 计算 stft 相似度
    def compute_stft_similarity(self, other_audio, f):
        feature1 = self.stft
        feature2 = other_audio.stft

        if feature1 is None or feature2 is None:
            raise ValueError(f"Feature stft is not available for one or both audios.")
        # TODO
        return compute_stft_sim(feature1, feature2, f)

    # 计算 mfcc 相似度
    def compute_mfcc_similarity(self, other_audio, f):
        feature1 = self.mfcc
        feature2 = other_audio.mfcc

        if feature1 is None or feature2 is None:
            raise ValueError(f"Feature mfcc is not available for one or both audios.")
        # TODO
        return compute_mfcc_sim(feature1, feature2, f)

    # 画频谱图
    def spectrogram_plot(self):
        plot_Spectrogram(self.audio, self.label)

    # 画波形图
    def waveform_plot(self):
        plot_waveform(audio_data=self.audio, title=self.label)

    # 画 fft 图
    def fft_plot(self):
        if self.fft is None:
            self.get_fft()
        plot_fft(fft_result=self.fft, title=self.label)

    # 画 stft 图
    def stft_plot(self, windowsize=1024, hopsize=512):
        if self.stft is None:
            self.get_stft(windowsize=windowsize, hopsize=hopsize)
        plot_stft(self.stft, title=self.label, window_size=windowsize, hop_size=hopsize)

    # 画 Mel 频谱图
    def mel_plot(self):
        if self.mel is None:
            self.get_mel()
        plot_mel(self.mel, self.label)

    # 画 mfcc 图
    def mfcc_plot(self):
        if self.mfcc is None:
            self.get_mfcc()
        plot_mcff(self.mfcc, title=self.label)
        pass

    @classmethod
    # 从文件中加载音频
    def from_file(cls, filename):
        """
        根据文件路径创建 Audio 对象。
        
        参数:
            file_path (str): 音频文件路径
        
        返回:
            Audio: 包含音频数据的 Audio 对象
        """
        base_path = Val_set_path if filename.startswith('5') else Train_set_path
        file_path = os.path.join(base_path, filename)
        audio, _ = librosa.load(file_path, sr=None, mono=True)
        if isinstance(audio, Audio):
            return audio
        return cls(label=filename, audio=audio)

    @classmethod
    def from_label(cls, label, audio_data, fft=None, stft=None, mel=None, mfcc=None):
        """
        根据标签和音频数据创建 Audio 对象。
        
        参数:
            label (str): 音频标签
            audio_data (numpy.ndarray): 音频数据
        
        返回:
            Audio: 包含音频数据的 Audio 对象
        """
        return cls(label=label, audio=audio_data, fft=fft, stft=stft, mel=mel, mfcc=mfcc)

    def set_features(self, fft=None, stft=None, mfcc=None):
        """
        设置音频的特征。
        
        参数:
            fft (numpy.ndarray): FFT 特征
            stft (numpy.ndarray): STFT 特征
            mfcc (numpy.ndarray): mfcc 特征
        """
        if fft is not None:
            self.fft = fft
        if stft is not None:
            self.stft = stft
        if mfcc is not None:
            self.mfcc = mfcc


class AudioData:
    def __init__(self, labels, audios, ffts=None, stfts=None, mels=None, mfccs=None,
                 is_validation=False, fft_top20=None, stft_top20=None, mel_top20=None, mfcc_top20=None):
        """
        初始化 AudioData 对象，用于存储多个音频的标签和特征数据。

        参数:
            labels (numpy.ndarray): 音频标签列表
            audios (numpy.ndarray): 音频数据列表
            is_validation (bool): 指示数据集是否为验证集。默认为 False。
            fft_top20 (dict): FFT 特征的 Top 20 相似项映射（仅验证集适用）。键为标签，值为相似项列表。
            stft_top20 (dict): STFT 特征的 Top 20 相似项映射（仅验证集适用）。键为标签，值为相似项列表。
            mfcc_top20 (dict): MFCC 特征的 Top 20 相似项映射（仅验证集适用）。键为标签，值为相似项列表。
        """

        self.labels = labels
        self.audios = audios
        self.is_validation = is_validation
        self.ffts = ffts
        self.stfts = stfts
        self.mels = mels
        self.mfccs = mfccs

        # 初始化验证集特征字典
        if is_validation:
            self.fft_top20 = fft_top20 if fft_top20 is not None else {}
            self.stft_top20 = stft_top20 if stft_top20 is not None else {}
            self.mel_top20 = mel_top20 if mel_top20 is not None else {}
            self.mfcc_top20 = mfcc_top20 if mfcc_top20 is not None else {}

    def release_memory(self):
        self.ffts = None
        self.stfts = None
        self.mfccs = None
        print("Release memory")

    @classmethod
    def load_data(cls, directory_path, is_validation=False):
        """
        从指定目录加载音频文件并生成 AudioData 对象。

        参数:
            directory_path (str): 音频文件所在的目录路径
            is_validation (bool): 是否为验证集

        返回:
            AudioData: 包含多个 Audio 对象的 AudioData 对象
        """
        audios = []
        labels = []

        audio_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        for file in tqdm(audio_files, desc="Loading audios", unit="file"):
            file_path = os.path.join(directory_path, file)
            audio, _ = librosa.load(file_path, sr=None, mono=True)
            audios.append(audio)
            labels.append(file)

        return cls(labels=np.array(labels), audios=np.array(audios, dtype=np.float32), is_validation=is_validation)

    def add_top20_features(self, label, fft_top20=None, stft_top20=None, mfcc_top20=None):
        """
        添加验证集音频的特征 Top 20 列表。
        
        参数:
            label (str): 标签名
            fft_top20 (list): FFT 的 Top 20 相似项
            stft_top20 (list): STFT 的 Top 20 相似项
            mfcc_top20 (list): mfcc 的 Top 20 相似项
        """
        if not self.is_validation:
            raise ValueError("This method is only applicable for validation datasets.")

        if fft_top20 is not None:
            self.fft_top20[label] = fft_top20
        if stft_top20 is not None:
            self.stft_top20[label] = stft_top20
        if mfcc_top20 is not None:
            self.mfcc_top20[label] = mfcc_top20

    def __getitem__(self, label):
        """
        根据标签获取音频数据。

        参数:
            label (str): 音频标签

        返回:
            numpy.ndarray: 对应的音频数据
        """
        index = np.where(self.labels == label)[0]
        if len(index) == 0:
            raise KeyError(f"No audio found for label: {label}")
        audio_data = self.audios[index[0]]
        audio_fft = None if self.ffts is None else self.ffts[index[0]]
        audio_stft = None if self.stfts is None else self.stfts[index[0]]
        audio_mfcc = None if self.mfccs is None else self.mfccs[index[0]]
        return Audio.from_label(label, audio_data, audio_fft, audio_stft, audio_mfcc)

    def __len__(self):
        """
        返回数据集中 Audio 对象的数量。
        """
        return len(self.audios)

    # 计算所有音频的 FFT 特征
    def compute_ffts(self, train_data=True):
        """
        计算所有音频的 FFT 特征。
        
        参数:
            load_from_data (bool): 是否从已有数据加载（占位）。
        """
        pkl_path = Val_ffts if self.is_validation else Train_ffts
        if train_data:  # 训练数据
            self.ffts = fft(self.audios)
            save_pkl(self.ffts, pkl_path)
        else:  # 从数据中加载
            self.ffts = load_pkl(pkl_path)

    # 计算所有音频的 STFT 特征
    def compute_stfts(self, window_size=1024, hop_size=512, use_library=False):
        """
        计算所有音频的 stft 特征。
        
        参数:
            load_from_data (bool): 是否从已有数据加载（占位）。
        """
        base_path = Val_stfts if self.is_validation else Train_stfts
        folder_name = f"{window_size}_{hop_size}"
        save_dir = os.path.join(base_path, folder_name)
        pkl_path = os.path.join(save_dir, "stfts.pkl")

        # 如果文件夹存在且文件已保存，加载数据
        if os.path.exists(pkl_path):
            self.stfts = load_pkl(pkl_path)
        else:
            if use_library:
                self.stfts = stft_lib(self.audios, fs=44100, window_size=window_size, hop_size=hop_size)
            else:
                self.stfts = stft(self.audios, window_size=window_size, hop_size=hop_size, use_library=True)
                save_pkl(self.stfts, pkl_path)

    # 计算所有音频的 mel 特征
    def compute_mels(self, window_size=1024, hop_size=512, num_mel_bins=128, use_library=False):
        """
        计算所有音频的 mel 特征。
        
        参数:
            load_from_data (bool): 是否从已有数据加载（占位）。
        """
        base_path = Val_mels if self.is_validation else Train_mels
        folder_name = f"{window_size}_{hop_size}_{num_mel_bins}"
        save_dir = os.path.join(base_path, folder_name)
        pkl_path = os.path.join(save_dir, "mels.pkl")

        # 如果文件夹存在且文件已保存，加载数据
        if os.path.exists(pkl_path):
            self.mels = load_pkl(pkl_path)
        else:
            if use_library:
                self.mels = librosa.feature.melspectrogram(y=self.audios, sr=44100, n_fft=window_size,
                                                           hop_length=hop_size,
                                                           n_mels=num_mel_bins, power=1.0)
            else:
                if self.stfts is None:
                    self.compute_stfts(window_size=window_size, hop_size=hop_size)
                self.mels = mels(self.stfts, num_mel_bins=num_mel_bins)
                save_pkl(self.mels, pkl_path)

    # 计算所有音频的 MFCC 特征
    def compute_mfccs(self, train_data=True):
        """
        计算所有音频的 mfcc 特征。
        
        参数:
            load_from_data (bool): 是否从已有数据加载（占位）。
        """
        pkl_path = Val_mfccs if self.is_validation else Train_mfccs
        if train_data:  # 训练数据
            self.mfccs = mfccs(self.mels)
            save_pkl(self.mfccs, pkl_path)
        else:  # 从数据中加载
            self.mfccs = load_pkl(pkl_path)

    # 计算 fft top20的准确率
    def get_fft_top20(self, Train_Data: 'AudioData', f=calculate_euclidean_distance):
        top20_dict = {}
        for val_label, val_fft in tqdm(zip(self.labels, self.ffts),
                                       desc="Processing FFT Top 20",
                                       total=len(self.labels)):
            similarities = []
            for train_label, train_fft in zip(Train_Data.labels, Train_Data.ffts):
                sim = compute_fft_sim(val_fft, train_fft, f=f)
                similarities.append((train_label, sim))
            top20 = sorted(similarities, key=lambda x: x[1], reverse=True)[:20]
            top20_labels = [item[0] for item in top20]
            top20_dict[val_label] = top20_labels
        self.fft_top20 = top20_dict

        # 计算 mel top20的准确率

    def get_mel_top20(self, Train_Data: 'AudioData', f=calculate_euclidean_distance):
        top20_dict = {}
        for val_label, val_mel in tqdm(zip(self.labels, self.mels),
                                       desc="Processing mel Top 20",
                                       total=len(self.labels)):
            similarities = []
            for train_label, train_mel in zip(Train_Data.labels, Train_Data.mels):
                sim = compute_mel_sim(val_mel, train_mel, f=f)
                similarities.append((train_label, sim))
            top20 = sorted(similarities, key=lambda x: x[1], reverse=True)[:20]
            top20_labels = [item[0] for item in top20]
            top20_dict[val_label] = top20_labels
        self.mel_top20 = top20_dict

    # 计算 stft top20的准确率
    def get_stft_top20(self):
        # TODO
        pass

    # 计算 mfcc top20的准确率
    def get_mfcc_top20(self):
        # TODO
        pass
