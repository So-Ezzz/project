import os
from src.utils.helpers import load_audio_files, load_metadata
from src.preprocess.fft import compute_fft
from src.preprocess.stft import compute_stft
from src.preprocess.mfcc import compute_mfcc
from src.model.scoring import score_audio
from src.evaluation.evaluate import evaluate_model

def main():
    # 1. 数据加载
    print("Loading data...")
    audio_dir = "project/data/raw/"
    meta_file = "project/data/raw/meta/esc50.csv"
    audio_data = load_audio_files(audio_dir)
    metadata = load_metadata(meta_file)

    # 2. 特征提取
    print("Extracting features...")
    fft_features = {k: compute_fft(v[1]) for k, v in audio_data.items()}
    stft_features = {k: compute_stft(v[1]) for k, v in audio_data.items()}
    mfcc_features = {k: compute_mfcc(v[1], v[0]) for k, v in audio_data.items()}

    # 3. 数据划分
    print("Splitting data...")
    query_fold = 5
    query_files = metadata[metadata['fold'] == query_fold]['filename']
    candidate_files = metadata[metadata['fold'] != query_fold]['filename']
    query_data = {k: mfcc_features[k] for k in query_files if k in mfcc_features}
    candidate_data = {k: mfcc_features[k] for k in candidate_files if k in mfcc_features}

    # 4. 匹配与打分
    print("Scoring...")
    scores = score_audio(query_data, candidate_data)

    # 5. 精度评估
    print("Evaluating...")
    results = evaluate_model(scores, metadata)
    print("Evaluation Results:", results)

    # 6. 可视化（可选）
    print("Visualizing results...")
    # 可视化代码留作后续实现

if __name__ == "__main__":
    main()
