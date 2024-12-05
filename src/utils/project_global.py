# src/utils/project_global.py
import os.path

Train_set_path = "data/raw/train_set"
Val_set_path = "data/raw/val_set"

Meta_path = "data/raw/meta"

Train_ffts = "data/processed/fft/train"
Val_ffts = "data/processed/fft/val"
Train_stfts = "data/processed/stft/train"
Val_stfts = "data/processed/stft/val"
Train_mels = "data/processed/mel/train"
Val_mels = "data/processed/mel/val"
Train_mfccs = "data/processed/mfcc/train"
Val_mfccs = "data/processed/mfcc/val"

chdir = os.path.realpath(os.path.join(__file__, '..', '..', '..'))
