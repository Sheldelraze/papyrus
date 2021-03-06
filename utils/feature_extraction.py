from pyAudioAnalysis import ShortTermFeatures
import numpy as np
import librosa
import os
import auditok
import scipy.io.wavfile
import traceback

from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import kurtosis
from python_speech_features.base import fbank
from functools import lru_cache
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

TEMP_FILE = 'temp.wav'
DEBUG = False
def trim_silence(x, pad=0, db_max=50):
    _, ints = librosa.effects.trim(x, top_db=db_max, frame_length=256, hop_length=64)
    start = int(max(ints[0] - pad, 0))
    end = int(min(ints[1] + pad, len(x)))
    return x[start:end]


def process_file(path, chunk=3):
    x, sr = librosa.load(path, sr=None)
    if len(x) / sr < 0.3 or len(x) / sr > 30:
        # print(len(x), sr, len(x) / sr, path)
        return None, None

    x = trim_silence(x, pad=0.25 * sr, db_max=50)
    x = x[:np.floor(chunk * sr).astype(int)]

    # pads to chunk size if smaller
    x_pad = np.zeros(int(sr * chunk))
    x_pad[:min(len(x_pad), len(x))] = x[:min(len(x_pad), len(x))]

    hop_length = np.floor(0.010 * sr).astype(int)
    win_length = np.floor(0.020 * sr).astype(int)
    return x_pad, sr, hop_length, win_length


@lru_cache(maxsize=None)
def get_MFCCS(path, final_dim=(300, 200), n_mfccs=200):
    audio, sr, hop_length, win_length = process_file(path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mels=200, n_mfcc=n_mfccs, n_fft=2048,
                                hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc = np.swapaxes(mfcc, 0, 1)
    mfcc = mfcc[:final_dim[0], :final_dim[1]]
    mfcc = np.expand_dims(mfcc, -1)
    return mfcc


@lru_cache(maxsize=None)
def get_MFCCS_v2(path, n_mfccs=200, num_segments=100, segment_length=1024):
    """
    num_segments: S
    segment_length: F
    time_skip: sigma (=ceil(num_samples / S))
    Feature:
        + mfcc with first and second difference
        + zero crossing rate
        + kurtosis
    """
    if 'train' in path:
        cache_path_file = f'cache/train_{os.path.basename(path).split(".")[0]}.npy'
    elif 'test' in path:
        cache_path_file = f'cache/test_{os.path.basename(path).split(".")[0]}.npy'
    elif 'validat' in path:
        cache_path_file = f'cache/validation_{os.path.basename(path).split(".")[0]}.npy'
    if os.path.exists(cache_path_file):
        data = np.load(cache_path_file)
    else:
        # audio, sr, hop_length, win_length = process_file(path)
        audio, sr = librosa.load(path, sr=None)
        hop_length = np.floor(0.010 * sr).astype(int)
        win_length = np.floor(0.020 * sr).astype(int)
        # try remove silence
        try:
            audio_regions = auditok.split(
                path,
                min_dur=0.3,  # minimum duration of a valid audio event in seconds
                max_dur=4,  # maximum duration of an event
                max_silence=0.05,  # maximum duration of tolerated continuous silence within an event
                energy_threshold=20  # threshold of detection
            )
            regions = sum(audio_regions)
            audio = regions.samples
            if DEBUG:
                regions.save(f"debug/{os.path.basename(path)}")
        except Exception as e:
            if DEBUG:
                traceback.print_exc()
            print(f"cant remove silence, path={path}, error -> {type(e).__name__}: {str(e)}. Skipping....")
            raise e
        data = None
        time_skip = np.ceil(len(audio) / num_segments)  # ?????
        try:
            for index in range(0, len(audio), int(time_skip)):
                segment = audio[index: index + segment_length]
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mels=200, n_mfcc=n_mfccs, n_fft=512,
                                            hop_length=hop_length)
                mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
                zcr = librosa.feature.zero_crossing_rate(segment)
                kur = kurtosis(segment)
                _, energy = fbank(segment, sr, nfft=2048)
                row = np.concatenate([np.mean(mfcc, axis=1),
                                      np.mean(mfcc_delta, axis=1),
                                      np.mean(mfcc_delta2, axis=1),
                                      [zcr.mean(), kur, np.log(energy.mean())]]).T
                row = np.expand_dims(row, -1)
                if data is None:
                    data = row
                else:
                    data = np.hstack((data, row))
        except Exception as e:
            traceback.print_exc()
            print(f"Error -> {type(e).__name__}: {str(e)}")
            print(path, len(audio) / sr, sr)
            raise e
        data = np.expand_dims(data, -1)
        np.save(cache_path_file, data)
    return data


def oversample(X, y):
    # reshape from 3D to 2D then use SMOTE to oversample
    X_rs = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    oversampler = SMOTE(random_state=69)
    X_rs, y_rs = oversampler.fit_resample(X_rs, y)
    X_rs = X_rs.reshape((X_rs.shape[0], X.shape[1], X.shape[2], 1))
    return X_rs, y_rs

def undersample(X, y):
    # random under sample
    X_rs = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    undersampler = RandomUnderSampler(random_state=69)
    X_rs, y_rs = undersampler.fit_resample(X_rs, y)
    X_rs = X_rs.reshape((X_rs.shape[0], X.shape[1], X.shape[2], 1))
    return X_rs, y_rs
