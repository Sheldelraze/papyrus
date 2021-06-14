from pyAudioAnalysis import ShortTermFeatures
import numpy as np
import librosa

from functools import lru_cache


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
def get_MFCCS(path, final_dim=(300, 200)):
    audio, sr, hop_length, win_length = process_file(path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mels=200, n_mfcc=200, n_fft=2048,
                                hop_length=hop_length)
    mfcc = np.swapaxes(mfcc, 0, 1)
    mfcc = mfcc[:final_dim[0], :final_dim[1]]
    mfcc = np.expand_dims(mfcc, -1)
    return mfcc