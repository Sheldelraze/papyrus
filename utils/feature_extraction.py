from pyAudioAnalysis import ShortTermFeatures
import numpy as np


def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step,
                           short_window, short_step):
    """
    Mid-term feature extraction
    """

    short_features, short_feature_names = \
        ShortTermFeatures.feature_extraction(signal, sampling_rate,
                                             short_window, short_step)

    n_stats = 2
    n_feats = len(short_features)
    # mid_window_ratio = int(round(mid_window / short_step))
    mid_window_ratio = round((mid_window -
                              (short_window - short_step)) / short_step)
    mt_step_ratio = int(round(mid_step / short_step))

    mid_features, mid_feature_names = [], []
    for i in range(n_stats * n_feats):
        mid_features.append([])
        mid_feature_names.append("")

    # for each of the short-term features:
    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])
        mid_feature_names[i] = short_feature_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = short_feature_names[i] + "_" + "std"

        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]

            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    mid_features = np.array(mid_features)
    mid_features = np.nan_to_num(mid_features)
    return mid_features, short_features, mid_feature_names
