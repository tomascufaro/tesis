# import parselmouth
# from parselmouth.praat import call
# import librosa
# import statistics
# import numpy as np


# def get_crest_factor_RMS(sound):
#     rms = np.mean(librosa.feature.rms(sound))
#     peak = max(np.abs(sound))
#     crest_factor = peak / rms
#     return crest_factor, rms


# def measureFormants(sound, f0min, f0max):
#     pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

#     formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
#     numPoints = call(pointProcess, "Get number of points")

#     f1_list = []
#     f2_list = []
#     f3_list = []
#     f4_list = []

#     # Measure formants only at glottal pulses
#     for point in range(0, numPoints):
#         point += 1
#         t = call(pointProcess, "Get time from index", point)

#         f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
#         f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
#         f3 = call(formants, "Get value at time", 3, t, "Hertz", "Linear")
#         f4 = call(formants, "Get value at time", 4, t, "Hertz", "Linear")

#         f1_list.append(f1)
#         f2_list.append(f2)
#         f3_list.append(f3)
#         f4_list.append(f4)

#     f1_list = [f1 for f1 in f1_list if str(f1) != "nan"]
#     f2_list = [f2 for f2 in f2_list if str(f2) != "nan"]
#     f3_list = [f3 for f3 in f3_list if str(f3) != "nan"]
#     f4_list = [f4 for f4 in f4_list if str(f4) != "nan"]

#     # calculate mean formants across pulses
#     f1_mean = statistics.mean(f1_list)
#     f2_mean = statistics.mean(f2_list)
#     f3_mean = statistics.mean(f3_list)
#     f4_mean = statistics.mean(f4_list)
#     f_means = [f1_mean, f2_mean, f3_mean, f4_mean]
#     # calculate median formants across pulses, this is what is used in all subsequent calcualtions
#     # you can use mean if you want, just edit the code in the boxes below to replace median with mean
#     f1_median = statistics.median(f1_list)
#     f2_median = statistics.median(f2_list)
#     f3_median = statistics.median(f3_list)
#     f4_median = statistics.median(f4_list)
#     f_medians = [f1_median, f2_median, f3_median, f4_median]
#     return f_means, f_medians


# def get_mfccs(y, sr):

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)
#     mfcc_delta1 = librosa.feature.delta(mfcc)
#     mfcc_delta2 = librosa.feature.delta(mfcc_delta1)

#     return mfcc, mfcc_delta1, mfcc_delta2


# def feature_extraction(audio, sr, f0min, f0max, unit="Hertz"):
#     sound = parselmouth.Sound(audio, sr)  # read the sound
#     pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object

#     f0, _, _ = librosa.pyin(
#         audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
#     )
#     f0_delta = librosa.feature.delta(f0)
#     meanF0 = np.nanmean(f0)
#     stdevF0 = np.nanstd(f0)
#     meanF0delta = np.nanmean(f0_delta)

#     harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
#     hnr = call(harmonicity, "Get mean", 0, 0)

#     # f_means, f_medians = measureFormants(sound, f0min, f0max)  # Formantes

#     mfcc, mfcc_delta1, mfcc_delta2 = get_mfccs(audio, sr)
#     crest_factor, rms = get_crest_factor_RMS(audio)
#     spectral_centroid = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
#     spectral_rollof = np.mean(
#         librosa.feature.spectral_rolloff(audio, sr=sr, roll_percent=0.85)
#     )

#     zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio, sr))

#     return (
#         mfcc,
#         mfcc_delta1,
#         mfcc_delta2,
#         np.array([meanF0]),
#         np.array([stdevF0]),
#         np.array([meanF0delta]),
#         np.array([hnr]),
#         np.array([crest_factor]),
#         np.array([rms]),
#         # np.array(f_means),
#         # np.array(f_medians),
#         np.array([spectral_centroid]),
#         np.array([spectral_rollof]),
#         np.array([zero_crossing_rate]),
#     )


# def dataset_normalization(dataset, n_mfcc):

#     for n in range(3):
#         mean = np.mean(dataset[:, n * n_mfcc : (n + 1) * n_mfcc])
#         std = np.std(dataset[:, n * n_mfcc : (n + 1) * n_mfcc])
#         dataset[:, n * n_mfcc : (n + 1) * n_mfcc] = (
#             dataset[:, n * n_mfcc : (n + 1) * n_mfcc] - mean
#         ) / std
#     N = n_mfcc * 3
#     mean = np.mean(dataset[:, N:], axis=0)
#     std = np.std(dataset[:, N:], axis=0)
#     dataset[:, N:] = (dataset[:, N:] - mean) / std
#     return dataset


# path = "/home/francoj/Documentos/Reconocimiento de emociones/tesis/Data/Enterface/Actor_08/03-01-05-01-02-02-08.mp3"


# audio, sr = librosa.load_mp3(path)

# output = (
#     mfcc,
#     mfcc_delta1,
#     mfcc_delta2,
#     meanF0,
#     stdevF0,
#     meanF0delta,
#     hnr,
#     crest_factor,
#     rms,
#     # f_means,
#     # f_medians,
#     spectral_centroid,
#     spectral_rollof,
#     zero_crossing_rate,
# ) = feature_extraction(audio, sr, 300, 3300, "Hertz")

# feature_names = (
#     "mfcc",
#     "mfcc_delta1",
#     "mfcc_delta2",
#     "meanF0",
#     "stdevF0",
#     "meanF0delta",
#     "hnr",
#     "crest_factor",
#     "rms",
#     # "f_means",
#     # "f_medians",
#     "spectral_centroid",
#     "spectral_rollof",
#     "zero_crossing_rate",
# )
# features = np.concatenate(output, axis=0)


# f = []
# f.append(features)

# f.append(features + 1)

# f = np.array(f)


# def dataset_normalization(dataset, n_mfcc):

#     for n in range(3):
#         mean = np.mean(dataset[:, n * n_mfcc : (n + 1) * n_mfcc])
#         std = np.std(dataset[:, n * n_mfcc : (n + 1) * n_mfcc])
#         dataset[:, n * n_mfcc : (n + 1) * n_mfcc] = (
#             dataset[:, n * n_mfcc : (n + 1) * n_mfcc] - mean
#         ) / std
#     N = n_mfcc * 3
#     mean = np.mean(dataset[:, N:], axis=0)
#     std = np.std(dataset[:, N:], axis=0)
#     dataset[:, N:] = (dataset[:, N:] - mean) / std
#     return dataset


# # dataset = dataset_normalization(f, 16)

# print(dataset)
feature_names = (
    ["mfcc"] * 16
    + ["mfcc_delta1"] * 16
    + ["mfcc_delta2"] * 16
    + [
        "meanF0",
        "stdevF0",
        "meanF0delta",
        "hnr",
        "crest_factor",
        "rms",
        # "f_means",
        # "f_medians",
        "spectral_centroid",
        "spectral_rollof",
        "zero_crossing_rate",
    ]
)
print(feature_names)