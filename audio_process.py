import parselmouth
from abc import abstractclassmethod, ABC
from parselmouth.praat import call
import librosa
import numpy as np
from scipy.signal import lfilter, butter


from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    ClippingDistortion,
    AddImpulseResponse,
)


class Processor(ABC):
    @abstractclassmethod
    def process(self, audio, fs):
        pass


class Filter(Processor):
    def __init__(self, order=5):
        self.order = order

    def process(self, audio, fs):
        b, a = self.butter_params(fs)
        y = lfilter(b, a, audio)
        return y

    def butter_params(self, fs):
        nyq = 0.5 * fs
        low_freq = 300
        high_freq = 3300
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = butter(self.order, [low, high], btype="band")
        return b, a


class TimeStretcher(Processor):
    def __init__(self, min_rate=0.6, max_rate=1.4, p=1):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p

    def process(self, audio, fs):
        augment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.2, max_amplitude=0.5, p=1),
                TimeStretch(
                    min_rate=self.min_rate,
                    max_rate=self.max_rate,
                    p=self.p,
                    leave_length_unchanged=False,
                ),
            ]
        )
        processed_audio = augment(samples=audio, sample_rate=fs)
        return processed_audio


class PitchShifter(Processor):
    def __init__(self, min_semitones=-2, max_semitores=4, p=1):
        self.min_semitones = min_semitones
        self.max_semitones = max_semitores
        self.p = p

    def process(self, audio, fs):
        augment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.2, max_amplitude=0.5, p=1),
                PitchShift(
                    min_semitones=-self.min_semitones,
                    max_semitones=self.max_semitones,
                    p=self.p,
                ),
            ]
        )
        processed_audio = augment(samples=audio, sample_rate=fs)
        return processed_audio


class Distorter(Processor):
    def __init__(self, max_percentile_threshold=25, p=1):
        self.max_percentile_threshold = max_percentile_threshold
        self.p = p

    def process(self, audio, fs):
        augment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.2, max_amplitude=0.5, p=1),
                ClippingDistortion(
                    p=self.p, max_percentile_threshold=self.max_percentile_threshold
                ),
            ]
        )
        processed_audio = augment(samples=audio, sample_rate=fs)
        return processed_audio


class Reverberator(Processor):
    def __init__(self, ir_path="./IR/", p=1):
        self.ir_path = ir_path
        self.p = p

    def process(self, audio, fs):
        augment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.2, max_amplitude=0.4, p=1),
                AddImpulseResponse(
                    ir_path=self.ir_path,
                    p=1,
                ),
            ]
        )
        processed_audio = augment(samples=audio, sample_rate=fs)
        return processed_audio


class Augmenter:
    def __init__(self, processors: list):
        self.processors = processors

    def augment(self, audio, fs):
        output = []
        for processor in self.processors:
            processed_audio = processor.process(audio, fs)
            output.append(processed_audio)
        return output


def get_crest_factor_RMS(sound):
    rms = np.mean(librosa.feature.rms(sound))
    peak = max(np.abs(sound))
    crest_factor = peak / rms
    return crest_factor, rms


def get_mfccs(y, sr, n_mfcc):

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta1 = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta1)
    mfcc = np.mean(mfcc, axis=1)
    mfcc_delta1 = np.mean(mfcc_delta1, axis=1)
    mfcc_delta2 = np.mean(mfcc_delta2, axis=1)
    return mfcc, mfcc_delta1, mfcc_delta2


def feature_extraction(audio, sr, f0min, f0max, n_mfcc, unit="Hertz"):
    bandpass_filter = Filter()
    audio = bandpass_filter.process(audio, sr)

    f0 = librosa.yin(audio, 60, 350, frame_length=4096)
    f0_delta = librosa.feature.delta(f0)
    meanF0 = np.nanmean(f0)
    stdevF0 = np.nanstd(f0)
    meanF0delta = np.nanmean(f0_delta)

    sound = parselmouth.Sound(audio, sr)  # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    mfcc, mfcc_delta1, mfcc_delta2 = get_mfccs(audio, sr, n_mfcc)
    crest_factor, rms = get_crest_factor_RMS(audio)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
    spectral_rollof = np.mean(
        librosa.feature.spectral_rolloff(audio, sr=sr, roll_percent=0.85)
    )

    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio, sr))
    output = np.concatenate(
        [
            mfcc,
            mfcc_delta1,
            mfcc_delta2,
            np.array([meanF0]),
            np.array([stdevF0]),
            np.array([meanF0delta]),
            np.array([hnr]),
            np.array([crest_factor]),
            np.array([rms]),
            # np.array(f_means),
            # np.array(f_medians),
            np.array([spectral_centroid]),
            np.array([spectral_rollof]),
            np.array([zero_crossing_rate]),
        ]
    )
    output = list(output)
    return output
