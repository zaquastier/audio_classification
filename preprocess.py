import numpy as np
import librosa

from config import *

import matplotlib.pyplot as plt
    
def open_file(path, sr=SR, duration=DURATION):
    x, sr = librosa.load(path, sr=sr, duration=duration)

    if(len(x) < sr*duration):
        pad = np.zeros(int(sr*duration - len(x)))
        padding = np.array(pad)
        x = np.concatenate((x, padding))

    return x

def time_augmentation(x, duration):
    n_augmentations = np.random.randint(0, 4)
    augmentation_funs = [add_white_noise, shift_signal, time_stretch, pitch_shift]
    for _ in range(n_augmentations):
        x = np.random.choice(augmentation_funs)(x, duration=duration)
    return x

def frequency_augmentation(x):
    return x

def mel_spectrogram(sig, sr=SR):
    S = librosa.feature.melspectrogram(y=sig, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def add_white_noise(sig, amplitude=0.05, duration=DURATION):
    noise = np.random.normal(0, np.std(sig), sig.size) * amplitude
    return sig + noise

def shift_signal(sig, duration=DURATION):
    shift = np.random.randint(0, int(duration*SR))
    return np.roll(sig, shift)

def time_stretch(sig, rate=None, sr=SR, duration=DURATION):
    if(rate):
        str_sig = librosa.effects.time_stretch(sig, rate=rate)
    else:
        str_sig = librosa.effects.time_stretch(sig, rate=np.random.random_sample() * 5) # 5? Optimal value?
    if(len(str_sig) < sr*duration):
        padding = np.zeros(int(sr*duration - len(str_sig)))
        str_sig = np.concatenate((str_sig, padding))
    else:
        str_sig = str_sig[:int(sr*duration)]

    return str_sig

def pitch_shift(sig, semitones=None, sr=SR, duration=DURATION):
    if(semitones):
        ptch_sig = librosa.effects.pitch_shift(sig, sr=sr, n_steps=semitones)
    else:
        ptch_sig = librosa.effects.pitch_shift(sig, sr=sr, n_steps=np.random.randint(-24,24))
    if(len(ptch_sig) < sr*duration):
        padding = np.zeros(int(sr*duration - len(ptch_sig)))
        ptch_sig = np.concatenate((ptch_sig, padding))
    else:
        ptch_sig = ptch_sig[:int(sr*duration)]

    return ptch_sig