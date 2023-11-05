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
    n_augmentations = np.random.randint(0, 3)
    augmentation_funs = [add_white_noise, shift_signal, pitch_shift]#time_stretch, pitch_shift]
    aug_sig = np.copy(x)
    for _ in range(n_augmentations):
        aug_sig = np.random.choice(augmentation_funs)(aug_sig, duration=duration)
    return aug_sig


def mel_spectrogram(sig, sr=SR):
    S = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=N_FFT, n_mels=N_MELS)
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

def frequency_augmentation(x):
    x = freq_mask(x, 0.1, 2)
    # x = time_mask(x, 0.1, 2)

    return x
def freq_mask(spec, max_bandwidth_pct, num_masks):
    n_mels = spec.shape[0]
    value = spec.mean()
    aug_spec = np.copy(spec)
    for _ in range(num_masks):
        bandwidth = np.random.randint(0, max_bandwidth_pct * n_mels)
        offset = np.random.randint(0, n_mels - bandwidth)
        aug_spec[:][offset:offset+bandwidth] = value
    return aug_spec

def time_mask(spec, max_duration_pct, num_masks):
    n_samples = spec.shape[1]
    value = spec.mean()
    aug_spec = np.copy(spec)
    for _ in range(num_masks):
        duration = np.random.randint(0, max_duration_pct * n_samples)
        offset = np.random.randint(0, n_samples - duration)
        aug_spec[:][offset:offset+duration] = value
    return aug_spec

from pathlib import Path

if __name__ == '__main__':
    filepath = Path('UrbanSound8K', 'audio', 'fold1', '7383-3-0-1.wav') 
    sig = open_file(filepath, duration=3)
    aug_sig = time_stretch(sig)
    mel = mel_spectrogram(sig)
    aug_mel = frequency_augmentation(mel)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    librosa.display.specshow(mel, x_axis='time', y_axis='mel', ax=axs[0,0])
    librosa.display.specshow(aug_mel, x_axis='time', y_axis='mel', ax=axs[0,1])
    axs[1,0].plot(sig)
    axs[1,1].plot(aug_sig)
    plt.show()

