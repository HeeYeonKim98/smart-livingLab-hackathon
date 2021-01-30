import os
import librosa
import numpy as np
import pandas as pd
import sys
from itertools import chain
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config

config = get_config()

  
    

def create_spectrogram(filename, name):
    print(name,'to jpg --> train')
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = './data/feature/train/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S


def create_spectrogram_eval(filename, name):
    print(name, 'to jpg --> eval')
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = './data/feature/eval/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S
class FEATURE_EXTRACTOR():
    def __init__(self):
        self.sampling_rate = config.sr
        self.n_fft = config.n_fft
        self.filter = config.filter
        self.mfc_dim = config.mfc
        self.hop_length = config.hop_len
        self.win_length = config.win_len

    def get_mfcc(self, file):
        S, _ = librosa.load(file, sr=self.sampling_rate)
        mel = librosa.feature.melspectrogram(S,
                                             sr=self.sampling_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.filter,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length)
        log_S = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(S=log_S,
                                    n_mfcc=self.mfc_dim,
                                    sr=self.sampling_rate,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length
                                    )
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        mfcc_delta2 = librosa.feature.delta(mfcc, width=3, order=2)

        feature = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

        return feature

if __name__ == "__main__":
    # Train data
    wavPath = config.data_path + '/train'
    featPath = config.feat_path + '/train'
    ctrlPath = config.ctrl

    with open(ctrlPath, 'r') as f:
        files = f.read().splitlines()

    extractor = FEATURE_EXTRACTOR()
    os.makedirs(featPath, exist_ok=True)

    for i in (range(len(files))):
        file = files[i]
        filename = './data/train/'+file + '.wav'

        create_spectrogram(filename, file)

    # Test data
    wavPath = config.data_path + '/eval'
    featPath = config.feat_path + '/eval'
    ctrlPath = config.ctrl.replace('train', 'eval')

    with open(ctrlPath, 'r') as f:
        files = f.read().splitlines()

    os.makedirs(featPath, exist_ok=True)

    for i in (range(len(files))):
        file = files[i]
        filename = './data/eval/' + file + '.wav'
        create_spectrogram_eval(filename, file)
