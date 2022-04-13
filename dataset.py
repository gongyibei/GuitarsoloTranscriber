import os
import numpy as np
import librosa
import jams
import math
from torch.utils.data import Dataset
from tqdm import tqdm

class GuitarDataset(Dataset):
    def __init__(self, dataset_dir, n_frame=12 ,fn_feature=None, fn_label=None, sr=44100, hop=512):
        self.anno_dir = os.path.join(dataset_dir, "anno")
        self.hex_dir = os.path.join(dataset_dir, "hex")
        self.mic_dir = os.path.join(dataset_dir, "mic")

        # get file list
        self.anno_files = []
        self.hex_files = []
        self.mic_files = []
        for anno_name in os.listdir(self.anno_dir):
            if anno_name.endswith('.jams'):
                anno_file = os.path.join(self.anno_dir, anno_name)
                self.anno_files.append(anno_file)

                hex_file = os.path.join(self.hex_dir, anno_name[:-5] + '_hex.wav')
                self.hex_files.append(hex_file)

                mic_file = os.path.join(self.mic_dir, anno_name[:-5] + '_mic.wav')
                self.mic_files.append(mic_file)


        features = np.vstack([fn_feature(mic_file, sr=sr, hop=hop) for mic_file in tqdm(self.mic_files)])
        labels = np.vstack([fn_label(anno_file, sr=sr, hop=hop) for anno_file in tqdm(self.anno_files)])
        self.features, self.labels = [], []

        # skip frame to get audio clip's feature and label
        N, k = len(features), 0
        while k + n_frame < N:
            self.features.append(features[k:k+n_frame])
            self.labels.append(labels[k:k+n_frame])
            k += 1
        self.features = np.array(self.features).astype(np.float32)
        self.labels = np.array(self.labels).astype(np.float32)

    def __len__(self):
        return len(self.labels)
        

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def cqt_feature(mic_file, sr=44100, hop=512, num=100):
    y, sr = librosa.load(mic_file, sr=44100)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop)).T
    return C

def cqt_label(anno_file, sr=44100, hop=512, num=100):
    jam = jams.load(anno_file)
    nframe = math.ceil(jam.file_metadata.duration/(hop/sr))
    labels = np.zeros([nframe, 84])

    annos = jam.annotations

    pitch_contours = [pitch for pitch in jam.search(namespace='pitch')]
    note_midis = [note for note in jam.search(namespace='note')]
    beat_position = jam.search(namespace='beat_position')
    tempo = jam.search(namespace='tempo')
    instructed_chords = jam.search(namespace='chord')[0]
    performed_chords = jam.search(namespace='chord')[1]
    key_mod = jam.search(namespace='key')

    for i, note_midi in enumerate(note_midis):
        for note in note_midi:
            # print(note)
            sta_time = note.time
            end_time = note.time + note.duration
            sta_frame = round(sta_time/(hop/sr))
            end_frame = round(end_time/(hop/sr))
            labels[sta_frame:end_frame, round(note.value)-24].fill(1)
    return labels