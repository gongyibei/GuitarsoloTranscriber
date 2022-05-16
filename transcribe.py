import os
import sys
import torch
from model import GuitarFormer
from dataset import cqt_feature, cqt_label
import numpy as np
import librosa.display
import librosa
import matplotlib.pyplot as plt
import threading
import pychord


class Transcriber(object):
    def __init__(self, model_path, sr=44100, dtype='float32', frame_size=512 * 4 - 2):
        self.dtype = dtype
        self.dtype_size = np.dtype(self.dtype).itemsize
        self.frame_size=frame_size
        
        self.model = GuitarFormer()
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        self.model.load_state_dict(state_dict)
        self.guitarsolo_stream = os.fdopen(sys.stdin.fileno(), 'rb', buffering=self.frame_size*self.dtype_size)
        self.sr = sr

        # threading.Thread(target=self.sync).start()
        
    def sync(self):
        while 1:
            self.guitarsolo_stream.read()
    
    def data_to_frame(self, data):
        data = data[:len(data)//self.dtype_size*self.dtype_size]
        data = np.frombuffer(data, dtype=self.dtype)
        frame = np.pad(data, (0, self.frame_size - len(data)))
        return frame


    def transcribe(self):
        while 1:
            data = self.guitarsolo_stream.read(self.frame_size*self.dtype_size)
            frames = self.data_to_frame(data)

            features = np.abs(librosa.cqt(frames, sr=self.sr)).T
            features = [features]
            features = np.array(features).astype(np.float32)
            result = self.model(torch.Tensor(features))
            result = result.detach().cpu().numpy().reshape(84)

            idx = np.argsort(result)[-5:]
            notes = []
            for i in idx:
                if result[i] > 0.8:
                    note = librosa.midi_to_note(i, unicode=False)
                    notes.append(note)
            if len(notes) > 0:
                chord = pychord.find_chords_from_notes(list(set([note[:-1] for note in notes])))
                print(chord, notes)



model_path = './checkpoint/epoch=158-step=146120.ckpt'
transcriber = Transcriber(model_path=model_path)
transcriber.transcribe()
