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
import time

# plt.style.use('dark_background')

class Transcriber(object):
    def __init__(self, model_path, sr=44100, dtype='float32', frame_size=512 * 4 - 2):
        # basic config
        self.dtype = dtype
        self.dtype_size = np.dtype(self.dtype).itemsize
        self.frame_size=frame_size
        self.sr = sr
        
        # model config 
        self.model = GuitarFormer()
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        #  state_dict = torch.load(model_path)['state_dict']
        self.model.load_state_dict(state_dict)
        self.guitarsolo_stream = os.fdopen(sys.stdin.fileno(), 'rb', buffering=self.frame_size*self.dtype_size)
        #  self.guitarsolo_stream = os.fdopen(sys.stdin.fileno(), 'rb', buffering=0)

        # init plot
        self.amp = np.zeros((64, 84))
        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow(self.amp.T, aspect="auto")
        self.image.set_clim(0, 1)
        self.ax.set_xlabel(f"Time frame")
        self.ax.set_ylabel(f"Frequency")
        self.fig.colorbar(self.image)
        self.result = [np.zeros((84)), np.zeros((84))]
        self.frames = np.zeros((self.frame_size))

        threading.Thread(target=self.transcribe).start()
        
    def data_to_frame(self, data):
        data = data[:len(data)//self.dtype_size*self.dtype_size]
        data = np.frombuffer(data, dtype=self.dtype)
        frame = np.pad(data, (0, self.frame_size - len(data)))
        return frame


    def transcribe(self):
        while 1:
            data = self.guitarsolo_stream.read(self.frame_size*self.dtype_size)
            frames = self.data_to_frame(data)
            #  frames = np.clip(frames, -0.09, 0.09)
            self.frames = frames

    def analysis(self, result):
        idx = np.argsort(result)[-5:]
        notes = []
        for i in idx:
            if result[i] > 0.8:
                note = librosa.midi_to_note(i, unicode=False)
                notes.append(note)
        if len(notes) > 0:
            chord = pychord.find_chords_from_notes(list(set([note[:-1] for note in notes])))
            print(chord, notes)
    
    def loop(self):
        while 1:
            frames = self.frames
            features = np.abs(librosa.cqt(frames, sr=self.sr, hop_length=512)).T
            features = [features]
            features = np.array(features).astype(np.float32)
            result = self.model(torch.Tensor(features))
            result = result.detach().cpu().numpy().reshape(84)
            old_result = self.result
            self.result = self.result[1:]
            self.result.append(result)


            for i in range(len(old_result)):
                old_result[i] = old_result[i][::-1]
            result = result[::-1]
            self.amp = np.roll(self.amp, -1, axis=0)
            self.amp[-1] = (result>0.5) * (old_result[-1]>0.5) *result
            self.image.set_data(self.amp.T)
            plt.pause(0.01)



model_path = './checkpoint/epoch=158-step=146120.ckpt'
transcriber = Transcriber(model_path=model_path)
transcriber.loop()
