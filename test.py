import torch
from model import GuitarFormer
from dataset import cqt_feature, cqt_label
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import librosa

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = GuitarFormer()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def plot_result(result, label, sr=44100):
    result = result.detach().cpu().numpy().reshape(-1, 84)
    fig, ax = plt.subplots(2, figsize=[24, 6])
    img = librosa.display.specshow(((result>0.7)*result).T, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[0])
    ax[0].set_title('predict result')
    img = librosa.display.specshow((label).T, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[1])
    ax[1].set_title('real label')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


def test(model_path, test_wav, test_anno):
    model = load_model(model_path)

    # get features 
    feature = cqt_feature(test_wav)
    label = cqt_label(test_anno)
    features = []
    win = 4
    N = len(feature)
    k = 0
    while k + win < N:
        features.append(feature[k:k+win])
        k += 1
    features = np.array(features).astype(np.float32)

    # predict
    result = model(torch.Tensor(features))
    plot_result(result, label)


model_path = './checkpoint/epoch=158-step=146120.ckpt'
test_wav = './testdata/mic/00_BN1-129-Eb_comp_mic.wav'
test_anno = './testdata/anno/00_BN1-129-Eb_comp.jams'
test(model_path, test_wav, test_anno)