import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def plot_result(out, y, sr=44100):
    fig, ax = plt.subplots(2, figsize=[24, 6])
    img = librosa.display.specshow(out,
                                    sr=sr,
                                    x_axis='time',
                                    y_axis='cqt_note',
                                    ax=ax[0])
    ax[0].set_title('Constant-Q power spectrum')
    img = librosa.display.specshow(y,
                                    sr=sr,
                                    x_axis='time',
                                    y_axis='cqt_note',
                                    ax=ax[1])
    ax[1].set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(
        canvas.tostring_rgb(),
        dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (3, ))
    image = image.transpose((2, 0, 1))
    image = image.reshape((1, 3, image.shape[1], image.shape[2]))
    plt.close()
    return image