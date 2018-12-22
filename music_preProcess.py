import librosa
import librosa.display
import numpy as np

def preProcess(path, fileName, plotFlag):

    # set parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 25  # generate 1172 frame

    fileName = path + fileName
    waveForm, sampRate = librosa.load(fileName, sr=SR)  # whole signal, waveForm is waveform, sr is sampling rate
    n_sample = waveForm.shape[0] # sample points
    n_sample_fit = int(DURA*SR)

    # Select the middle
    if n_sample < n_sample_fit:  # if too short
        waveForm = np.hstack((waveForm, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        waveForm = waveForm[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]

    # Compute a mel-scaled spectrogram.
    mel_gram = librosa.feature.melspectrogram(y=waveForm, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS)

    # Convert an amplitude spectrogram to dB-scaled spectrogram.
    db_spec = librosa.power_to_db(mel_gram**2,ref=1.0)
    db_spec = db_spec[ np.newaxis, :,:,np.newaxis]

    # Plot dB-scaled spectrogram or not
    if plotFlag is True:
        plt.figure()
        plt.subplot(2, 1, 1)
        librosa.display.specshow(mel_gram**2, sr=sr, y_axis='log')
        plt.colorbar()
        plt.title('Power spectrogram')
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.power_to_db(mel_gram**2, ref=1.0),
                                 sr=sr, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-Power spectrogram')
        plt.tight_layout()
        plt.savefig('db_gram.png')
        plt.show()


    return db_spec
