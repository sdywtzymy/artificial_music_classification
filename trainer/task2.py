import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from os import listdir
from keras import backend as K
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU, LSTM
from keras.utils.data_utils import get_file


def preProcess(path, fileName, plotFlag):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 25  # to make it 1172 frame..
    fileName = path + fileName
    waveForm, sampRate = librosa.load(fileName, sr=SR)  # whole signal, waveForm is waveform, sr is sampling rate
    n_sample = waveForm.shape[0] # sample points
    n_sample_fit = int(DURA*SR)
    if n_sample < n_sample_fit:  # if too short
        waveForm = np.hstack((waveForm, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        waveForm = waveForm[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    mel_gram = librosa.feature.melspectrogram(y=waveForm, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS) # Compute a mel-scaled spectrogram.
    db_spec = librosa.power_to_db(mel_gram**2,ref=1.0) #Convert an amplitude spectrogram to dB-scaled spectrogram.
    db_spec = db_spec[ np.newaxis, :,:,np.newaxis]
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
        plt.show()
    return db_spec


def generateData(taggedFilePath, tagNum, trainDataNum):
    pop_fileNames = listdir(taggedFilePath[0])
    electro_fileNames = listdir(taggedFilePath[1])
    country_fileNames = listdir(taggedFilePath[2])
    hiphop_fileNames = listdir(taggedFilePath[3])
    jazz_fileNames = listdir(taggedFilePath[4])
    light_fileNames = listdir(taggedFilePath[5])

    trainDataX = np.zeros((0, 96, 1172, 1))
    trainDataY = np.zeros((0,6))
    testDataX = np.zeros((0, 96, 1172, 1))
    testDataY = np.zeros((0,6))

    # train data
    for file in pop_fileNames[0:trainDataNum]:
        pop_Y = np.array([1,0,0,0,0,0])
        trainDataX = np.vstack([trainDataX, preProcess(taggedFilePath[0], file, False)])
        trainDataY = np.vstack([trainDataY,pop_Y])

    for file in electro_fileNames[0:trainDataNum]:
        elec_Y = np.array([0,1,0,0,0,0])
        trainDataX = np.vstack([trainDataX, preProcess(taggedFilePath[1], file, False)])
        trainDataY = np.vstack([trainDataY,elec_Y])

    for file in country_fileNames[0:trainDataNum]:
        country_Y = np.array([0,0,1,0,0,0])
        trainDataX = np.vstack([trainDataX, preProcess(taggedFilePath[2], file, False)])
        trainDataY = np.vstack([trainDataY,country_Y])

    for file in hiphop_fileNames[0:trainDataNum]:
        hiphop_Y = np.array([0,0,0,1,0,0])
        trainDataX = np.vstack([trainDataX, preProcess(taggedFilePath[3], file, False)])
        trainDataY = np.vstack([trainDataY,hiphop_Y])

    for file in jazz_fileNames[0:trainDataNum]:
        jazz_Y = np.array([0,0,0,0,1,0])
        trainDataX = np.vstack([trainDataX, preProcess(taggedFilePath[4], file, False)])
        trainDataY = np.vstack([trainDataY,jazz_Y])

    for file in light_fileNames[0:trainDataNum]:
        light_Y = np.array([0,0,0,0,0,1])
        trainDataX = np.vstack([trainDataX, preProcess(taggedFilePath[5], file, False)])
        trainDataY = np.vstack([trainDataY,light_Y])

    # test data
    for file in pop_fileNames[trainDataNum:-1]:
        pop_Y = np.array([1,0,0,0,0,0])
        testDataX = np.vstack([testDataX, preProcess(taggedFilePath[0], file, False)])
        testDataY = np.vstack([testDataY,pop_Y])

    for file in electro_fileNames[trainDataNum:-1]:
        elec_Y = np.array([0,1,0,0,0,0])
        testDataX = np.vstack([testDataX, preProcess(taggedFilePath[1], file, False)])
        testDataY = np.vstack([testDataY,elec_Y])

    for file in country_fileNames[trainDataNum:-1]:
        country_Y = np.array([0,0,1,0,0,0])
        testDataX = np.vstack([testDataX, preProcess(taggedFilePath[2], file, False)])
        testDataY = np.vstack([testDataY,country_Y])

    for file in hiphop_fileNames[trainDataNum:-1]:
        hiphop_Y = np.array([0,0,0,1,0,0])
        testDataX = np.vstack([testDataX, preProcess(taggedFilePath[3], file, False)])
        testDataY = np.vstack([testDataY,hiphop_Y])

    for file in jazz_fileNames[trainDataNum:-1]:
        jazz_Y = np.array([0,0,0,0,1,0])
        testDataX = np.vstack([testDataX, preProcess(taggedFilePath[4], file, False)])
        testDataY = np.vstack([testDataY,jazz_Y])

    for file in light_fileNames[trainDataNum:-1]:
        light_Y = np.array([0,0,0,0,0,1])
        testDataX = np.vstack([testDataX, preProcess(taggedFilePath[5], file, False)])
        testDataY = np.vstack([testDataY,light_Y])

    return trainDataX, trainDataY, testDataX, testDataY


class CRNN():
    trainX = None
    trainY = None
    testX = None
    testY = None
    model = None
    score = None
    tag = ['pop music', 'electronic music', 'country music', 'hiphop music', 'jazz music', 'light music']

    def __init__ (self,
                  dropout_layer_rate = 0.1,
                  rnn_dropout_rate = 0.1,
                  nb_epoch = 50,
                  batch_size = 16,
                  loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  save_model = True):

        self.dropout_layer_rate = dropout_layer_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer #rmsprop, adam
        self.save_model = save_model


    def NN_getData(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        return True


    def NN_predict(self, melgram):
        if self.model is None:
            self.model = load_model('crnn_model.h5')
        result = model.predict(melgram)
        loc = np.argmax(result)
        tag = self.tag[loc]

        return tag


    def NN_model_train(self):
        trainX = self.trainX
        trainY = self.trainY
        testX = self.testX
        testY = self.testY
        dropout_layer_rate = self.dropout_layer_rate
        rnn_dropout_rate = self.rnn_dropout_rate

        # Determine proper input shape
        if K.image_dim_ordering() == 'th': #theano
            input_shape = (1, 96, 1172)
        else: # tensorflow
            input_shape = (96, 1172, 1)

        weights='msd'
        melgram_input = Input(shape=input_shape)

        # Determine input axis
        if K.image_dim_ordering() == 'th':
            channel_axis = 1
            freq_axis = 2
            time_axis = 3
        else:
            channel_axis = 3
            freq_axis = 1
            time_axis = 2

        # Block 0
        x = ZeroPadding2D(padding = (0,37),
                          name = "ZeroPad_layer_0")(melgram_input)

        x = BatchNormalization(axis=freq_axis,
                               name="BatchNorm_layer_0")(x)

        # Conv Block 1
        x = Convolution2D(filters = 64, # the dimensionality of the output space
                          kernel_size = (3,3), # the height and width of the 2D convolution window
                          strides = (1,1), # specifying the strides of the convolution along the height and width.
                          padding="same", # valid, same
                          name="Conv_layer_1")(x)
        x = BatchNormalization(axis = channel_axis,
                               name = "BatchNorm_layer_1")(x)
        x = ELU()(x) # ELU function
        x = MaxPooling2D(pool_size=(2,2),
                         strides = (2,2),
                         name="MaxPool_layer_1")(x) # Max pooling
        x = Dropout(dropout_layer_rate,
                    name="Dropout_layer_1")(x)

        # Conv Block 2
        x = Convolution2D(filters = 128,
                          kernel_size = (3,3),
                          strides = (1,1),
                          padding="same",
                          name="Conv_layer_2")(x)
        x = BatchNormalization(axis = channel_axis, name = "BatchNorm_layer_2")(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3,3),strides = (3,3), name="MaxPool_layer_2")(x)
        x = Dropout(dropout_layer_rate,name="Dropout_layer_2")(x)

        # Conv Block 3
        x = Convolution2D(filters = 128,
                          kernel_size = (3,3),
                          strides = (1,1),
                          padding="same",
                          name="Conv_layer_3")(x)
        x = BatchNormalization(axis = channel_axis, name = "BatchNorm_layer_3")(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(4,4),strides = (4,4), name="MaxPool_layer_3")(x)
        x = Dropout(dropout_layer_rate,name="Dropout_layer_3")(x)

        # Conv Block 4
        x = Convolution2D(filters = 128,
                          kernel_size = (3,3),
                          strides = (1,1),
                          padding="same",
                          name="Conv_layer_4")(x)
        x = BatchNormalization(axis = channel_axis, name = "BatchNorm_layer_4")(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(4,4),strides = (4,4), name="MaxPool_layer_4")(x)
        x = Dropout(dropout_layer_rate,name="Dropout_layer_4")(x)

        # Reshape Block
        if K.image_dim_ordering == "th":
            x = Permute((3,1,2))(x)
        x = Reshape((12,128))(x)
        # add in look_back = 5

        # GRU Block 1 (OR LSTM Block)
        x = GRU(16, return_sequences=True,
                name = "GRU_Layer_1",
                recurrent_dropout=rnn_dropout_rate)(x)
        # x = LSTM(32, return_sequences=True, name = "LSTM_Layer_1", recurrent_dropout=rnn_dropout_rate)(x)

        # GRU Block 2 (OR LSTM Block)
        x = GRU(16, return_sequences=True,
                name = "GRU_Layer_2",
                recurrent_dropout=rnn_dropout_rate)(x)
        # x = LSTM(16, return_sequences=True, name = "LSTM_Layer_2", recurrent_dropout=rnn_dropout_rate)(x)

        # GRU Block 3 (OR LSTM Block)
        x = GRU(16, return_sequences=False,
                name = "GRU_Layer_3",
                recurrent_dropout=rnn_dropout_rate)(x)
        #x = LSTM(32, return_sequences=False, name = "LSTM_Layer_3", recurrent_dropout=rnn_dropout_rate)(x)

        # Output
        #x = Flatten()(x) # Flattens the input.
        x = Dense(6, activation='sigmoid', name='output')(x) # Covert multi-dim to one dim

        self.model = Model(melgram_input, x)

        # Set learning process
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])

        # Train the model (nb_epoch)
        self.model.fit(x = trainX, y = trainY, epochs = self.nb_epoch, batch_size = self.batch_size, validation_data = (testX, testY))

        # Save the model
        if save_model is True:
            self.model.save('gs://ymy_ece544/crnn_model.h5')

        # Evaluate the model
        self.score = self.model.evaluate(trainX, trainY, self.batch_size)
        print("Model evaluation: {}".format(score))

        return self.model, self.score


#taggedFilePath = ['/music/pop/', './music/electro/','./music/country/', './music/hiphop/','./music/jazz/', './music/light/']
taggedFilePath = ['gs://ymy_ece544/music/pop/', 'gs://ymy_ece544/music/electro/','gs://ymy_ece544/music/country/', 'gs://ymy_ece544/music/hiphop/','gs://ymy_ece544/music/jazz/', 'gs://ymy_ece544/music/light/']
melgrams=np.zeros((0, 96, 1366, 1))
trainDataX, trainDataY, testDataX, testDataY = generateData(taggedFilePath,6,50)
np.save('gs://ymy_ece544/trainDataX.npy', trainDataX)
np.save('gs://ymy_ece544/testDataX.npy', testDataX)
np.save('gs://ymy_ece544/trainDataY.npy', trainDataY)
np.save('gs://ymy_ece544/testDataY.npy', testDataY)
obj_NN = CRNN(dropout_layer_rate = 0.1,
              rnn_dropout_rate = 0.1,
              nb_epoch = 20,
              optimizer = "adam",
              batch_size = 32)
obj_NN.NN_getData(trainDataX, trainDataY, testDataX, testDataY)
model1, score1 = obj_NN.NN_model_train()
