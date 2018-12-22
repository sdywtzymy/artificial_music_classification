import numpy as np
import keras
import tensorflow as tf
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

class CRNN():
    trainX = None
    trainY = None
    testX = None
    testY = None
    model = None
    score = None
    train_history = None
    tag = ['pop music', 'electronic music', 'country music', 'hiphop music', 'jazz music', 'light music']

    def __init__ (self,
                  dropout_layer_rate = 0.1,
                  rnn_dropout_rate = 0.1,
                  nb_epoch = 50,
                  batch_size = 16,
                  loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  save_model = True,
                  save_model_path = ''):

        self.dropout_layer_rate = dropout_layer_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer #rmsprop, adam
        self.save_model = save_model
        self.save_model_path = save_model_path


    def NN_getData(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        return True


    def NN_predict(self, melgram):
        if self.model is None:
            self.model = load_model('crnn_model.h5')
        result = self.model.predict(melgram)
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

        #RocAuc = RocAucEvaluation(validation_data=(testX,testY), interval=1)

        # Train the model (nb_epoch)
        #self.model.fit(x = trainX, y = trainY, epochs = self.nb_epoch, batch_size = self.batch_size, validation_data = (testX, testY),callbacks=[RocAuc], verbose=2)
        self.train_history = self.model.fit(x = trainX, y = trainY, epochs = self.nb_epoch, batch_size = self.batch_size, validation_data = (testX, testY))
        # Save the model
        if save_model is True:
            self.model.save(save_model_path + 'crnn_model.h5')

        # Evaluate the model
        self.score = self.model.evaluate(trainX, trainY, self.batch_size)
        print("Model evaluation: {}".format(self.score))

        return self.model, self.score
