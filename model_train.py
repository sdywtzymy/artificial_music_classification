import numpy as np
import argparse

import music_preProcess
import generateData
from crnn_model import CRNN

# Plot loss and accuracy
def plotLossAcc(epochs, train_log):
    plt.figure()
    plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
    plt.title("Training Loss on train and val")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

    plt.figure()
    plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
    plt.title("Accuracy on train and val")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--generate_data', required=True)
args = parser.parse_args()
train_data_path = args.train_data_path
output_path = args.output_path

if args.generate_data:
    taggedFilePath = [train_data_path+'/pop/', train_data_path+'/electro/',train_data_path+'/country/', train_data_path+'/hiphop/',train_data_path+'/jazz/', train_data_path+'/light/']

    # Generate input data (mel-grams)
    melgrams=np.zeros((0, 96, 1366, 1))
    trainDataX, trainDataY, testDataX, testDataY = generateData.generateData(taggedFilePath,6,50)

    # Save input data
    np.save(output_path + '/trainDataX.npy', trainDataX)
    np.save(output_path + '/testDataX.npy', testDataX)
    np.save(output_path + '/trainDataY.npy', trainDataY)
    np.save(output_path + '/testDataY.npy', testDataY)
else:
    trainDataX = np.load('trainDataX.npy')
    testDataX = np.load('testDataX.npy' )
    trainDataY = np.load('trainDataY.npy' )
    testDataY = np.load('testDataY.npy' )


obj_NN = CRNN(dropout_layer_rate = 0.1,
              rnn_dropout_rate = 0.1,
              nb_epoch = 200,
              optimizer = "adam",
              batch_size = 32,
              save_model = True,
              save_model_path = output_path)
obj_NN.NN_getData(trainDataX, trainDataY, testDataX, testDataY)
model1, score1 = obj_NN.NN_model_train()
plotLossAcc(200, obj_NN.train_history)
