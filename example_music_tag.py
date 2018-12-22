import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import keras
import time
import numpy as np
from keras.models import load_model

import music_preProcess
import crnn_model

def tagging(result):
    tags = ['pop music', 'electronic music', 'country music', 'hiphop music', 'jazz music', 'light music']
    loc = np.argmax(result)
    tag = tags[loc]

    return tag

print("Start music genre...\n")

# Load model
print("Load model...",end='')
time_start = time.time()
crnn_model = load_model('./data/crnn_model.h5')
time_end = time.time()
print('done')
print('Cost time: ' + str(time_end - time_start) + 's' + '\n')

# Load music data
print("Load music data...",end='')

music_set = ['Anna Of The North - Us(electro).mp3', 'Beyonce - Single ladies(pop).mp3',
             'Candy Dulfer - Still I Love You(jazz).mp3', 'Giovanni Marradi - Come Back To Me.mp3',
             'Hanne Sorvaag - Say Hello To Goodbye.mp3']

print('done')
print('loaded music:')
for music in music_set:
    print(music)
print('')

# Generate mel-grams and save
print("Generate mel-grams...")
melgrams = []
for music in music_set:
    time_start = time.time()
    melgram = music_preProcess.preProcess('./data/', music, False)
    time_end = time.time()
    print(music+' mel-gram is generated. Cost time: ' + str(time_end - time_start) + 's')

    # Save melgrams
    melName = music.split('.')
    melName = melName[0] + ".npy"
    np.save('./data/' + melName, melgram)
    print(melName+" is saved")

    melgrams.append(melgram)
print("done\n")

# Use model to predict
print("Predict...")
tags = []
for i in range(len(melgrams)):
    time_start = time.time()
    result = crnn_model.predict(melgrams[i])
    time_end = time.time()
    print("prediction" + str(i+1) + ":",end='')
    print(result, end = '')
    print(" Cost time: " + str(time_end - time_start) + "s")
    tag = tagging(result)
    tags.append(tag)
print("done\n")

print("Result:")
for i in range(len(music_set)):
    print(music_set[i] + " is a " + tags[i] + " song")
