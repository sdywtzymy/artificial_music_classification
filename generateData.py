import numpy as np
import os
import music_preProcess as mp

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
        trainDataX = np.vstack([trainDataX, mp.preProcess(taggedFilePath[0], file, False)])
        trainDataY = np.vstack([trainDataY,pop_Y])

    for file in electro_fileNames[0:trainDataNum]:
        elec_Y = np.array([0,1,0,0,0,0])
        trainDataX = np.vstack([trainDataX, mp.preProcess(taggedFilePath[1], file, False)])
        trainDataY = np.vstack([trainDataY,elec_Y])

    for file in country_fileNames[0:trainDataNum]:
        country_Y = np.array([0,0,1,0,0,0])
        trainDataX = np.vstack([trainDataX, mp.preProcess(taggedFilePath[2], file, False)])
        trainDataY = np.vstack([trainDataY,country_Y])

    for file in hiphop_fileNames[0:trainDataNum]:
        hiphop_Y = np.array([0,0,0,1,0,0])
        trainDataX = np.vstack([trainDataX, mp.preProcess(taggedFilePath[3], file, False)])
        trainDataY = np.vstack([trainDataY,hiphop_Y])

    for file in jazz_fileNames[0:trainDataNum]:
        jazz_Y = np.array([0,0,0,0,1,0])
        trainDataX = np.vstack([trainDataX, mp.preProcess(taggedFilePath[4], file, False)])
        trainDataY = np.vstack([trainDataY,jazz_Y])

    for file in light_fileNames[0:trainDataNum]:
        light_Y = np.array([0,0,0,0,0,1])
        trainDataX = np.vstack([trainDataX, mp.preProcess(taggedFilePath[5], file, False)])
        trainDataY = np.vstack([trainDataY,light_Y])

    # test data
    for file in pop_fileNames[trainDataNum:-1]:
        pop_Y = np.array([1,0,0,0,0,0])
        testDataX = np.vstack([testDataX, mp.preProcess(taggedFilePath[0], file, False)])
        testDataY = np.vstack([testDataY,pop_Y])

    for file in electro_fileNames[trainDataNum:-1]:
        elec_Y = np.array([0,1,0,0,0,0])
        testDataX = np.vstack([testDataX, mp.preProcess(taggedFilePath[1], file, False)])
        testDataY = np.vstack([testDataY,elec_Y])

    for file in country_fileNames[trainDataNum:-1]:
        country_Y = np.array([0,0,1,0,0,0])
        testDataX = np.vstack([testDataX, mp.preProcess(taggedFilePath[2], file, False)])
        testDataY = np.vstack([testDataY,country_Y])

    for file in hiphop_fileNames[trainDataNum:-1]:
        hiphop_Y = np.array([0,0,0,1,0,0])
        testDataX = np.vstack([testDataX, mp.preProcess(taggedFilePath[3], file, False)])
        testDataY = np.vstack([testDataY,hiphop_Y])

    for file in jazz_fileNames[trainDataNum:-1]:
        jazz_Y = np.array([0,0,0,0,1,0])
        testDataX = np.vstack([testDataX, mp.preProcess(taggedFilePath[4], file, False)])
        testDataY = np.vstack([testDataY,jazz_Y])

    for file in light_fileNames[trainDataNum:-1]:
        light_Y = np.array([0,0,0,0,0,1])
        testDataX = np.vstack([testDataX, mp.preProcess(taggedFilePath[5], file, False)])
        testDataY = np.vstack([testDataY,light_Y])

    return trainDataX, trainDataY, testDataX, testDataY
