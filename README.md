# Artificial Music Genre

An artificial music genre with CRNN.

This project is for ECE544 (2018 Fall) course in UIUC.

## Dependencies
* Python3
* Librosa
* Keras, with version >= 2.0
* Tensorflow, with version >= 1.4

Notice: this project uses Keras with Tensorflow backend. Though we try to set parameters' dimensions to fit both two backends, you may meet errors if you are using Theano as backend.

## Prerequisites
If you want to run files in '/trainer', you need [Google Cloud](https://cloud.google.com)

## Files summary
* Folder '/data': This folder contains the stored model 'crnn_model.h5', some test songs, and the mel-grams data generated by these songs.
* Folder '/img': This folder contains the images needed by this readme.
* Folder '/trainer' : This folder is for Google Cloud trainning, along with file 'config.yaml' and 'setup.py'.
* File 'music_preProcess.py': Convert a song into its paticular mel-gram spectrum. 
* File 'generateData.py': Crate trainning data.
* File 'crnn_model.py': The CRNN model.
* File 'model_train.py' : Model trainning.
* File 'example_music_tag.py': Use our model to predict.

## Model Trainning
You can train this model on Google Cloud or your local machine.
### on Google Cloud
Use '/trainer/task.py'
```
PROJECT="YOUR_PROJECT_NAME"
BUCKET="gs://${PROJECT}"
JOB_NAME="YOUR_JOB_NAME"
PATH="PROJECT_LOCAL_PATH"
gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path ${PATH}/artificial_music_classification/trainer \
  --module-name trainer.task \
  --staging-bucket ${BUCKET} \
  --runtime-version=1.6 \
  --python-version=3.5 \
  --region us-central1 \
  --config ${PATH}/artificial_music_classification/config.yaml \
  -- \
  --train_data_path ${PATH}\YOUR_TRAINNING_DATA
  --output_path ${PATH}\YOUR_OUTPUT_FOLDER
  --generate_data 1/0
```
### on local machine
Use 'model_train.py'
```
python model_train.py \
  ---train_data_path [YOUR_TRAINNING_DATA_PATH]
  --output_path [YOUR OUTPUT PATH]
  --generate_data [1/0]
```
Note: For program own options, 
* 'train_data_path' is the path of tranning data
* 'output_path' is the path of results
* 'generate_data' is for using music files(like '.mp3') or mel-gram files(like '.npy') as input. '1' stands for using music data and '0' stands for using mel-gram data.

## Model
![CRNN](https://github.com/sdywtzymy/artificial_music_classification/blob/master/img/crnn.png "CRNN model")
The CRNN model is built in class CRNN in 'crnn_model.py'

## Example Usage
```
python example_music_tag.py
```

## Example Result
![result](https://github.com/sdywtzymy/artificial_music_classification/blob/master/img/result.png "result")

## More Infomation
you can read our report of this project.
