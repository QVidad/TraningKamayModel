# Kamay Training Model
Training the machine learning model using Mediapipe Hands that will be use for the android application<br> ❗ _️**This code is based on this [original repo](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe). **_ ❗
<br> 

This repository contains the following contents.
* Sample program in Pyhton
* Hand gesture recognition model(TFLite)
* Collecting data for hand gesture recognition

# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│     │  keypoint.csv
│     │  keypoint_classifier.hdf5
│     │  keypoint_classifier.py
│     │  keypoint_classifier.tflite
│     └─ keypoint_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is a sample program for inference.<br>
In addition, collecting data (key points) for hand gesture recognition,<br>
### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
Hand gesture recognition can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection
Press the keys according to the following:<br>
"a" - if you are collecting data <10<br>
"q" - if you are collecting data >=10 AND <20<br>
"w" - if you are collecting data >=20 AND <30<br>
"s" - if you are collecting data >=30 AND <40<br>

If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
*NOTE: the value of this will change according to the presses key above.<br>
*EXAMPLE: If you press key "q" then 2, the value on csv will be 12<br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
#### 2.Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 3" <br>and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.<br><br>

#### X.Model structure
The image of the model prepared in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>


# Reference
* [MediaPipe](https://mediapipe.dev/)
* [Training Model using Mediapipe Hands](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
