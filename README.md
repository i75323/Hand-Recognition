# Hand Gesture Recognition System

This research mainly uses [MediaPipe](https://google.github.io/mediapipe/),an open source project developed by Google, to design the overall system. The palm skeleton detected by MediaPipe uses a deep learning architecture to recognize gestures. It mainly uses the classification and recognition architecture of the deep learning architecture Darknet19 to process the real-time detected images, and uses the Socket communication protocol to stream the recognition results to the web page of the Flask framework, so that the user can immediately get the gesture judgment. At the same time, it can control the robot to perform specific tasks, and finally the information is sent back to the user to achieve interactive communication. Can see the detection window and the robot window in the picture.

![image](https://user-images.githubusercontent.com/69238937/179921614-b69f042d-1e04-463d-b551-3a60a7da43c6.png)

# Installation

The main environment is executed in the darknet environment, and the required dependencies are described in yaml, which is exported by conda, so you can directly use the hand0707.yaml file to directly create a new virtual environment

* MediaPipe environment

  Instructions for setting up the environment :
  
  ```P
  conda env create -n your name -f hand0707.yaml
  ```

  Conda environment erection reference URL : 
  * (https://medium.com/python4u/%E7%94%A8conda%E5%BB%BA%E7%AB%8B%E5%8F%8A%E7%AE%A1%E7%90%86python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83-b61fd2a76566)
  * (https://ithelp.ithome.com.tw/articles/10218768)

* Darknet environment

  The darknet environment is set up according to AlexeyAB's set-up steps , The part of needism is that the following settings need to be turned on and set to 1 during construction , Mainly in the setting parameter of **LIBSO = 0** in the darknet Makefile, if it is not turned on, it will not be able to execute the subsequent programs smoothly
```P
# Mainly for the programs in the Makefile
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=1
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
  
  AlexeyAB Reference URL :
  
  * (https://github.com/AlexeyAB/darknet)
  
  

# Screenshots


# Tutorials and API

# Demo

