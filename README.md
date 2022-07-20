# Hand Gesture Recognition System

This research mainly uses [MediaPipe](https://google.github.io/mediapipe/),an open source project developed by Google, to design the overall system. The palm skeleton detected by MediaPipe uses a deep learning architecture to recognize gestures. It mainly uses the classification and recognition architecture of the deep learning architecture Darknet19 to process the real-time detected images, and uses the Socket communication protocol to stream the recognition results to the web page of the Flask framework, so that the user can immediately get the gesture judgment. At the same time, it can control the robot to perform specific tasks, and finally the information is sent back to the user to achieve interactive communication. Can see the detection window and the robot window in the picture.

<img src="https://user-images.githubusercontent.com/69238937/179921614-b69f042d-1e04-463d-b551-3a60a7da43c6.png" width="660" /><br/>

# Installation

  The main environment is executed in the darknet environment, and the required dependencies are described in yaml, which is exported by conda, so you can directly use the Setup/hand0707.yaml file to directly create a new virtual environment

* MediaPipe environment

  Instructions for setting up the environment :
  
  ```P
  conda env create -n yourname -f hand0707.yaml
  ```

  Conda environment erection reference URL : 
  * <https://medium.com/python4u/%E7%94%A8conda%E5%BB%BA%E7%AB%8B%E5%8F%8A%E7%AE%A1%E7%90%86python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83-b61fd2a76566>
  * <https://ithelp.ithome.com.tw/articles/10218768>

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
  
  * <https://github.com/AlexeyAB/darknet>

# Weight

  In the weight folder, the weights trained by darknet19 are placed, which will need to be used for identification. Three different samples and the weights trained are used, including self-collected samples, Massey University samples, and Surrey, UK. University sample .
       
  * 0306資料夾 ([英國薩里大學-ASL美國手語](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out))

  * 0507資料夾 (自行收集樣本)
  
  * 0517資料夾 ([梅西大學手勢樣本](https://www.massey.ac.nz/~albarcza/gesture_dataset2012.html))

# Data


# Demo

* darknet_images_backup.py

  ```P
  python darknet_images_backup.py
  ```

  It is mainly used to perform detection and identification of reading a single image.

https://user-images.githubusercontent.com/69238937/179927707-0eee3549-7c9b-4799-a276-46ea3ef782d3.mp4 

* darknet_viedo.py

  ```P
  python darknet_viedo.py
  ```

  Can perform gesture detection, gesture recognition, spelling, and most importantly, the distance between the fingers

https://user-images.githubusercontent.com/69238937/179927773-9fe97f19-f843-4349-9550-1759c3afd865.mp4

* darknet_viedo_canrun_backup.py

  ```P
  python darknet_video_canrun_backup.py
  ```

  Can simply perform gesture detection and gesture recognition

https://user-images.githubusercontent.com/69238937/179927824-5a264eb9-39b0-4d62-a863-1a2267317c6b.mp4

* darknet_viedo_flask.py

  ```P
  # Open two terminals, Execute two commands separately
  1. python robo.py
  2. python darknet_video_canrun_backup.py
  ```
  
  Perform gesture detection, gesture recognition, spelling correction, and sending commands to the robot (displayed on the web page)

https://user-images.githubusercontent.com/69238937/179931888-6d25d5ed-9d4f-4f8f-8b08-2f626b43d375.mp4


