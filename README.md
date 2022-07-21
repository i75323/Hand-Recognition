# Hand Gesture Recognition System

This research mainly uses [MediaPipe](https://google.github.io/mediapipe/),an open source project developed by Google, to design the overall system. The palm skeleton detected by MediaPipe uses a deep learning architecture to recognize gestures. It mainly uses the classification and recognition architecture of the deep learning architecture Darknet19 to process the real-time detected images, and uses the Socket communication protocol to stream the recognition results to the web page of the Flask framework, so that the user can immediately get the gesture judgment. At the same time, it can control the robot to perform specific tasks, and finally the information is sent back to the user to achieve interactive communication. Can see the detection window and the robot window in the picture.

[![IMAGE ALT TEXT ](https://user-images.githubusercontent.com/69238937/180015201-5a3e2d6c-a2af-4141-8cc9-9a338c1dbf0e.png)](https://youtu.be/J3_uRAxIRy4 "Hand Gesture Recognition System" )

## Installation

  The main environment is executed in the darknet environment, and the required dependencies are described in yaml, which is exported by conda, so you can directly use the Setup/hand0707.yaml file to directly create a new virtual environment

* MediaPipe environment

  Instructions for setting up the environment :
  
  ```P
  conda env create -n yourname -f hand0707.yaml
  ```
  
  HandTrackingModule.py -> This file needs to be placed under the same folder as the executive party, because this program will be referenced when the main program is executed
  
  ```P
  #In the gesture detection recognition program
  from HandTrackingModule import HandDetector
  ```

  Conda environment erection reference URL : 
  * <https://medium.com/python4u/%E7%94%A8conda%E5%BB%BA%E7%AB%8B%E5%8F%8A%E7%AE%A1%E7%90%86python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83-b61fd2a76566>
  * <https://ithelp.ithome.com.tw/articles/10218768>

* Darknet Environment

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

## Weight

  In the weight folder, the weights trained by darknet19 are placed, which will need to be used for identification. Three different samples and the weights trained are used, including self-collected samples, Massey University samples, and Surrey, UK. University sample .
       
  * 0306 Folder ([英國薩里大學-ASL美國手語](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out))

  * 0507 Folder (自行收集樣本)
  
  * 0517 Folder ([梅西大學手勢樣本](https://www.massey.ac.nz/~albarcza/gesture_dataset2012.html))

## Data

  The samples are mainly English gestures A~Z with a total of 24 letters (excluding J and Z). This system uses a single screen for identification, so it is impossible to classify and identify dynamic gestures.The main samples used are located in the hand_data folder.
  
  * Ministry of Education announces English gestures
  
    <img src="https://user-images.githubusercontent.com/69238937/180123269-98319d3d-a0cf-412d-823d-1c6eaa8a6748.jpg" width="500" /><br/>
    
  * Homemade Gesture Skeleton Sample , These samples are placed in the hand_data folder
  
    <img src="https://user-images.githubusercontent.com/69238937/180156719-1eebfc37-b85d-4420-942e-6bfa263bfea3.png" width="500" /><br/>
  

  
   	 
## Make Gesture Data

  If you want to collect and make gesture samples by yourself, use the create_image_opencarmera.py program in do_dataset Folder, you can perform Mediapipe detection on the read images, and then convert them into skeleton samples with a simple black background.

  <img src="https://user-images.githubusercontent.com/69238937/180124835-0c8ae854-7401-4d6d-8aa3-92844f6a3192.png" width="500" /><br/>
  
  You can easily generate skeleton samples by modifying the location where the data is read below and where it is stored. Remember that the path must not use Chinese paths, and the program cannot be executed.There will be a total of three places in the file that need to be modified in create_image_opencarmera.py program.
  
  ```P
  # Programs in the create_image_opencarmera.py file
  
  ★ for filename in os.listdir("your-storage-location")
    
  ★ img = cv2.imread("your-storage-location" +'/' + filename )
    
  ★ cv2.imwrite("your-storage-location"+"/"+str(filename)+'.jpg',img2)
  ```
  
  


## Demo

* darknet_images_backup.py
  
  Execute instruction : `python darknet_images_backup.py`
  
  * It is mainly used to perform detection and identification of reading a single image.
  * The main modifications are as follows :
    ```P
    ★ detector = HandDetector(detectionCon=0.5, maxHands=1)
    # The main part is to set the threshold, detectionCon is the confidence of detecting shots, and maxHands is how many hands can be detected.
    ★ img = cv2.imread('/home/i75323/darknet/2022-05-19-231845.jpg')
    # The main thing is to read the location of the picture, you need to modify it according to your own picture path, and then the path and picture name cannot have   Chinese, and there are Chinese export bugs.
    ★ def parser(): in --weights, --config_file, --data_file
    # You need to modify it yourself according to the classification structure, training weight, etc. you use, because you need to identify the classification structure,   which must be changed; the current experiment basically uses the weight of the 0507 folder.
    ```

https://user-images.githubusercontent.com/69238937/179927707-0eee3549-7c9b-4799-a276-46ea3ef782d3.mp4 

* darknet_viedo.py
  
  Execute instruction : `python darknet_viedo.py`

  * Can perform gesture detection, gesture recognition, spelling, and most importantly, the distance between the fingers
  * The main modifications are as follows :
    ```P
    ★ def parser(): in --weights, --config_file, --data_file
    # You need to modify it yourself according to the classification structure, training weight, etc. you use, because you need to identify the classification structure, which must be changed; the current experiment basically uses the weight of the 0507 folder.
    ★ cap = cv2.VideoCapture(input_path)
    # The input_path may need to be modified, because the different experimental hardware may cause the problem that the lens may not be read (although there is basically no error~~).
    ★ detector = HandDetector(detectionCon=0.8, maxHands=2)
    # The main part is to set the threshold value, detectionCon is the confidence of detecting shots, and maxHands is how many hands can be detected. In this program, maxHands must be set to 2, because the left and right hands will be used at the same time.
    ★ for button in buttonList :
    # This is the threshold setting for function selection. The setting will only be selected within a fixed range. If you want to modify the selection position, you can modify it here.
    ```

https://user-images.githubusercontent.com/69238937/179927773-9fe97f19-f843-4349-9550-1759c3afd865.mp4

* darknet_viedo_canrun_backup.py
  
  Execute instruction : `python darknet_video_canrun_backup.py`

  * Can simply perform gesture detection and gesture recognition
  * The main modifications are as follows : 
    ```P
    ★ def parser(): in --weights, --config_file, --data_file
    # You need to modify it yourself according to the classification structure, training weight, etc. you use, because you need to identify the classification structure, which must be changed; the current experiment basically uses the weight of the 0507 folder.
    ★ cap = cv2.VideoCapture(input_path)
    # The input_path may need to be modified, because the different experimental hardware may cause the problem that the lens may not be read (although there is basically no error~~).
    ★ detector = HandDetector(detectionCon=0.8, maxHands=1)
    # The main part is to set the threshold, detectionCon is the confidence of detecting shots, and maxHands is how many hands can be detected.
    ```

https://user-images.githubusercontent.com/69238937/179927824-5a264eb9-39b0-4d62-a863-1a2267317c6b.mp4

* darknet_viedo_flask.py mith robo.py

  Execute instruction : `python robo.py` and `python darknet_video_canrun_backup.py`
 
  * Perform gesture detection, gesture recognition, spelling correction, and sending commands to the robot (displayed on the web page) , When using the darknet_video_flask program, you need to pull the templates folder to the same directory, because the template folder is where the web and html are stored. If the program is not executed, it will not be read and will not be moved. There are two html that are login and result.
  * The main modifications are as follows :
  
    darknet_viedo_flask.py :
    ```P
    ★ def parser(): in --weights, --config_file, --data_file
    # You need to modify it yourself according to the classification structure, training weight, etc. you use, because you need to identify the classification structure, which must be changed; the current experiment basically uses the weight of the 0507 folder.
    ★ cap = cv2.VideoCapture(input_path)
    # The input_path may need to be modified, because the different experimental hardware may cause the problem that the lens may not be read (although there is basically no error~~).
    ★ detector = HandDetector(detectionCon=0.8, maxHands=2)
    # The main part is to set the threshold value, detectionCon is the confidence of detecting the shot, and maxHands is how many hands can be detected. In this program, maxHands must be set to 2, because the left and right hands will be used at the same time.
    ★ for button in buttonList :
    # This is the threshold setting for function selection. The setting will only be selected within a fixed range. If you want to modify the selection position, you can modify it here.
    ★ client_socket.connect(('192.168.186.81', 8485))
    # The IP location needs to be modified according to the current URL during the experiment. Usually, it is modified to the IP of the WIFI sent by the robot. If you don't want to control the robot but just want to test the overall system, you can set it to ('0.0.0.0', 8485) and you can do it Use, if the IP is not modified, it will not be able to execute.
    ★ app.run(host="0.0.0.0", port=5000)
    # This is the IP setting of the flask web page, it can be used with basic (0.0.0.0)
    ```
    robo.py :
    ```P
    1. ep_robot = robot.Robot()
    2. ep_robot.initialize(conn_type="ap")
    3. ep_chassis = ep_robot.chassis
    4. ep_chassis.move
    5. ep_arm = ep_robot.robotic_arm
    6. ep_arm.move
    # The above are all instructions for controlling the robot. They are currently annotated in the program. If you want to control the robot, you must unpack these programs and annotate the robot to move. The annotations are only for convenience. You can test without controlling the robot. Whether the entire system can be executed. https://github.com/dji-sdk/RoboMaster-SDK/tree/master/examplesThe above URL is the official SDK for controlling the robot . 
    ```
https://user-images.githubusercontent.com/69238937/179931888-6d25d5ed-9d4f-4f8f-8b08-2f626b43d375.mp4


