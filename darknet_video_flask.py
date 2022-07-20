# 引入檔參數-----------------------------------------------------------------
from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread
from queue import Queue
from HandTrackingModule import HandDetector
import cvzone
import numpy as np
from time import sleep
from textblob import TextBlob
from spellchecker import SpellChecker
from pynput.keyboard import Controller
from flask import Flask, render_template, Response, redirect, url_for,request
import socket
#--------------------------------------------------------------------------
#Flask網頁設置
app = Flask(__name__)
#-----------------------------------------------------------------------------
#參數初始設置

result_img = 0
# img_map_c = 0
# img_map = np.zeros([480, 480, 3], np.uint8)
img_map_b = cv2.imread("/home/i75323/darknet/templates/abc.jpg")
img_map_c = cv2.resize(img_map_b,(480,480))
#-----------------------------------------------------------------------------
#Socket傳輸網址設置(一定要改喔不然會不能動)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.186.81', 8485))
#-----------------------------------------------------------------------------
#畫面鍵盤設置設置
spell = SpellChecker()
keys = [["Space"],["Enter"],["Delete"],["Send"]]
global finalText 
global product
keyboard = Controller()
#-----------------------------------------------------------------------------
#繪製、設置按鈕的程式部分
def drawAll(img, buttonList):                           #繪製按鈕的程式部分
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        # print(x)
        # print(y)
        # cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
        #                   20, rt=0)
        cv2.rectangle(imgNew,(x-20,y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x-20 , y + 55),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    # return img
    out = img.copy()
    alpha = 0.3
    mask = imgNew.astype(bool)
    # print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    return out

class Button():                                         #繪製按鈕的部分(大小樣式)
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []                                         #按鈕文字
for i in range(len(keys)):                              #按鈕文字排列(這邊是由上往下)
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 550, 100 * i + 30], key))
#-----------------------------------------------------------------------------
#建立執行程式時可以使用的項目
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")                                           
    parser.add_argument("--input", type=str, default=0,                                                             #輸入影像
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",                                                     #儲存影片
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="/home/i75323/darknet/0507/darknet19/backup/darknet19_60000.weights",  #架構權重
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',                                                         #不顯示影像
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',                                            
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="/home/i75323/darknet/0507/darknet19/darknet19.cfg",               #分類架構
                        help="path to config file")
    parser.add_argument("--data_file", default="/home/i75323/darknet/0507/darknet19/voc.data",                      #data檔(為了抓取類別名稱)
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=0.1,                                                        #閥值
                        help="remove detections with confidence below this value")
    return parser.parse_args()

#-----------------------------------------------------------------------------
def str2int(video_path):    #影片的輸入位置(沒有田就預設開鏡頭0的位置)
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

#-----------------------------------------------------------------------------
def check_arguments_errors(args):           #轉換輸入的位置
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

#-----------------------------------------------------------------------------
def set_saved_video(input_video, output_video, size):                           #如果需要儲存影片時可以使用的(目前程式沒有使用)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

#-----------------------------------------------------------------------------
def convert2relative(bbox):                                                     #沒有用到(原本是yolo偵測畫框程式,但手勢便是是模改得所以沒有使用到)
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

#-----------------------------------------------------------------------------
def convert2original(image, bbox):                                              #沒有用到(原本是yolo偵測畫框程式,但手勢便是是模改得所以沒有使用到)
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

#-----------------------------------------------------------------------------
def convert4cropping(image, bbox):                                              #沒有用到(原本是yolo偵測畫框程式,但手勢便是是模改得所以沒有使用到)
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

#-----------------------------------------------------------------------------
def video_capture(frame_queue,hand_queue, darknet_image_queue):                     #讀取影片(開鏡頭)(第一個值行緒)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img2 = np.zeros([500, 500, 3], np.uint8)                                    #創建黑色背景
        # frame_queue.put(frame_resized)

        hands, frame_resized ,img2= detector.findHands(frame,img2)                  #偵測手部(Mediapipe偵測)
        frame_resized = drawAll(frame_resized, buttonList)                          #將輸出圖畫上按鈕
        # frame_resized1 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_queue.put(frame_resized)                                              #多執行緒傳遞值包裝值的動作
        hand_queue.put(hands)                                                       #多執行緒傳遞值包裝值的動作
        img2 = cv2.resize(img2, (darknet_width, darknet_width),interpolation=cv2.INTER_LINEAR)  #壓縮輸入架構的圖片
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)                   #建立原本yolo偵測的東西
        darknet.copy_image_from_bytes(img_for_detect, img2.tobytes())                           #建立原本yolo偵測的東西(我應該沒用到應該)
        darknet_image_queue.put(img_for_detect)                                     #多執行緒傳遞值包裝值的動作

        del img2                                                                    #清空值,為的是防止記憶體使用過量,不清除會一直累積
        del hands
        del frame_resized
            

    cap.release()                                                                   #關鏡頭(記憶體釋放)

#-----------------------------------------------------------------------------

def image_classification(image, network, class_names):                              #圖片辨識分類的副程式部分
    # width = darknet.network_width(network)
    # height = darknet.network_height(network)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_resized = cv2.resize(image_rgb, (width, height),
    #                             interpolation=cv2.INTER_LINEAR)
    # darknet_image = darknet.make_image(width, height, 3)
    # darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, image)                              #進行架構引入以及辨識 
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)] #得出辨識結果名稱,信心度
    
    return sorted(predictions, key=lambda x: -x[1])

#-----------------------------------------------------------------------------

def inference(darknet_image_queue, detections_queue, fps_queue):                    #進行圖片分類辨識主程式部分(第二個值行緒)
    
    while cap.isOpened():                                                           #設置當開起鏡頭就一直運作
        darknet_image = darknet_image_queue.get()                                   #執行緒接收資料的動作
        # print(f" test : {darknet_image} ")
        prev_time = time.time()                                                     #計算FPS時間
        # darknet_image, img2 = hand_tracking(darknet_image)
        predictions = image_classification(darknet_image,network, class_names)      #進行副程式辨識(將值給取出來)
        detections_queue.put(predictions)                                           #多執行緒傳遞值包裝值的動作
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)                                                          #多執行緒傳遞值包裝值的動作

        # darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)                                           #清空暫存器的值(為的是防止記憶體使用過量,不清除會一直累積)
    cap.release()                                                                   #關鏡頭(記憶體釋放)

#-----------------------------------------------------------------------------

def drawing(frame_queue, detections_queue, fps_queue,hand_queue):                   #將辨識結果畫至顯示畫面上(第三個執行緒)
    random.seed(3)  # deterministic bbox colors                                     #不知道這做啥的
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))    #沒用到
    detector = HandDetector(detectionCon=0.8, maxHands=2)                           #設置Mediapipe偵測(maxHands=>可以同時偵測幾隻手,detectionCon=>信心度閥值)
    finalText = ""
    Spell = ""
    Spell1 = ""
    data1 = ""
    l = ""
    global result_img
    global product
    global img_map_c
    
    count_same_frames = 0
    # predictions = []
    while cap.isOpened():
        prev_time = time.time()
        img_map = np.zeros([480, 480, 3], np.uint8)                                 #建立新的畫面,為了放後續結果圖
        frame = frame_queue.get()                                                   #多執行緒接收資料動作
        predictions = detections_queue.get()                                        #多執行緒接收資料動作
        # print(f" test : {predictions[0][0]} ")
        fps = fps_queue.get()                                                       #多執行緒接收資料動作
        hands = hand_queue.get()                                                    #多執行緒接收資料動作
        # detections_adjusted = []
        # if frame is not None:
            # for label, confidence in predictions:
            # #     bbox_adjusted = convert2original(frame, bbox)
            #     detections_adjusted.append((str(label), confidence))
            # image = darknet.draw_boxes(detections_adjusted, frame, class_colors)

            # detections_adjusted.append(predictions)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(f" test : {predictions[0][1]} ")
        product = predictions[0][0]                                                 #讀取結果數值(信心度值)
        if predictions[0][1] > 0.25:                                                #當超過閥值時啟動
            if hands:
                hand1 = hands[0]                                                    #手的整體偵測資訊
                lmList1 = hand1["lmList"]  # List of 21 Landmark points             #21骨架點的資訊
                handType1 = hand1["type"]                                           #看是左手還是右手
                # if not args.dont_show:
                for button in buttonList:                                           #設置選取功能按鈕的部分
                    x, y = button.pos                                               #讀去按鈕位置
                    w, h = button.size                                              #讀取按鈕長寬
                    
                    if handType1=="Left":                                           #當是左手才進行功能選取
                        l, _, _ = detector.findDistance(lmList1[8], lmList1[12], frame)#將左手食指與左手中指指尖的距離讀取出來
                    # count_same_frames = 0
                        if 530 < lmList1[8][0] < 530 + w and 130 < lmList1[8][1] < 130 + h and l < 30:#判斷指尖距離是否有在按鈕範圍(總共有4個按鈕)

                        

                            # keyboard.press(button.text)
                            cv2.rectangle(frame, (530,130), (550 + w, 130 + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, "Enter", (530 , 130 + 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            count_same_frames += 1
                            print("outside == ",count_same_frames)
                            if count_same_frames == 30:
                                print("insude == ",count_same_frames)
                                productlower = product.lower()
                                finalText += productlower

                            # clas_text += pred_text
                                count_same_frames = 0
                        elif 530 < lmList1[8][0] < 530 + w and 230 < lmList1[8][1] < 230 + h and l < 30 :

                        

                            # keyboard.press(button.text)
                            cv2.rectangle(frame, (530,230), (550 + w, 230 + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, "Delete", (530 , 230 + 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            count_same_frames += 1
                            print("outside == ",count_same_frames)
                            if count_same_frames == 30:
                                print("insude == ",count_same_frames)
                                finalText = finalText[:-1]
                            # print(type(finalText))

                            # clas_text += pred_text
                                count_same_frames = 0
                        elif 530 < lmList1[8][0] < 530 + w and 330 < lmList1[8][1] < 330 + h and l < 30 :

                        

                            # keyboard.press(button.text)
                            cv2.rectangle(frame, (530,330), (550 + w, 330 + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, "Send", (530 , 330 + 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            count_same_frames += 1
                            print("outside == ",count_same_frames)
                            if count_same_frames == 30:
                                print("insude == ",count_same_frames)
                                Spell = TextBlob(finalText)
                                Spell1 = str(Spell.correct())
                                client_socket.send(Spell1.encode())

                                data = client_socket.recv(1024)
                                data1 = data.decode()
                                print(data1)

                                # Spell = spell.unknown(finalText)
                                # Spell1 = spell.correction(Spell)
                                
                            # print(type(finalText))

                            # clas_text += pred_text
                                count_same_frames = 0
                        elif 530 < lmList1[8][0] < 530 + w and 30 < lmList1[8][1] < 30 + h and l < 30 :

                        

                            # keyboard.press(button.text)
                            cv2.rectangle(frame, (530,30), (550 + w, 30 + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, "Space", (530 , 30 + 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            count_same_frames += 1
                            print("outside == ",count_same_frames)
                            if count_same_frames == 30:
                                print("insude == ",count_same_frames)
                                # productlower = product.lower()
                                finalText += ' '

                                # Spell = spell.unknown(finalText)
                                # Spell1 = spell.correction(Spell)
                                
                            # print(type(finalText))

                            # clas_text += pred_text
                                count_same_frames = 0
                        else : 
                            count_same_frames = 0

                            
                        
            
            cv2.putText(img_map, "Detected letters:  " ,
                                    (20, 50), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)                       #放數值於顯示畫面上(辨識字母)
            cv2.putText(img_map, product,(350, 65), cv2.FONT_HERSHEY_PLAIN,5, (255, 0, 255), 5)                 #放數值於顯示畫面上(辨識信心度)
            cv2.putText(img_map, "Correct rate: "+ "{:0.2%}".format(predictions[0][1]) ,
                                    (20, 90), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
            cv2.putText(img_map, "Spelling:" + finalText, (10, 300),cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  #放數值於顯示畫面上(拼字語句)
            cv2.putText(img_map, "Correct:"+ Spell1, (10, 370),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)       #放數值於顯示畫面上(校正後的語句)
            cv2.putText(img_map, "Task:"+ data1, (10, 440),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)         #放數值於顯示畫面上(任務)
            image = np.concatenate((frame, img_map,img_map_c), axis=1)                                          #將多個畫面連接起來

            
            

            ret, buffer = cv2.imencode('.jpg', image)                                                           #將顯示畫面進行編碼
            result_img = buffer.tobytes()                                                                       #顯示資訊丟至暫存器當中

            

                
                            
        else:
            cv2.putText(img_map, "Detected letters:",
                                    (20, 50), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
            cv2.putText(img_map, "Correct rate: " ,(20, 90), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
            cv2.putText(img_map, "Spelling:" + finalText, (10, 300),cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img_map, "Correct:"+ Spell1, (10, 370),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(img_map, "Task:"+ data1, (10, 440),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            image = np.concatenate((frame, img_map,img_map_c), axis=1) 
            # cv2.imshow('Inference', image)
            # cv2.waitKey(1)
            ret, buffer = cv2.imencode('.jpg', image)
            result_img = buffer.tobytes()

            
    cap.release()                                                                                               #鏡頭關閉
    video.release()                                                                                             #暫存器清空
    cv2.destroyAllWindows()                                                                                     #所有畫面關掉
#-----------------------------------------------------------------------------



def flask_start():                                                                                              #flask開啟
    app.run(host="0.0.0.0", port=5000)                                                                          #IP設置

def gen_frames():                                                                                               #開啟影像串流部分
    global result_img                                                                                           #全域變數(為了讓不同副程式的資訊流通)
    
    while True:
        try:
            yield (b'--frame\r\n'                                                                               #網頁影像串流編碼部分
                    b'Content-Type: image/jpeg\r\n\r\n' + result_img + b'\r\n')  # concat frame one by one and show result
        except:
            continue
def gen_map():                                                                                                  #這個好像沒用到
    global result_img_c
    while True:
        try:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + result_img_c+ b'\r\n')  # concat frame one by one and show result
        except:
            continue

@app.route('/video_feed')
def video_feed():                                                                                               #執行網頁影像串流部分
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_map')
def video_map():
    return Response(gen_map(), mimetype='multipart/x-mixed-replace; boundary=frame')

#------------------------------------------------設置網頁部分(有設置登入介面,雖然很爛很爛~~)
@app.route('/')
def index():                                                                                                    
    """Video streaming home page."""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def result():
    
    if request.method == 'POST':
        name = request.values['user']
        # name = 'user'
    # KEY = predictions[0][1]

        return render_template('result.html', name = name , KEY = product)
#-------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=False)
    frame_queue = Queue()                                                   #多執行緒的初始設置(設置使用的記憶體大小應該吧?)
    darknet_image_queue = Queue(maxsize=1)                                  #多執行緒的初始設置(maxsize可以調)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    hand_queue = Queue(maxsize=1)
    # class_names = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    print(f'{darknet_width}')
    darknet_height = darknet.network_height(network)
    print(f'{darknet_height}')
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Thread(target=video_capture, args=(frame_queue, hand_queue,darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue,hand_queue)).start()
    Thread(target=flask_start).start()
    # Thread(target=socket_start).start()
    
