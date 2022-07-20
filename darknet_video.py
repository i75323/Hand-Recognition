from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread
from queue import Queue
import math

from HandTrackingModule import HandDetector
import cvzone
import numpy as np

from time import sleep

from textblob import TextBlob
from spellchecker import SpellChecker

from pynput.keyboard import Controller

spell = SpellChecker()

keys = [["Space"],["Enter"],["Delete"],["Send"]]
global finalText 
keyboard = Controller()

def drawAll(img, buttonList):
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

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 550, 100 * i + 30], key))


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="/home/i75323/darknet/0507/darknet19/backup/darknet19_60000.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="/home/i75323/darknet/0507/darknet19/darknet19.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="/home/i75323/darknet/0507/darknet19/voc.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=0.1,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):    #影片的輸入位置(沒有田就預設開鏡頭0的位置)
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


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


def set_saved_video(input_video, output_video, size):      #如果需要儲存影片時可以使用的 
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):                                 #畫框時要用的
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
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


def video_capture(frame_queue,hand_queue, darknet_image_queue):            #讀取影片
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while cap.isOpened():
        
        ret, frame = cap.read()
        # prev_time = time.time()
        if not ret:
            break
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_rgb, (video_width,video_height ),
        #                            interpolation=cv2.INTER_LINEAR)
        
        
        # frame_queue.put(frame_resized)

        img2 = np.zeros([500, 500, 3], np.uint8)
        
        # frame_queue.put(frame_resized)
        
        hands, frame_resized ,img2= detector.findHands(frame,img2)

        frame_resized = drawAll(frame_resized, buttonList)
        
        # frame_resized1 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        frame_queue.put(frame_resized)

        hand_queue.put(hands)
        
        img2 = cv2.resize(img2, (darknet_width, darknet_width),interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, img2.tobytes())
        darknet_image_queue.put(img_for_detect)
        
        # print('time0:  %0.4f  second' %fps1 )
        # fps1 = (time.time() - prev_time)
        # print('time0:  %0.4f  second' %fps1 )

        del img2
        del hands
        del frame_resized
            

    cap.release()

def image_classification(image, network, class_names):
    # width = darknet.network_width(network)
    # height = darknet.network_height(network)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_resized = cv2.resize(image_rgb, (width, height),
    #                             interpolation=cv2.INTER_LINEAR)
    # darknet_image = darknet.make_image(width, height, 3)
    # darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    
    return sorted(predictions, key=lambda x: -x[1])


def inference(darknet_image_queue, detections_queue, fps_queue):
    
    while cap.isOpened():
        prev_time = time.time()
        darknet_image = darknet_image_queue.get()
        # print(f" test : {darknet_image} ")
        
        # darknet_image, img2 = hand_tracking(darknet_image)


        predictions = image_classification(darknet_image,network, class_names)
        
        detections_queue.put(predictions)
        # fps1 = (time.time() - prev_time)
        # print('time1:  %0.4f  second' %fps1 )

        # print(f" test : {predictions} ")
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        # predictions1.put(predictions[0][0])
        # predictions2.put(predictions[0][1])
        # print("FPS: {}".format(fps))
        # fps1 = (time.time() - prev_time)
        # print('time1:  %0.4f  second' %fps1 )

        # darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue,hand_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    finalText = ""
    distan = ""
    Spell = ""
    Spell1 = ""
    l = 0
    count_same_frames = 0
    # predictions = []
    while cap.isOpened():
        prev_time = time.time()
        frame = frame_queue.get()
        predictions = detections_queue.get()
        # print(f" test : {predictions[0][0]} ")
        fps = fps_queue.get()
        hands = hand_queue.get()
        # detections_adjusted = []
        # if frame is not None:
            # for label, confidence in predictions:
            # #     bbox_adjusted = convert2original(frame, bbox)
            #     detections_adjusted.append((str(label), confidence))
            # image = darknet.draw_boxes(detections_adjusted, frame, class_colors)

            # detections_adjusted.append(predictions)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(f" test : {predictions[0][1]} ")
        product = predictions[0][0]
        if predictions[0][1] > 0.25:
            if hands:
                hand1 = hands[0]
                lmList1 = hand1["lmList"]  # List of 21 Landmark points
                handType1 = hand1["type"]
                # if not args.dont_show:
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size
                    # print(x)
                    # print(y)
                    # cv2.rectangle(imgNew,(x-20,y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
                    # cv2.putText(imgNew, button.text, (x-20 , y + 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    # print(w)
                    # print("button",button.text[0])
                    if handType1=="Left":
                        l, _, _ = detector.findDistance(lmList1[8], lmList1[12], frame)
                        # print(type(l))
                    # count_same_frames = 0
                        if 530 < lmList1[8][0] < 530 + w and 130 < lmList1[8][1] < 130 + h and l < 30:

                        # l, _, _ = detector.findDistance(lmList1[8], lmList1[12], frame)
                        # if l < 15:

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

                        # l, _, _ = detector.findDistance(lmList1[8], lmList1[12], frame)
                        # if l < 15:

                            # keyboard.press(button.text)
                            cv2.rectangle(frame, (530,230), (550 + w, 230 + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, "Delete", (530 , 230 + 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            count_same_frames += 1
                            print("outside == ",count_same_frames)
                            if count_same_frames == 30:
                                print("insude == ",count_same_frames)
                                finalText = finalText[:-1]
                                # finalText += ' '
                            # print(type(finalText))

                            # clas_text += pred_text
                                count_same_frames = 0
                        elif 530 < lmList1[8][0] < 530 + w and 330 < lmList1[8][1] < 330 + h and l < 30 :

                        # l, _, _ = detector.findDistance(lmList1[8], lmList1[12], frame)
                        # if l < 15:

                            # keyboard.press(button.text)
                            cv2.rectangle(frame, (530,330), (550 + w, 330 + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, "Send", (530 , 330 + 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            count_same_frames += 1
                            print("outside == ",count_same_frames)
                            if count_same_frames == 30:
                                print("insude == ",count_same_frames)
                                Spell = TextBlob(finalText)
                                Spell1 = str(Spell.correct())

                                # Spell = spell.unknown(finalText)
                                # Spell1 = spell.correction(Spell)
                                
                            # print(type(finalText))

                            # clas_text += pred_text
                                count_same_frames = 0
                        elif 530 < lmList1[8][0] < 530 + w and 30 < lmList1[8][1] < 30 + h and l < 30 :

                        # l, _, _ = detector.findDistance(lmList1[8], lmList1[12], frame)
                        # if l < 15:

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

                            
                        # print("%d",l)
                            # l == 50
                        # sleep(0.15)
                            # cv2.waitKey(1) distan
            # Spell1 = Spell.correct()
            distan = int(l)
            # print("predictions ",predictions[0][1])
            cv2.putText(frame, product,
                                    (20, 50), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
            cv2.putText(frame, "{:0.2%}".format(predictions[0][1]) ,
                                    (20, 80), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)

            cv2.putText(frame, "distance : " + str(distan) ,
                                    (20, 120), cv2.FONT_HERSHEY_PLAIN,3, (0, 0, 255), 3)

            cv2.putText(frame, "Spelling:" + finalText, (10, 380),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(frame, "Correct:"+ Spell1, (10, 450),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow('Inference', frame)
            # fps1 = (time.time() - prev_time)
            # print('time:  %0.4f  second' %fps1 )
            fps1 = int(1/(time.time() - prev_time))
            print("FPS: {}".format(fps1))

                
                            
        else:
            cv2.putText(frame, "Spelling:" + finalText, (10, 380),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(frame, "Correct:"+ Spell1, (10, 450),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow('Inference', frame)

        if cv2.waitKey(fps) == 13:
            cv2.imshow('Inference2', frame)
            # break
        # fps1 = (time.time() - prev_time)
        # print('time2:  %0.2f  second' %fps1 )
    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
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
    # prev_time = time.time()
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Thread(target=video_capture, args=(frame_queue, hand_queue,darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue,hand_queue)).start()

    # fps1 = (time.time() - prev_time)
    # print('time2:  %0.4f  second' %fps1 )
