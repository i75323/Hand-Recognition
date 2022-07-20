#----------------------------------------------------------------------------
#引入檔參數
from flask import Flask, render_template, Response,request, redirect, url_for
import cv2
import numpy as np
import sys
import socket
import struct
import pickle
import zlib
from datetime import datetime
import threading
import yaml
from robomaster import robot
import time
import robomaster
#-----------------------------------------------------------------------------
#參數初始化
result_img = 0
img_map_b = 0
global status #全域變數
status = 0
#-----------------------------------------------------------------------------
#Flask網頁設置
app = Flask(__name__)
#-----------------------------------------------------------------------------
#參數初始設置
Column = 1
Row = 3
padding_top = 60
padding_bottom = 180
padding_side = 60
# Lot_width = int((img_map.shape[1] - 2*padding_side)/Row)
# Lot_height = int((img_map.shape[0] - (padding_top + padding_bottom))/Column)
# ep_robot = robot.Robot()              #當要控制機器人時記得把註解拿掉
# ep_robot.initialize(conn_type="ap")   #當要控制機器人時記得把註解拿掉
x_val = 0.5                             #機器人移動量-X軸(單位M)
y_val = 0.6                             #機器人移動量-Y軸(單位M)
z_val = 90                              #機器人移動量-Z軸
#-----------------------------------------------------------------------------
#機器人電池查詢的副程式設置(robomaster SDK裏頭官方程式)
def sub_info_handler(batter_info, ep_robot):
    percent = batter_info
    print("Battery: {0}%.".format(percent))
    ep_led = ep_robot.led
    brightness = int(percent * 255 / 100)
    ep_led.set_led(comp="all", r=brightness, g=brightness, b=brightness)


    socket_data_back = "Battery: {0}%.".format(percent)
    conn.send(socket_data_back.encode())

    return percent
#-----------------------------------------------------------------------------
#機器人終端機的SOCKET端設置 詳細設置在網址裡(基本是用最簡單的) : https://shengyu7697.github.io/python-tcp-socket/
def socket_init():
    global payload_size, conn
    HOST=''
    PORT=8485

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr=s.accept()

    socket_data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    return socket_data, payload_size
#-----------------------------------------------------------------------------
#機器人終端機的SOCKET端主要程式部分
def socket_start():  # generate frame by frame from camera
    global socket_data, result_img, img_map, img_map_b,status
    while True:
        x_val = 0.5
        y_val = 0.6
        z_val = 90
        socket_data = conn.recv(4096)                                               #socket傳輸接收端
        socket_data_finally = socket_data.decode()                                  #socket接收後解碼
        print(socket_data_finally)
        if socket_data_finally == 'forward' :
            socket_data_back ='car forward 50cm'
            # ep_chassis = ep_robot.chassis                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_chassis.move(x=x_val, y=0, z=0, xy_speed=0.7).wait_for_completed() #當要控制機器人時記得把註解拿掉
            print("car forward 50cm")

        elif socket_data_finally == 'backward' :
            socket_data_back ='car backward 50cm'
            # ep_chassis = ep_robot.chassis                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_chassis.move(x=-x_val, y=0, z=0, xy_speed=0.7).wait_for_completed()#當要控制機器人時記得把註解拿掉
            
            print("car backward 50cm")

        elif socket_data_finally == 'right' :
            socket_data_back ='car right 50cm'
            # ep_chassis = ep_robot.chassis                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_chassis.move(x=0, y=y_val, z=0, xy_speed=0.7).wait_for_completed() #當要控制機器人時記得把註解拿掉
            
            print("car right 50cm")

        elif socket_data_finally == 'left' :
            socket_data_back ='car left 50cm'
            # ep_chassis = ep_robot.chassis                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_chassis.move(x=0, y=-y_val, z=0, xy_speed=0.7).wait_for_completed()#當要控制機器人時記得把註解拿掉
            
            print("car left 50cm")

        elif socket_data_finally == 'battery' :
            
            #----------------------------▼-------------------------------------------當要控制機器人時記得把註解拿掉
            
            # ep_battery = ep_robot.battery
            # ep_battery.sub_battery_info(1, sub_info_handler, ep_robot)
            # time.sleep(2)
            # ep_battery.unsub_battery_info()
            # ep_robot.close()
            
            #----------------------------▲-------------------------------------------
            print("battery")

        elif socket_data_finally == 'arm forward' :
            socket_data_back ='arm forward 50cm'
            # ep_arm = ep_robot.robotic_arm                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_arm.move(x=50, y=0).wait_for_completed()                           #當要控制機器人時記得把註解拿掉
            
            print("arm forward 50cm")

        elif socket_data_finally == 'arm backward' :
            socket_data_back ='arm backward 50cm'
            # ep_arm = ep_robot.robotic_arm                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_arm.move(x=-50, y=0).wait_for_completed()                          #當要控制機器人時記得把註解拿掉
            
            print("arm backward 50cm")

        elif socket_data_finally == 'arm up' :
            socket_data_back ='arm up 50cm'
            # ep_arm = ep_robot.robotic_arm                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_arm.move(x=0, y=50).wait_for_completed()                           #當要控制機器人時記得把註解拿掉
            
            print("arm up 50cm")

        elif socket_data_finally == 'arm down' :
            socket_data_back ='arm down 50cm'
            # ep_arm = ep_robot.robotic_arm                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())
            # ep_arm.move(x=0, y=-50).wait_for_completed()                          #當要控制機器人時記得把註解拿掉
            
            print("arm down 50cm")

        elif socket_data_finally == 'claw open' :
            socket_data_back ='claw open'
            # ep_gripper = ep_robot.gripper                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())

            # ep_gripper.open(power=50)                                             #當要控制機器人時記得把註解拿掉
            # time.sleep(1)                                                         #當要控制機器人時記得把註解拿掉(夾爪需要停頓時間)
            # ep_gripper.pause()                                                    #當要控制機器人時記得把註解拿掉
            
            
            print("claw open")

        elif socket_data_finally == 'claw close' :
            socket_data_back ='claw close'
            # ep_gripper = ep_robot.gripper                                         #當要控制機器人時記得把註解拿掉
            conn.send(socket_data_back.encode())

            # ep_gripper.close(power=50)                                            #當要控制機器人時記得把註解拿掉
            # time.sleep(1)                                                         #當要控制機器人時記得把註解拿掉 (夾爪需要停頓時間)
            # ep_gripper.pause()                                                    #當要控制機器人時記得把註解拿掉
            
            
            print("claw close")

        else :
            print(" NO active ")
            socket_data_back ='NO active'
            conn.send(socket_data_back.encode())

#-----------------------------------------------------------------------------
#主要python執行部分
if __name__ == '__main__':

    socket_data, payload_size = socket_init()
    added_thread1 = threading.Thread(target=socket_start)
    added_thread1.start()                                                           #不一定要用thread寫，可以用其他方式
#-----------------------------------------------------------------------------