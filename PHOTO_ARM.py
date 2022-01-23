#  -------------------------------------------------------------
#   Copyright (c) Cavedu.  All rights reserved.
#  -------------------------------------------------------------

import argparse
import json
import os

import numpy as np
from PIL import Image

import cv2

import math

import time

import serial


def UART_Servo_Command(servo,angle):    
    COMMAND = str(servo) + ',' + str(int(angle)) + '\n'
    Send_UART_command = COMMAND.encode(encoding='utf-8')    
    ser.write(Send_UART_command)  # 訊息必須是位元組類型

def main():
    global COM_PORT
    global BAUD_RATES
    global ser

    global servo_16_angle
    global servo_11_angle
    global servo_9_angle
    global servo_8_angle
    global servo_7_angle

    servo_16_angle = 90
    servo_11_angle = 45
    servo_9_angle = 180
    servo_8_angle = 135
    servo_7_angle = 90

    Motion_Step = 0
  
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--video', help='Video number', required=False, type=int, default=0)
    parser.add_argument(
        '--com', help='Number of UART prot.', required=True)
    args = parser.parse_args()

    COM_PORT = 'COM'+str(args.com)
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    key_detect = 0
    times = 1

    scale=50
    box_left = 0
    box_top = 0
    box_right = 0
    box_bottom = 0

    key_flag = 1
    confirm = 0

    UART_Servo_Command(16,servo_16_angle)
    UART_Servo_Command(11,servo_11_angle)
    UART_Servo_Command(9,servo_9_angle)
    UART_Servo_Command(8,servo_8_angle)
    UART_Servo_Command(7,servo_7_angle)

    Motion_buzy = 0
    
    while (key_detect == 0):
        ret,img_src = cap.read()

        if (args.video == 0):
            image_src = cv2.flip(img_src,1)
        else:
            image_src = img_src

        frame_width = image_src.shape[1]
        frame_height = image_src.shape[0]

        crop_img_2 = image_src

        if (confirm == 0):
            crop_img = crop_img_2
        if (confirm == 1):
            crop_img = crop_img_2[box_top:box_top + scale, box_left:box_left + scale]
                            
        read_dir_key = cv2.waitKeyEx(1)
        if (read_dir_key != -1):
            print(read_dir_key)
            # 按 +鍵 擴大影像辨識範圍
            if (read_dir_key == 43):
                if ((box_top + scale) < crop_img_2.shape[0]) and ((box_left + scale) < crop_img_2.shape[1]):
                    key_flag = 1
                    scale = scale + 2
                    
            # 按 -鍵 縮小影像辨識範圍
            elif (read_dir_key == 45):
                key_flag = 1
                scale = scale - 2
                if (scale <= 50):
                    scale = 50

            # 按 上鍵 移動影像辨識框向上        
            elif ((read_dir_key == 2490368) or (read_dir_key == 56)):
                key_flag = 1
                box_top = box_top - 10
                if (box_top <= 0):
                    box_top = 0

            # 按 下鍵 移動影像辨識框向下        
            elif ((read_dir_key == 2621440) or (read_dir_key == 50)):
                key_flag = 1
                box_top = box_top + 10
                if ((box_top + scale) >= crop_img_2.shape[0]):
                    over_screen = 1
                    box_top = crop_img_2.shape[0] - scale

            # 按 左鍵 移動影像辨識框向左        
            elif ((read_dir_key == 2424832) or (read_dir_key == 52)):
                key_flag = 1
                box_left = box_left - 10
                if (box_left <= 0):
                    box_left = 0

            # 按 右鍵 移動影像辨識框向右        
            elif ((read_dir_key == 2555904) or (read_dir_key == 54)):
                key_flag = 1
                box_left = box_left + 10
                if ((box_left + scale) >= crop_img_2.shape[1]):
                    over_screen = 1
                    box_left = crop_img_2.shape[1] - scale

            # 按 Enter鍵 確認影像辨識框的位置與範圍        
            elif (read_dir_key == 13):
                key_flag = 0
                confirm = 1

            # 按 C鍵 開關抓夾    
            elif (read_dir_key == 99):
                if (servo_7_angle == 135):
                    servo_7_angle = 90
                    UART_Servo_Command(7,servo_7_angle)
                elif (servo_7_angle == 90):
                    servo_7_angle = 135
                    UART_Servo_Command(7,servo_7_angle)

            # 按 Home鍵 手臂回到初始狀態    
            elif (read_dir_key == 2359296):
                servo_16_angle = 90
                servo_11_angle = 45
                servo_9_angle = 180
                servo_8_angle = 135
                servo_7_angle = 135
                UART_Servo_Command(16,servo_16_angle)
                UART_Servo_Command(11,servo_11_angle)
                UART_Servo_Command(9,servo_9_angle)
                UART_Servo_Command(8,servo_8_angle)
                UART_Servo_Command(7,servo_7_angle)

            # 按 PageDown鍵 手臂置於夾取預備位置    
            elif (read_dir_key == 2228224):
                servo_16_angle = 90
                servo_11_angle = 155
                servo_9_angle = 155
                servo_8_angle = 55
                UART_Servo_Command(8,servo_8_angle)
                UART_Servo_Command(9,servo_9_angle)
                UART_Servo_Command(11,servo_11_angle)
                UART_Servo_Command(16,servo_16_angle)


            # 按 F1鍵 控制底座馬達Servo_16 向左    
            elif (read_dir_key == 7340032):
                servo_16_angle = servo_16_angle + 1
                if (servo_16_angle >= 180):
                    servo_16_angle = 180
                UART_Servo_Command(16,servo_16_angle)

            # 按 F2鍵 控制底座馬達Servo_16 向右    
            elif (read_dir_key == 7405568):
                servo_16_angle = servo_16_angle - 1
                if (servo_16_angle <= 0):
                    servo_16_angle = 0
                UART_Servo_Command(16,servo_16_angle)

            # 按 F3鍵 控制Servo_11 角度變大    
            elif (read_dir_key == 7471104):
                servo_11_angle = servo_11_angle + 1
                if (servo_11_angle >= 180):
                    servo_11_angle = 180
                UART_Servo_Command(11,servo_11_angle)

            # 按 F4鍵 控制Servo_11 角度變小    
            elif (read_dir_key == 7536640):
                servo_11_angle = servo_11_angle - 1
                if (servo_11_angle <= 0):
                    servo_11_angle = 0
                UART_Servo_Command(11,servo_11_angle)
            
            # 按 F5鍵 控制Servo_9 角度變大    
            elif (read_dir_key == 7602176):
                servo_9_angle = servo_9_angle + 1
                if (servo_9_angle >= 180):
                    servo_9_angle = 180
                UART_Servo_Command(9,servo_9_angle)

            # 按 F6鍵 控制Servo_9 角度變小    
            elif (read_dir_key == 7667712):
                servo_9_angle = servo_9_angle - 1
                if (servo_9_angle <= 0):
                    servo_9_angle = 0
                UART_Servo_Command(9,servo_9_angle)

            # 按 F7鍵 控制Servo_8 角度變大    
            elif (read_dir_key == 7733248):
                servo_8_angle = servo_8_angle + 1
                if (servo_8_angle >= 180):
                    servo_8_angle = 180
                UART_Servo_Command(8,servo_8_angle)

            # 按 F8鍵 控制Servo_8 角度變小    
            elif (read_dir_key == 7798784):
                servo_8_angle = servo_8_angle - 1
                if (servo_8_angle <= 0):
                    servo_8_angle = 0
                UART_Servo_Command(8,servo_8_angle)


                
            # 按 q鍵 或 Esc鍵 結束程式    
            elif ((read_dir_key == 113) or (read_dir_key == 27)):
                key_detect = 1

        if (key_flag == 1):
            cv2.rectangle(image_src,
                (box_left,box_top),(box_left+scale,box_top+scale),
                (0,255,0),2)

        elif (key_flag == 0):
            cv2.rectangle(image_src,
                (box_left,box_top),(box_left+scale,box_top+scale),
                (0,0,255),2)

        if (confirm == 1) and (read_dir_key == 32):
            file_number = str(int(time.time() * 10))
            file_name = 'Photo_' + file_number + '.jpg'
            save_img=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_name,save_img)
            

        box_center_x = box_left + int(scale / 2)
        box_center_y = box_top + int(scale / 2)

        dw = int(frame_width / 2) - box_center_x
        dh = frame_height - box_center_y
        angle = (math.atan(dw / dh)) / math.pi * 180

        cv2.putText(image_src, str(round(angle,3)),
                (int(frame_width * 0.6),30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,255), 6, cv2.LINE_AA)
        cv2.putText(image_src, str(round(angle,3)),
                (int(frame_width * 0.6),30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,0), 2, cv2.LINE_AA)

        cv2.line(image_src,
                 (int(frame_width / 2),frame_height),
                 (box_center_x,box_center_y),
                 (0,255,255),2)

        cv2.imshow('Detecting_1 ....',image_src)

    cap.release()
    cv2.destroyAllWindows()
    ser.close()


if __name__ == "__main__":
    main()
