#  -------------------------------------------------------------
#   Copyright (c) Cavedu.  All rights reserved.
#  -------------------------------------------------------------

import argparse
import json
import os

import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite

import cv2

import math

import time

import serial

Condition_1 = 'OBJ_1'
Condition_2 = 'OBJ_2'
Condition_3 = 'OBJ_3'
Condition_4 = 'OBJ_4'
Condition_Other = 'Other'

def get_prediction(image, interpreter, signature):
    # process image to be compatible with the model
    input_data = process_image(image, image_shape)

    # set the input to run
    interpreter.set_tensor(model_index, input_data)
    interpreter.invoke()

    # grab our desired outputs from the interpreter!
    # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
    outputs = {key: interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}

    # postprocessing! convert any byte strings to normal strings with .decode()
    for key, val in outputs.items():
        if isinstance(val, bytes):
            outputs[key] = val.decode()

    return outputs


def process_image(image, input_shape):
    width, height = image.size
    # ensure image type is compatible with model and convert if not
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.reshape(input_shape).astype(np.float32)

def UART_Servo_Command(servo,angle):    
    COMMAND = str(servo) + ',' + str(int(angle)) + '\n'
    Send_UART_command = COMMAND.encode(encoding='utf-8')    
    ser.write(Send_UART_command)  # 訊息必須是位元組類型


def main():
    global signature_inputs
    global input_details
    global model_inputs
    global signature_outputs
    global output_details
    global model_outputs
    global image_shape
    global model_index
  
    global COM_PORT
    global BAUD_RATES
    global ser

    global servo_16_angle
    global servo_11_angle
    global servo_9_angle
    global servo_8_angle
    global servo_7_angle
  
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',
        help='Model path of saved_model.tflite and signature.json file',
        required=True)
    parser.add_argument('--video',
        help='Set video number of Webcam.',
        required=False, type=int, default=0)
    parser.add_argument(
        '--com', help='Number of UART prot.', required=True)
    args = parser.parse_args()

    COM_PORT = args.com
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)

    with open( args.model + "/signature.json", "r") as f:
        signature = json.load(f)

    model_file = signature.get("filename")

    interpreter = tflite.Interpreter(args.model + '/' + model_file)
    interpreter.allocate_tensors()
    # print('interpreter=',interpreter.get_input_details())

    # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
    signature_inputs = signature.get("inputs")
    input_details = {detail.get("name"): detail for detail in interpreter.get_input_details()}
    model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
    signature_outputs = signature.get("outputs")
    output_details = {detail.get("name"): detail for detail in interpreter.get_output_details()}
    model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}
    image_shape = model_inputs.get("Image").get("shape")
    model_index = model_inputs.get("Image").get("index")

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
    classification_flag = 0
    confirm = 0

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

    Condition_1_count = 0
    Condition_2_count = 0
    Condition_3_count = 0
    Condition_4_count = 0

    Motion_Busy = 0
    
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

        image = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))

        if (times==1) and (key_flag == 0) and (Motion_Busy ==0):
            prediction = get_prediction(image, interpreter, signature)

            Label_name = signature['classes']['Label'][prediction['Confidences'].index(max(prediction['Confidences']))]
            prob = round(max(prediction['Confidences']),3)
            classification_flag = 1
             
        if (key_flag == 0) and (classification_flag == 1) and (Motion_Busy ==0):
            cv2.putText(image_src, Label_name + " " +
                str(round(prob,3)),
                (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,255), 6, cv2.LINE_AA)
            cv2.putText(image_src, Label_name + " " +
                str(round(prob,3)),
                (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,0), 2, cv2.LINE_AA)
            
            if (classification_flag == 1) and (Label_name == Condition_1) and (prob > 0.5):
                Condition_1_count = Condition_1_count + 1
                Condition_2_count = 0
                Condition_3_count = 0
                Condition_4_count = 0

            elif (classification_flag == 1) and (Label_name == Condition_2) and (prob > 0.5):
                Condition_1_count = 0
                Condition_2_count = Condition_2_count + 1
                Condition_3_count = 0
                Condition_4_count = 0

            elif (classification_flag == 1) and (Label_name == Condition_3) and (prob > 0.5):
                Condition_1_count = 0
                Condition_2_count = 0
                Condition_3_count = Condition_3_count + 1
                Condition_4_count = 0

            elif (classification_flag == 1) and (Label_name == Condition_4) and (prob > 0.5):
                Condition_1_count = 0
                Condition_2_count = 0
                Condition_3_count = 0
                Condition_4_count = Condition_4_count + 1
            
            elif (classification_flag == 1) and (Label_name == Condition_Other):
                Condition_1_count = 0
                Condition_2_count = 0
                Condition_3_count = 0
                Condition_4_count = 0
                servo_16_angle = 90
                servo_11_angle = 45
                servo_9_angle = 180
                servo_8_angle = 135
                servo_7_angle = 135

        if (Condition_1_count == 50):
            Motion_Busy = 1
            servo_16_angle = 90
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 60
            servo_7_angle = 85
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 85
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 30
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 60
            servo_7_angle = 135
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 135
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)
            UART_Servo_Command(16,servo_16_angle)

            Condition_1_count = 0
            Condition_2_count = 0
            Condition_3_count = 0
            Condition_4_count = 0

            Motion_Busy = 0

        elif (Condition_2_count == 50):
            Motion_Busy = 1
            servo_16_angle = 90
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 60
            servo_7_angle = 85
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 85
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 45
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 60
            servo_7_angle = 135
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 135
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)
            UART_Servo_Command(16,servo_16_angle)

            Condition_1_count = 0
            Condition_2_count = 0
            Condition_3_count = 0
            Condition_4_count = 0

            Motion_Busy = 0


        elif (Condition_3_count == 50):
            Motion_Busy = 1
            servo_16_angle = 90
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 55
            servo_7_angle = 85
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 85
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 135
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 55
            servo_7_angle = 135
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 135
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)
            UART_Servo_Command(16,servo_16_angle)

            Condition_1_count = 0
            Condition_2_count = 0
            Condition_3_count = 0
            Condition_4_count = 0

            Motion_Busy = 0

        elif (Condition_4_count == 50):
            Motion_Busy = 1
            servo_16_angle = 90
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 55
            servo_7_angle = 85
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 85
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 150
            servo_11_angle = 155
            servo_9_angle = 155
            servo_8_angle = 55
            servo_7_angle = 135
            UART_Servo_Command(16,servo_16_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(7,servo_7_angle)

            servo_16_angle = 90
            servo_11_angle = 45
            servo_9_angle = 180
            servo_8_angle = 135
            servo_7_angle = 135
            UART_Servo_Command(11,servo_11_angle)
            UART_Servo_Command(9,servo_9_angle)
            UART_Servo_Command(8,servo_8_angle)
            UART_Servo_Command(7,servo_7_angle)
            UART_Servo_Command(16,servo_16_angle)

            Condition_1_count = 0
            Condition_2_count = 0
            Condition_3_count = 0
            Condition_4_count = 0

            Motion_Busy = 0


        times=times+1
        if (times >= 10) and (Motion_Busy ==0):
            times=1

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

        box_center_x = box_left + int(scale / 2)
        box_center_y = box_top + int(scale / 2)

        dw = int(frame_width / 2) - box_center_x
        dh = frame_height - box_center_y
        angle = (math.atan(dw / dh)) / math.pi * 180

        cv2.line(image_src,
                 (int(frame_width / 2),frame_height),
                 (box_center_x,box_center_y),
                 (0,255,255),2)

        cv2.putText(image_src, str(round(angle,3)),
                (int(frame_width * 0.6),30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,255), 6, cv2.LINE_AA)
        cv2.putText(image_src, str(round(angle,3)),
                (int(frame_width * 0.6),30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,0), 2, cv2.LINE_AA)


        cv2.imshow('Detecting_1 ....',image_src)

    cap.release()
    cv2.destroyAllWindows()
    ser.close()


if __name__ == "__main__":
    main()
