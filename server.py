#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import cv2
import sys
import os
import numpy as np
import base64
import io
from PIL import Image
from sys import platform
import argparse
import math

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:6555")


def get_keypoint_string(my_arr):
    my_str = str(int(my_arr[0][0])) + ',' + str(int(my_arr[0][1]))
    for i in range(1,25):
        my_str += ',' + str(int(my_arr[i][0])) + ',' + str(int(my_arr[i][1]))
    return my_str

# Take in base64 string and return PIL image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    #image = image.save("geeks.jpg") 
    return cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()


    ##################### Kalman filter ##################
    '''
    It has 3 input parameters
    dynam_params  :the dimension of the state space here is 4
    measure_param : The dimension of the measurement value is 2 here
    control_params：The dimension of the control vector, the default is 0。Since there are no control variables in this model it is also 0
    kalman.processNoiseCov    ：It is the noise of the model system. The larger the noise, the more unstable the prediction result, and the easier it is to access the predicted value of the model system, and the greater the single-step change. Conversely, if the noise is small, the prediction result is not much different from the previous calculation result.
    kalman.measurementNoiseCov：Is the covariance matrix of the measurement system, the smaller the variance, the closer the prediction result is to the measured value
    '''
    kalman_RAnkle = cv2.KalmanFilter(4,2)
    kalman_RAnkle.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman_RAnkle.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman_RAnkle.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 1e-4
    kalman_RAnkle.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.005
    kalman_RAnkle.errorCovPost = np.array([[1,0],[0,1]], np.float32) * 1

    kalman_RToe = cv2.KalmanFilter(4,2)
    kalman_RToe.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman_RToe.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman_RToe.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 1e-4
    kalman_RToe.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.005
    kalman_RToe.errorCovPost = np.array([[1,0],[0,1]], np.float32) * 1

    ori_RAnkle = np.array([[0],[0]],np.float32)
    pre_RAnkle = np.array([[0],[0]],np.float32)
    ori_RToe = np.array([[0],[0]],np.float32)
    pre_RToe = np.array([[0],[0]],np.float32)

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    while True:
        #  Wait for next request from client
        message = socket.recv()


        message_string = message.decode("utf-8")
        frame = stringToRGB(message_string)
        #print(image1)
        print(frame.shape)

        if(frame.shape[0]!=480 or frame.shape[1]!=640):
            print("image is not in right size , width should be 960 and height 540")
            break

        frame_crop = frame.copy()
        datum = op.Datum()
        datum.cvInputData = frame_crop
        opWrapper.emplaceAndPop([datum])

        cv2_img = datum.cvOutputData.copy()
        #cv2.putText(cv2_img, "FPS: %f" % (1.0/(time.time() - fps_time)), (30, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.imshow('frame', cv2_img)
        #cv2.waitKey(1)
        #fps_time = time.time()

        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        #print(datum.poseKeypoints.dtype)               # float32
        #print(datum.poseKeypoints.shape)               # (people_num, 25, 3)

        if (len(datum.poseKeypoints.shape) == 3):       # detect human
            key_points = datum.poseKeypoints[0]

            R_deg = '999'
            L_deg = '999'

            R_ankel_x = key_points[11][0]
            R_ankel_y = key_points[11][1]
            R_toe_x = key_points[22][0]
            R_toe_y = key_points[22][1]

            L_ankel_x = key_points[14][0]
            L_ankel_y = key_points[14][1]
            L_toe_x = key_points[19][0]
            L_toe_y = key_points[19][1]


            R_hip_x = key_points[9][0]
            R_hip_y = key_points[9][1]
            R_knee_x = key_points[10][0]
            R_knee_y = key_points[10][1]

            L_hip_x = key_points[12][0]
            L_hip_y = key_points[12][1]
            L_knee_x = key_points[13][0]
            L_knee_y = key_points[13][1]
            
            R_leg_len = 0
            L_leg_len = 0

            ####################### Kalman filter #######################
            # if(key_points[11][0] != 0 or key_points[11][1] != 0):
            #     ori_RAnkle = np.array([[key_points[11][0]],[key_points[11][1]]], np.float32)
            #     kalman_RAnkle.correct(ori_RAnkle)
            #     pre_RAnkle = kalman_RAnkle.predict()
            
            # if(key_points[22][0] != 0 or key_points[22][1] != 0):
            #     ori_RToe = np.array([[key_points[22][0]],[key_points[22][1]]], np.float32)
            #     kalman_RToe.correct(ori_RToe)
            #     pre_RToe = kalman_RToe.predict()

            # key_points[11][0] = pre_RAnkle[0,0]
            # key_points[11][1] = pre_RAnkle[1,0]
            # key_points[22][0] = pre_RToe[0,0]
            # key_points[22][1] = pre_RToe[1,0]

            # R_ankel_x = pre_RAnkle[0,0]
            # R_ankel_y = pre_RAnkle[1,0]
            # R_toe_x = pre_RToe[0,0]
            # R_toe_y = pre_RToe[1,0]

            keypoint_str = get_keypoint_string(datum.poseKeypoints[0])
            #print(keypoint_str)

            if ((R_toe_x==0.0 and R_toe_y==0.0) or (R_ankel_x==0.0 and R_ankel_y==0.0)):  # No ankle or thumb detected
                #print('deg = 999')
                pass
            else:
                if (abs(R_toe_x - R_ankel_x) < 1):        # If x1 is too close to x2, it means vertical
                    R_deg = '0'
                    #print('zero!!')
                elif (abs(R_toe_y - R_ankel_y) < 1):
                    if R_ankel_x > R_toe_x:
                        R_deg = '90'
                    else:
                        R_deg = '-90'
                else:
                    c = math.sqrt((R_toe_x - R_ankel_x)**2 + ((R_toe_y - R_ankel_y)**2))
                    b = abs((R_toe_y - R_ankel_y))
                    #print(str(c) + ', ' + str(b))
                    R_deg = int(math.degrees(math.acos((b/c))))       # Adjacent / hypotenuse
                    if (R_deg > 0) and (R_deg < 90):
                        if R_ankel_x > R_toe_x:                     # turn right
                            R_deg = str(R_deg)
                        else:                                       # turn left
                            R_deg = str(-R_deg)
                    #print(R_deg)

            if ((L_toe_x==0.0 and L_toe_y==0.0) or (L_ankel_x==0.0 and L_ankel_y==0.0)):  # No ankle or thumb detected
                #print('no L_deg')
                pass
            else:
                if (abs(L_toe_x - L_ankel_x) < 1):        # If x1 is too close to x2, it means vertical
                    L_deg = '0'
                    #print('zero!!')
                elif (abs(L_toe_y - L_ankel_y) < 1):
                    if L_ankel_x > L_toe_x:
                        L_deg = '90'
                    else:
                        L_deg = '-90'
                else:
                    c = math.sqrt((L_toe_x - L_ankel_x)**2 + ((L_toe_y - L_ankel_y)**2))
                    b = abs((L_toe_y - L_ankel_y))
                    #print(str(c) + ', ' + str(b))
                    L_deg = int(math.degrees(math.acos((b/c))))       # adjacent/hypoteneuse
                    if (L_deg > 0) and (L_deg < 90):
                        if L_ankel_x > L_toe_x:                     # turn right
                            L_deg = str(L_deg)
                        else:                                       # turn left
                            L_deg = str(-L_deg)
                    #print(L_deg)
            #print(str(R_ankel_x) + ' | ' + str(R_ankel_y) + ' | ' + str(R_toe_x) + ' | ' + str(R_toe_y))

            R_leg_len = str(int(math.sqrt((R_hip_x - R_knee_x)**2 + (R_hip_y - R_knee_y)**2) + math.sqrt((R_knee_x - R_ankel_x)**2 + (R_knee_y - R_ankel_y)**2)))
            L_leg_len = str(int(math.sqrt((L_hip_x - L_knee_x)**2 + (L_hip_y - L_knee_y)**2) + math.sqrt((L_knee_x - L_ankel_x)**2 + (L_knee_y - L_ankel_y)**2)))

            with open('shoe_index.txt', 'r') as f:
                temp = f.readline()
            #print(temp)
            keypoint_str_byte = bytes(keypoint_str, 'ascii') + b',' + bytes(R_deg, 'ascii') + b',' + bytes(L_deg, 'ascii') + b',' + bytes(R_leg_len, 'ascii') + b',' + bytes(L_leg_len, 'ascii') + b',' + bytes(temp.replace('\n', ''), 'ascii')
            #keypoint_str_byte = bytes(keypoint_str, 'ascii') + b',' + bytes(R_deg, 'ascii') + b',' + bytes(L_deg, 'ascii') + b',' + bytes(R_leg_len, 'ascii') + b',' + bytes(L_leg_len, 'ascii')
            socket.send(keypoint_str_byte)
        else:
            socket.send(b'0' + b',0'*49 + b',999,999,100,100,-1')        # total : 25 (x, y), so a total of 50 coordinate values
        time.sleep(.01)
except Exception as e:
    print(e)
    sys.exit(-1)
