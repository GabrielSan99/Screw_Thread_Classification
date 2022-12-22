import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import requests

'''
INFO SECTION
- if you want to monitor raw parameters of ESP32CAM, open the browser and go to http://192.168.x.x/status
- command can be sent through an HTTP get composed in the following way http://192.168.x.x/control?var=VARIABLE_NAME&val=VALUE (check varname and value in status)
- Esse programa funciona rodando junto com o espcamera exemplo
'''

# ESP32 URL
URL = "http://192.168.137.51"
AWB = True
CAPTURE_FRAME = False
CLASSIFY = False
PATH = "background"


# Face recognition and opencv setup
cap = cv2.VideoCapture(URL + ":81/stream")

# Arquivo HaarCascade são utilizados para identificar objetos sem o uso de gpu
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem

model = load_model('model_6.h5')

def active_relay(url: str, active: int=0):
    requests.get(url + "/control?var=relay_status&val={}".format(str(active)))
    if active == 1:
        print("Relay activated!")
    else:
        print("Relay Deactivated!")

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(c0x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

# def save_frame(filename, img):
#     cv2.imwrite(filename, img)

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

if __name__ == '__main__':
    set_resolution(URL, index=8)
    while True:
        if cap.isOpened():
            key = cv2.waitKey(1)
            ret, frame = cap.read()

            # if ret:
            #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #     gray = cv2.equalizeHist(gray)

                # faces = face_classifier.detectMultiScale(gray)
                # for (x, y, w, h) in faces:
                #     center = (x + w//2, y + h//2)
                #     frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)

            cv2.imshow("frame", frame)
            
            if CAPTURE_FRAME == True:
                if time.time() - now > 0.04:
                    cv2.imwrite(PATH + "/img_{}.jpg".format(str(index)), frame)	
                    index += 1
                    now = time.time()

            if CLASSIFY == True:
                if time.time() - now > 1:
                    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grayFrame = cv2.resize(grayFrame, (100, 100))
                    npFrame = np.array(grayFrame)
                    npFrame = npFrame.reshape(-1, 100, 100, 1)
                    
                    predict = model.predict(npFrame)

                    if np.argmax(predict) == 2:
                        print("Peça com Rosca!")
                        active_relay(URL, 1)

                    elif np.argmax(predict) == 1:
                        print("Peça sem Rosca!")
                        active_relay(URL, 0)

                    else:
                        print("Sem Peça no Local...")

                    print(model.predict(npFrame))
                    now = time.time()
            

            if key == ord('r'):
                idx = int(input("Select resolution index: "))
                set_resolution(URL, index=idx, verbose=True)

            elif key == ord("s"):
                print("Started save frames!!")
                CAPTURE_FRAME = True
                now = time.time()
                index = 0

            elif key == ord("a"):
                active_relay(URL, 1)

            elif key == ord("c"):
                print("Started Classify Stream!!")
                CLASSIFY = True
                now = time.time()

            #test classify
            elif key == ord("t"):
                print("Classify Test!!")
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grayFrame = cv2.resize(grayFrame, (100, 100))
                npFrame = np.array(grayFrame)
                npFrame = npFrame.reshape(-1, 100, 100, 1)
                print(npFrame.shape)
                print(model.predict(npFrame))
                print(np.argmax(model.predict(npFrame)))
                cv2.imshow("peca", frame)

            elif key == ord('q'):
                val = int(input("Set quality (10 - 63): "))
                set_quality(URL, value=val)

            elif key == ord('a'):
                AWB = set_awb(URL, AWB)

            # esc
            elif key == 27:
                break

    cv2.destroyAllWindows()
    cap.release()