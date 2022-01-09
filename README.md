# IOT-Final-Project

# **情緒零食機**
![](https://i.imgur.com/rVpnKgw.jpg)


## 一、專案啟發

身為一位時時不時就想吃點點心零食的人來說，房間裡總是備滿各式各樣的餅乾糖果，但又時常不知道該吃哪一種零食才好。因此想做一個系統讓它透過辨識使用者的臉部情緒，推薦應該吃哪種零食。

## 二、專案構想

首先透過網站上開關開啟系統，然後系統便會開啟鏡頭偵測使用者的臉部表情來判斷使用者情緒，同時偵測溫度和濕度，接著根據判斷出來的結果選擇提供的零食種類。例如：表情苦悶時，系統選擇巧克力給使用者，幫助他轉換一下心情。

## 完成圖
![](https://i.imgur.com/U07xcJk.jpg)

![](https://i.imgur.com/qGGcyzy.jpg)



## 三、用到的物品

* Raspberry Pi 3 Model B *1
* Arducam Noir Camera for Raspberry Pi [B0036]
* DHT22 Temperature Humidity Sensor Module *1
* Intel® Neural Compute Stick 2 *1
* 5V small server motor*2
* 麵包板
* 熱熔膠
* 零食盒

**零食盒完成圖(純手工超花時間)**
<img src="https://i.imgur.com/p1ME6tp.jpg" width=300 height=650 />


## 四、零件配置說明

### server motor 伺服馬達電路圖
接線顏色       | 1號伺服馬達(上面)連接的pin | 2號伺服馬達(下面)連接的pin
--------------|:-----:|:-----:|
黑色(接地)  | pin6|  pin9 | 
紅色(電源)    | pin4 |  pin2 | 
黃色(控制)  | pin11 | pin13 | 

**!!注意，以上的pin number是實體pin編號**

<img src="https://i.imgur.com/ZfaWHfx.png" width=700 height=350 />



### DHT22 溫溼度感測器電路圖
接線顏色       | DHT22 孔位 | 樹梅派的PIN位
--------------|:-----:|:-----:|
黑色(接地)  | GND|  pin34 | 
紅色(電源)    | VCC |  pin1 | 
黃色(數據信號)  | DAT | pin36 | 


<img src="https://i.imgur.com/1v2KlL8.jpg" width=600 height=300 />

**!!注意，以上的pin number是實體pin編號**


## 五、實作步驟

### 0. 硬體配置
    
**請在開始實作前確保以下硬體皆有配置完成**

1.Pi-Camera
2.DHT22 溫濕度感測器
3.server motor 伺服馬達
4.Raspberry pi

### 1. 用DHT22取得溫度資料
**1. install the CircuitPython-DHT library**
```
pip3 install adafruit-circuitpython-dht

sudo apt-get install libgpiod2
```
**2. Testing the CircuitPython DHT Library**
```
import time
import board
import adafruit_dht

# Initial the dht device, with data pin connected to:
dhtDevice = adafruit_dht.DHT22(board.D16) //D16=PIN 16

while True:
    try:
        # Print the values to the serial port
        temperature_c = dhtDevice.temperature
        temperature_f = temperature_c * (9 / 5) + 32
        humidity = dhtDevice.humidity
        print(
            "Temp: {:.1f} C    Humidity: {}% ".format(
                temperature_f, temperature_c, humidity
            )
        )

    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        dhtDevice.exit()
        raise error

    time.sleep(2.0)
```
**3. 執行程式**
```
python3 xxx.py  //xxx is your file name
```

code reference：https://learn.adafruit.com/dht-humidity-sensing-on-raspberry-pi-with-gdocs-logging/python-setup

### 2. 配置伺服馬達
![](https://i.imgur.com/TzJOntQ.png)

**1. 測試伺服馬達程式(來回旋轉90度)**
```
import RPi.GPIO as GPIO
import time
 
CONTROL_PIN = 17
PWM_FREQ = 50
STEP=15
 
GPIO.setmode(GPIO.BCM)
GPIO.setup(CONTROL_PIN, GPIO.OUT)
 
pwm = GPIO.PWM(CONTROL_PIN, PWM_FREQ)
pwm.start(0)
 
def angle_to_duty_cycle(angle=0):#把角度換算成脈衝寬度
    duty_cycle = (0.05 * PWM_FREQ) + (0.19 * PWM_FREQ * angle / 180)
    return duty_cycle
 
try:
    print('按下 Ctrl-C 可停止程式')
    for angle in range(0, 90, STEP):
        dc = angle_to_duty_cycle(angle)
        pwm.ChangeDutyCycle(dc)
        print('角度={: >3}, 工作週期={:.2f}'.format(angle, dc))
        time.sleep(2)
    for angle in range(90, 0, -STEP):
        dc = angle_to_duty_cycle(angle)
        print('角度={: >3}, 工作週期={:.2f}'.format(angle, dc))
        pwm.ChangeDutyCycle(dc)
        time.sleep(2)
    pwm.ChangeDutyCycle(angle_to_duty_cycle(0))
    while True:
        next
except KeyboardInterrupt:
    print('關閉程式')
finally:
    pwm.stop()
    GPIO.cleanup()
```
**2. 執行程式**
```
python3 xxx.py  //xxx is your file name
```

code reference：https://blog.everlearn.tw/%E7%95%B6-python-%E9%81%87%E4%B8%8A-raspberry-pi/raspberry-pi-3-mobel-3-%E5%88%A9%E7%94%A8-pwm-%E6%8E%A7%E5%88%B6%E4%BC%BA%E6%9C%8D%E9%A6%AC%E9%81%94

### 3. 情緒辨識 
**1. Installing OpenCV on Raspberry Pi**
```
sudo apt-get install libhdf5-dev -y
sudo apt-get install libhdf5-serial-dev –y
sudo apt-get install libatlas-base-dev –y
sudo apt-get install libjasper-dev -y
sudo apt-get install libqtgui4 –y
sudo apt-get install libqt4-test –y

pip3 install opencv-contrib-python==4.1.0.25 //install OpenCV
```
**2.Installing Tensorflow and Keras on Raspberry Pi**
Before installing Tensorflow and Keras, install the following mentioned libraries that are needed.
```
sudo apt-get install python3-numpy
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install python3-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install gfortran
sudo apt-get install python3-setuptools
sudo apt-get install python3-scipy
sudo apt-get update
sudo apt-get install python3-h5py
```
Install tensorflow and Keras
```
pip3 install tensorflow
pip3 install keras
```
**3. Programming Raspberry Pi for Facial Expression Recognition**
```
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
 # Load the model
model = Sequential()
classifier = load_model('ferjj.h5') # This model has a set of 6 classes
# We have 6 labels for the model
class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())
# print(class_labels)
face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')
# This function is for designing the overlay text on the predicted image boxes.
def text_on_detected_boxes(text,text_x,text_y,image,font_scale = 1,
                           font = cv2.FONT_HERSHEY_SIMPLEX,
                           FONT_COLOR = (0, 0, 0),
                           FONT_THICKNESS = 2,
                           rectangle_bgr = (0, 255, 0)):
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    # Set the Coordinates of the boxes
    box_coords = ((text_x-10, text_y+4), (text_x + text_width+10, text_y - text_height-5))
    # Draw the detected boxes and labels
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)
# Detection of the emotions on an image:
def face_detector_image(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) # Convert the image into GrayScale image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img
    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
    return rects, allfaces, img
def emotionImage(imgPath):
    img = cv2.imread(imgPath)
    rects, faces, image = face_detector_image(img)
    i = 0
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np. expand_dims(roi, axis=0)
        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] - 10))
        i = + 1
        # Overlay our detected emotion on the picture
        text_on_detected_boxes(label, label_position[0],label_position[1], image)
    cv2.imshow("Emotion Detector", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Detection of the expression on video stream
def face_detector_video(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    return (x, w, y, h), roi_gray, img
def emotionVideo(cap):
    while True:
        ret, frame = cap.read()
        rect, face, image = face_detector_video(frame)
        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (rect[0] + rect[1]//50, rect[2] + rect[3]//50)
            text_on_detected_boxes(label, label_position[0], label_position[1], image) # You can use this function for your another opencv projects.
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(image, str(fps),(5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "No Face Found", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('All', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    camera = cv2.VideoCapture(0) # If you are using an USB Camera then Change use 1 instead of 0.
    emotionVideo(camera)
    # IMAGE_PATH = "provide the image path"
    # emotionImage(IMAGE_PATH) # If you are using this on an image please provide the path
```
code reference：https://circuitdigest.com/microcontroller-projects/raspberry-pi-based-emotion-recognition-using-opencv-tensorflow-and-keras

### 4. 零食推薦判定
判定參數      | 配分 | 
-------------|:-----:|
情緒 50% | happy：1, surprise：2, neutral：3, sad：4, angry：5| 
溫度 25%    | 溫度：>20°C：2, <=20°C：-2 | 
濕度 25% | 濕度：>80%：2, <=80%：-2 |


**判定：如果總分<=1.5給甜食，>1.5給鹹食**
舉例：偵測到情緒為happy(1)、溫度21°C(2)、濕度78%(-2)，則總分為1 * 0.5+2 * 0.25+(-2) * 0.25 = 0.5。
因為0.5<=1.5所以會給甜食



### 5. Line Bot 使用教學
**1. 註冊 LINE Developers 帳號**

**2. 進到會員畫面**
![](https://i.imgur.com/IecmobE.png)

**3. 滑鼠移到下面選擇create provider**
![](https://i.imgur.com/7AHVzbh.png)

**4. 選擇 "create a Messaging API channel"**
![](https://i.imgur.com/eOY4nYh.png)

**5. 填入provider資料後完成建立**

**6. 選擇 "Basic settings"**
![](https://i.imgur.com/CrQjW41.png)

**7. 記下該頁籤下方的 "Channel secret"** 

**8. 回到上方選擇 "Messaging API"**
![](https://i.imgur.com/2ps05m9.png)

**9. 記下該頁籤下方的 "Channel access token"**

**10. 打開[Github](https://github.com/chrispa39/IOT-Final-Project-)上方的"LineBotTest"py檔案**

**11. 把剛剛記下的"Channel secret"、"Channel access token"輸入範例程式標註的位置，結束後儲存檔案**

**12.從raspberry pi開啟瀏覽器，到[ngrok](https://ngrok.com/)官網註冊並下載ngrok**

**13. 下載好後打開終端機輸入以下指令**
```
./ngrok http 80 //80代表port number
```
**14. 等待畫面跑完，複製開頭為forwarding的第二個句子從"http"到".io"。**

**15. 貼上至line developers的messaging API底下的"Webhook URL"，且貼上後不要按下verify**

**16. 打開另一個終端機執行剛剛的LineBotTest檔案，回到Webhook URL按下verify，打開下方Use webhook**

**17. 與自己的機器人加好友並與它對話，如果它學你說話就代表你的line bot建好咯!!**

## 六、遭遇的問題
1. 情緒便是程式判定不夠敏銳時常偵測不到，且不知道如何接情緒參數讀入程式中，因此在實作中是隨機產生一個情緒(但還是有實做出情緒辨識)
2. 手工製作的零食機材料較便宜且不夠堅固，所以零食會卡住，但理想狀態零食會順利滑出，因此demo是手動將零時取出

## 七、程式說明
1. DHT22_test ： 測試溫溼度感測器
2. LineBotTest：LINE bot 範例程式
3. Main_LineBot：用line bot來控制main主程式
4. ServerMotorTest：測試伺服馬達
5. emotionRecognition、Haarcascades、ferjj.h5：都是情緒辨識會用到的檔案
6. main：主程式

## 八、參考資料

* [電路圖繪製](https://fritzing.org/fbclid=IwAR3lSwAwnpekSEm1o9-QgZLrdO6w4uN85vTHICg35zV_jgX6m2r6oG2Ul7M)
* [DHT22 溫溼度感測器](https://learn.adafruit.com/dht-humidity-sensing-on-raspberry-pi-with-gdocs-logging/python-setup)
* [步進馬達](http://hophd.com/raspberry-pi-stepper-motor-control/?fbclid=IwAR0ygmnAUfaVSL9JtIuv3BQyEC7gyL8mxWEaqLNAOo_ysgk332m4NEGCIRs)
* [伺服馬達](https://blog.everlearn.tw/%E7%95%B6-python-%E9%81%87%E4%B8%8A-raspberry-pi/raspberry-pi-3-mobel-3-%E5%88%A9%E7%94%A8-pwm-%E6%8E%A7%E5%88%B6%E4%BC%BA%E6%9C%8D%E9%A6%AC%E9%81%94)
* [情緒辨識](https://circuitdigest.com/microcontroller-projects/raspberry-pi-based-emotion-recognition-using-opencv-tensorflow-and-keras)
* Line Bot 使用教學
    * [Line Bot 環境教學](https://ithelp.ithome.com.tw/articles/10238680)
    * [Line BOT Github](https://github.com/line/line-bot-sdk-python)

## 九、demo影片
* [情緒辨識測試](https://youtu.be/Rho5Ex_yBxo)
* [情緒零食機demo](https://youtu.be/wX7QTDe74Wc)


**在此感謝George、助教、淳浲、浩宸、昕哲、名容、澤陽、禹安、棨翔、亭佑、冠廷、彥邦、亦傑、恩翔陪我或幫助我修這門課。最後還要特別感謝在最後拉我一把的黃鉦鈞。**
