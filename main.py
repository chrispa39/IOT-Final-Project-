
import RPi.GPIO as GPIO
import time
import random
import board
import adafruit_dht
#Emotion_list[random_EmotionNumber-1] temperature_c humidity
#Get emotion data
def program():
    
    Emotion_list = ["happy", "surprise", "neutral", "sad", "angry"]
    li = []
    
    random_EmotionNumber = random.randint(1,5)
    print(random_EmotionNumber)
    print(Emotion_list[random_EmotionNumber-1])

    #Get humidity and temperature data
    dhtDevice = adafruit_dht.DHT22(board.D16, use_pulseio=False) #D16 = BCM16

    try:
        # Print the values to the serial port
        temperature_c = dhtDevice.temperature
        temperature_f = temperature_c * (9 / 5) + 32
        humidity = dhtDevice.humidity
        print("Temp: {:.1f} C    Humidity: {}% "
                .format(temperature_c, humidity))
    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        temperature_c = 20
        humidity = 80
        print("Temp: 20 C    Humidity: 80% ")
     
    time.sleep(2.0)
    #single_message = str(temperature_c) + str(humidity) + str(Emotion_list[random_EmotionNumber-1])
    li.append(str(temperature_c))
    li.append(str(humidity))
    li.append(str(Emotion_list[random_EmotionNumber-1]))
    #Caculate the point
    #1. temperature point
    if temperature_c > 20:
        t_point = 2
    else:
        t_point = -2
    #2. humidity point
    if humidity > 80:
        h_point = 2
    else:
        h_point = -2
    #3. total point
    total_point = t_point*0.25 + h_point*0.25 + random_EmotionNumber*0.5
    print("total point is: ",total_point)

    if total_point <= 1.5:
        CONTROL_PIN = 17 #sweet
        result = "sweet"
    else:
        CONTROL_PIN = 27 #salt
        result = "salt"
    
    li.append(str(result))
    PWM_FREQ = 50
    STEP=15
     
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(CONTROL_PIN, GPIO.OUT)
     
    pwm = GPIO.PWM(CONTROL_PIN, PWM_FREQ)
    pwm.start(0)
     
    def angle_to_duty_cycle(angle=0):
        duty_cycle = (0.05 * PWM_FREQ) + (0.19 * PWM_FREQ * angle / 180)
        return duty_cycle
     
    try:
        print('按下 Ctrl-C 可停止程式')
        for angle in range(0, 60, STEP):
            dc = angle_to_duty_cycle(angle)
            pwm.ChangeDutyCycle(dc)
            print('角度={: >3}, 工作週期={:.2f}'.format(angle, dc))
            time.sleep(2)
        for angle in range(60, 0, -STEP):
            dc = angle_to_duty_cycle(angle)
            print('角度={: >3}, 工作週期={:.2f}'.format(angle, dc))
            pwm.ChangeDutyCycle(dc)
            time.sleep(2)
        pwm.ChangeDutyCycle(angle_to_duty_cycle(0))
    except KeyboardInterrupt:
        print('關閉程式')
    finally:
        pwm.stop()
        GPIO.cleanup()
        
    return li

