import RPi.GPIO as GPIO
import time

# Inputs
TOGGLE = 29
BUTTON = 31

OUTPUT = 33

GPIO.setmode(GPIO.BOARD)

GPIO.setup(TOGGLE, GPIO.IN)
GPIO.setup(BUTTON, GPIO.IN)

GPIO.setup(OUTPUT, GPIO.OUT)

pan_left = 0
pan_right = 0

# time.sleep(0.1)

def enable_int():
    GPIO.add_event_detect(TOGGLE, GPIO.FALLING, callback=toggle_cb, bouncetime=200)
    GPIO.add_event_detect(BUTTON, GPIO.BOTH, callback=button_cb, bouncetime=200)

def disable_int():
    GPIO.remove_event_detect(TOGGLE)
    GPIO.remove_event_detect(BUTTON)

def toggle_cb(channel):
    print("Rising edge detected")
    print(GPIO.input(channel))
    
def button_cb(channel):
    print("Button callback")
    if channel == BUTTON:
        global pan_left
        if GPIO.input(channel):
            pan_left = 1
        else:
            pan_left = 0
        GPIO.output(OUTPUT, GPIO.input(channel))
            
def photo_cb(channel):
    disable_int()
    photo_flag = 1
            
    
input("Press Enter when ready \n")

enable_int()

count = 0
while(True):
    if pan_left:
        count = count + 1
        time.sleep(0.2)
        
    #else:
        #print(count)

