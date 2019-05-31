from djitellopy import Tello
import time

# parameters
step = 100
sleepTime = 8

tello = Tello()

tello.connect()

tello.takeoff()
time.sleep(sleepTime)

tello.move_forward(step)
time.sleep(sleepTime)

tello.move_right(step)
time.sleep(sleepTime)

tello.move_back(step)
time.sleep(sleepTime)

tello.move_left(step)
time.sleep(sleepTime)

tello.land()
time.sleep(sleepTime)

tello.end()