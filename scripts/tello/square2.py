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

tello.rotate_clockwise(90)
time.sleep(sleepTime)

tello.move_forward(step)
time.sleep(sleepTime)

tello.rotate_clockwise(90)
time.sleep(sleepTime)

tello.move_forward(step)
time.sleep(sleepTime)

tello.rotate_clockwise(90)
time.sleep(sleepTime)

tello.move_forward(step)
time.sleep(sleepTime)

tello.rotate_clockwise(90)
time.sleep(sleepTime)

tello.land()
time.sleep(sleepTime)

tello.end()