from djitellopy import Tello
import cv2
import numpy as np
import matplotlib.pyplot as plt

# initializations
tello = Tello()

ok = tello.connect()

# In case streaming is on. This happens when we quit this program without the escape key.
ok = tello.streamoff()

ok = tello.streamon()

frame_read = tello.get_frame_read()

plt.figure()
img = np.zeros((720,960,3))
h = plt.imshow(img)
plt.show(block=False)
iterations = 1000

counter = 0
while counter < iterations:
    counter += 1
    frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    # frame = np.rot90(frame)
    # frame = np.flipud(frame)
    # plt.imshow(frame)
    h.set_data(frame)
    plt.draw()
    plt.pause(0.001)


print('Done!')