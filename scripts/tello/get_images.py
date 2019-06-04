from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello

# --- parameters
save_images = True
output_folder = r'C:\Users\Moshe\Sync\Projects\tello\images\camera1'

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
iterations = 300

counter = 0
frame_num = 0
while counter < iterations:
    counter += 1
    frame = frame_read.frame
    frame_rgb = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    # frame = np.rot90(frame)
    # frame = np.flipud(frame)
    # plt.imshow(frame)

    if save_images:
        print('saving frame {}', frame_num)
        img_name = os.path.join(output_folder, '{0:03d}.jpg'.format(frame_num))
        cv2.imwrite(img_name, frame)
        frame_num += 1
        print('saving frame {}')

    h.set_data(frame_rgb)
    plt.draw()
    plt.pause(0.001)


print('Done!')