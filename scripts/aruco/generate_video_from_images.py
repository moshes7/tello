import os
import cv2
import numpy as np

# --- parameters
max_frame_num = None
# max_frame_num = 300

images_dir = r'C:\Users\Moshe\Sync\Projects\tello\images\tello_stream\2019-06-12_23.49.29 _First_Good_Tracking_With_Azimuth'
output_dir = os.path.join(images_dir, 'video')
os.makedirs(output_dir, exist_ok=True)
output_filename = 'video.mp4'
output_full_file = os.path.join(output_dir, output_filename)

# get images list
image_list = [os.path.join(images_dir, fn) for fn in os.listdir(images_dir) if fn.endswith('.jpg')]

if max_frame_num is not None:
    image_list = image_list[:max_frame_num]

N = len(image_list)

# extract image dimensions
img = cv2.imread(image_list[0])
height , width , layers =  img.shape

# instanciate video object
# fps = 20 # [Hz]
fps = 10 # [Hz]
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter(output_full_file,fourcc,fps,(width,height))

for n, img_name in enumerate(image_list):
    img = cv2.imread(img_name)
    video.write(img)

    if np.mod(n, 20) == 0:
        print('{}/{}'.format(n, N))


cv2.destroyAllWindows()
video.release()

print('Done!')