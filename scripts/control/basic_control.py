"""
This demo calculates multiple things for different scenarios.

Here are the defined reference frames:

TAG:
                A y
                |
                |
                |tag center
                O---------> x
               z
CAMERA:
               z
                X--------> x
                | frame center
                |
                |
                V y

DRONE:
               x
                X--------> y
                | frame center
                |
                |
                V z

F1: Flipped (180 deg) tag frame around x axis
F2: Flipped (180 deg) camera frame around x axis

The attitude of a generic frame 2 respect to a frame 1 can obtained by calculating euler(R_21.T)

We are going to obtain the following quantities:
    > from aruco library we obtain tvec and Rct, position of the tag in camera frame and attitude of the tag
    > position of the Camera in Tag axis: -R_ct.T*tvec
    > Transformation of the camera, respect to f1 (the tag flipped frame): R_cf1 = R_ct*R_tf1 = R_cf*R_f
    > Transformation of the tag, respect to f2 (the camera flipped frame): R_tf2 = Rtc*R_cf2 = R_tc*R_f
    > R_tf1 = R_cf2 an symmetric = R_f


"""

import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math, os
import datetime
from djitellopy import Tello
from simple_pid import PID

# define pid controllers
# camera reference frame:
#   x: right
#   y: down
#   z: forward
SPEED = 50
sleepTime = 0.5
x_ref = 100
y_ref = 0
z_ref = 0
az_ref = 0

TIME_BTW_COMMANDS = 0.25  # [sec]

pid_x = PID(Kp=1., Ki=0, Kd=0.1, setpoint=x_ref, sample_time=TIME_BTW_COMMANDS, output_limits=(-100, 100), auto_mode=True, proportional_on_measurement=False)
pid_y = PID(Kp=0.75, Ki=0.0, Kd=0.2, setpoint=y_ref, sample_time=TIME_BTW_COMMANDS, output_limits=(-100, 100), auto_mode=True, proportional_on_measurement=False)
pid_z = PID(Kp=0.75, Ki=0.0, Kd=0.2, setpoint=z_ref, sample_time=TIME_BTW_COMMANDS, output_limits=(-100, 100), auto_mode=True, proportional_on_measurement=False)
pid_az = PID(Kp=0.75, Ki=0.0, Kd=0.2, setpoint=az_ref, sample_time=TIME_BTW_COMMANDS, output_limits=(-45, 45), auto_mode=True, proportional_on_measurement=False)

# --- Define Tag
id_to_find = 0
marker_size = 15  # - [cm]

read_images_from_dir = False
input_images_dir = r'C:\Users\Moshe\Sync\Projects\tello\images\tello_stream\2019-06-10_19.31.23_with_raw_images\raw'
save_frames = True
save_frames_raw = True# save raw frames, without markers and text
output_dir = r'C:\Users\Moshe\Sync\Projects\tello\images\tello_stream'
time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
output_dir = os.path.join(output_dir, time_str)
os.makedirs(output_dir, exist_ok=True)

if save_frames_raw:
    output_dir_raw = os.path.join(output_dir, 'raw')
    os.makedirs(output_dir_raw, exist_ok=True)

# ------------------------------------------------------------------------------
# ------- ROTATIONS https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# ------------------------------------------------------------------------------
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
# R: yaw -> pitch -> roll
# Rt: roll -> pitch -> yaw
def eulerAnglesToRotationMatrix(roll, pitch, yaw, degrees=True):

    if degrees:
        # convert to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]
                    ])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]
                    ])

    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]
                    ])

    # R = np.dot(R_z, np.dot(R_y, R_x))
    R = np.dot(R_x, np.dot(R_y, R_z))
    Rt = np.transpose(R)

    return R, Rt

# --- Get the camera calibration path
calib_path = r'C:\Users\Moshe\Sync\Projects\tello\images\calibration_camera1/'
camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

# --- 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# R_tag_to_drone, R_drone_to_tag = eulerAnglesToRotationMatrix(roll=0, pitch=-90, yaw=90)
# R_camera_to_tag, R_tag_to_camera = eulerAnglesToRotationMatrix(roll=0, pitch=0, yaw=180)
#
# R_camera_to_drone = R_tag_to_drone * R_camera_to_tag

# roll, pitch, yaw = rotationMatrixToEulerAngles(R_camera_to_tag) # sanity check
# roll, pitch, yaw = rotationMatrixToEulerAngles(R_camera_to_drone) # sanity check
# R_flip = R_tag_to_camera # FIXME!

# --- Define the aruco dictionary
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
parameters = aruco.DetectorParameters_create()

# --- Capture the videocamera (this may also be a video or a picture)
# cap = cv2.VideoCapture(0)
# # -- Set the camera size as the one it was calibrated with
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


if read_images_from_dir:
    images_list = [os.path.join(input_images_dir, image_name) for image_name in os.listdir(input_images_dir) if image_name.endswith('jpg')]
else:
    # initializations
    tello = Tello()

    ok = tello.connect()

    # In case streaming is on. This happens when we quit this program without the escape key.
    ok = tello.streamoff()

    ok = tello.streamon()

    ok = tello.set_speed(SPEED)

    frame_read = tello.get_frame_read()

    time.sleep(3)


    ok = tello.takeoff()

    time.sleep(3)


    time_received_last_command = time.time()


# tello.takeoff()
# time.sleep(5)
# tello.move_up(50)


# plt.figure()
# img = np.zeros((720,960,3))
# h = plt.imshow(img)
# plt.show(block=False)

# -- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN
font_size = 1.2

notDone = True
t0 = time.time()
t_prev = t0
frame_num = 0

frame_vec = []
time_vec = []
dt_vec = []

while notDone:

    # -- Read the camera frame
    if read_images_from_dir:

        if frame_num >= len(images_list):
            notDone = False

        frame_num = np.mod(frame_num, len(images_list))

        image_name = images_list[frame_num]
        frame = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    else:
        # ret, frame = cap.read()
        frame = frame_read.frame

    # raw frame
    frame_raw = np.copy(frame)

    # -- Convert in gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # -- remember, OpenCV stores color images in Blue, Green, Red

    # -- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                 cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    if ids is not None and ids[0] == id_to_find:
        # -- ret = [rvec, tvec, ?]
        # -- array of rotation and position of each marker in camera frame
        # -- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
        # -- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        # -- Unpack the output, get only the first
        # rvec: Rodrigues parameters
        # tvec: translation vector
        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        # -- Draw the detected marker and put a reference frame over it
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

        # -- Print the tag position in camera frame
        # cv2.putText(frame, str_position, (0, 100), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

        # -- Obtain the rotation matrix tag->camera
        R_ct = np.matrix(cv2.Rodrigues(rvec)[0]) # camera->tag
        R_tc = R_ct.T # tag->camera

        # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
        # roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

        # -- Print the marker's attitude respect to camera frame
        # str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
        # math.degrees(roll_marker), math.degrees(pitch_marker),
        # math.degrees(yaw_marker))
        # cv2.putText(frame, str_attitude, (0, 150), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

        # -- Now get Position and attitude f the camera respect to the marker
        pos_camera = -R_tc * np.matrix(tvec).T

        str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (pos_camera[0], pos_camera[1], pos_camera[2])
        cv2.putText(frame, str_position, (0, 200), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

        # -- Get the attitude of the camera respect to the frame
        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
        str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
            math.degrees(roll_camera), math.degrees(pitch_camera),
            math.degrees(yaw_camera))
        cv2.putText(frame, str_attitude, (0, 250), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

        # # drone position
        # pos_drone = R_camera_to_drone * R_tc * (-1 * np.matrix(tvec).T)
        #
        # str_position = "DRONE Position x=%4.0f  y=%4.0f  z=%4.0f" % (pos_drone[0], pos_drone[1], pos_drone[2])
        # cv2.putText(frame, str_position, (0, 300), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
        #
        # # -- Get the attitude of the drone respect to the frame
        # roll_drone, pitch_drone, yaw_drone= rotationMatrixToEulerAngles(R_tag_to_drone * R_ct)
        # str_attitude = "Drone Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
        #     math.degrees(roll_drone), math.degrees(pitch_drone), math.degrees(yaw_drone))
        # cv2.putText(frame, str_attitude, (0, 350), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

        # control
        # x_command = -1 * np.asarray(pid_x(pos_camera[2])).squeeze()
        x_command = 1 * np.asarray(pid_x(pos_camera[2])).squeeze()
        # x_command = np.asarray(pid_x(pos_camera[0])).squeeze()
        # y_command = 0# -1 *np.asarray(pid_y(pos_camera[0])).squeeze()
        y_command = 1 *np.asarray(pid_y(pos_camera[0])).squeeze()
        # z_command = 0 #np.asarray(pid_z(pos_camera[2])).squeeze()
        z_command = 1 * np.asarray(pid_z(pos_camera[2])).squeeze()
        # z_command = -1 * np.asarray(pid_z(pos_camera[2])).astype(int)
        az_command = 0 #np.asarray(pid_az(math.degrees(pitch_camera))).squeeze() # camera pitch corresponds to azimuth

        str_attitude = "Control Reference: x=%4.0f  y=%4.0f  z=%4.0f  az=%4.0f" % (
            x_ref, y_ref, z_ref, az_ref)
        cv2.putText(frame, str_attitude, (0, 400), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

        str_attitude = "Control Commands: x=%4.0f  y=%4.0f  z=%4.0f  az=%4.0f" % (
            x_command, y_command, z_command, az_command)
        cv2.putText(frame, str_attitude, (0, 450), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

        if not read_images_from_dir:

            if (time.time() - time_received_last_command) >= TIME_BTW_COMMANDS:

                # speed_sign = 1 if x_command >= 0 else -1
                # x_command = abs(x_command)
                # tello.send_command_without_return('go {} {} {} {}'.format(x_command, 0, 0, speed_sign * SPEED))

                left_right = y_command
                forward_backward = x_command
                up_down = z_command
                az = az_command

                # tello.send_command_without_return('rc {} {} {} {}'.format(left_right, forward_backward, up_down, az))
                tello.send_rc_control(left_right, forward_backward, up_down, az)

                # if z_command > 0:
                #     tello.move_forward(z_command)
                # else:
                #     tello.move_back(-z_command)

                # FPS = 25
                # time.sleep(1 / FPS)

    # --- Display the frame
    cv2.imshow('frame', frame)

    if save_frames:
        img_name = os.path.join(output_dir, '{0:07d}.jpg'.format(frame_num))
        cv2.imwrite(img_name, frame)

    if save_frames_raw:
        img_name_raw = os.path.join(output_dir_raw, '{0:07d}.jpg'.format(frame_num))
        cv2.imwrite(img_name_raw, frame_raw)


    # save timing
    t_new = time.time()
    dt = (t_new - t_prev) * 1000 # [sec] -> [ms]

    frame_vec.append(frame_num)
    time_vec.append(t_new - t0)
    dt_vec.append(dt)


    frame_num += 1
    t_prev = t_new

    # --- use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        notDone = False
        # cap.release()
        cv2.destroyAllWindows()

        if not read_images_from_dir:
            time.sleep(0.5)
            tello.land()
            time.sleep(5)
            tello.end()

        break

# save times to file
frame_vec = np.asarray(frame_vec)
time_vec = np.asarray(time_vec)
dt_vec = np.asarray(dt_vec)
data_mat = np.stack((frame_vec, time_vec, dt_vec), axis=1)
filename = os.path.join(output_dir, 'timings.txt')
np.savetxt(filename, data_mat, fmt=['%08.8d', '%06.5f', '%05.2f'], delimiter='  , ', header='frame   , time [sec], dt [ms]')

print('Done!')