from __future__ import print_function, division, unicode_literals, absolute_import

import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math, os
import datetime
from djitellopy import Tello
from simple_pid import PID
import configparser
from autonomous_tello.utils import rotationMatrixToEulerAngles, euler2mat, mat2euler

class AutoTello(object):

    """
    Autonomous Tello class
    """

    def __init__(self, config_file='./config/config_default.ini'):

        # read config file
        config = configparser.ConfigParser()
        config.read(config_file)
        self.config = config

        # set PID parameters
        # camera reference frame:
        #   x: right
        #   y: down
        #   z: forward
        self.pid_x = PID(Kp=float(config['pid']['kp_x']), Ki=float(config['pid']['ki_x']), Kd=float(config['pid']['kd_x']),
                         setpoint=float(config['control_reference']['x']), sample_time=float(config['pid']['time_between_commands']),
                         output_limits=(- float(config['pid']['velocity']), float(config['pid']['velocity'])),
                         auto_mode=True, proportional_on_measurement=False)

        self.pid_y = PID(Kp=float(config['pid']['kp_y']), Ki=float(config['pid']['ki_y']), Kd=float(config['pid']['kd_y']),
                         setpoint=float(config['control_reference']['y']), sample_time=float(config['pid']['time_between_commands']),
                         output_limits=(-float(config['pid']['velocity']), float(config['pid']['velocity'])),
                         auto_mode=True, proportional_on_measurement=False)

        self.pid_z = PID(Kp=float(config['pid']['kp_z']), Ki=float(config['pid']['ki_z']), Kd=float(config['pid']['kd_z']),
                         setpoint=float(config['control_reference']['z']), sample_time=float(config['pid']['time_between_commands']),
                         output_limits=(-float(config['pid']['velocity']), float(config['pid']['velocity'])),
                         auto_mode=True, proportional_on_measurement=False)

        self.pid_az = PID(Kp=float(config['pid']['kp_az']), Ki=float(config['pid']['ki_az']), Kd=float(config['pid']['kd_az']),
                         setpoint=float(config['control_reference']['az']), sample_time=float(config['pid']['time_between_commands']),
                         output_limits=(-float(config['pid']['angular_velocity']), float(config['pid']['angular_velocity'])),
                         auto_mode=True, proportional_on_measurement=False)


        # Get the camera calibration path
        calib_path = config['general']['calib_path']
        self.camera_matrix = np.loadtxt(os.path.join(calib_path, 'cameraMatrix.txt'), delimiter=',')
        self.camera_distortion = np.loadtxt(os.path.join(calib_path, 'cameraDistortion.txt'), delimiter=',')

        # --- Define the aruco dictionary
        # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_parameters = aruco.DetectorParameters_create()

        # set rotation matrices
        R_flip, R_tag_to_drone = self.set_rotation_matrices()
        self.R_flip = R_flip
        self.R_tag_to_drone = R_tag_to_drone

        # set output directory
        output_dir = config['general']['output_dir']
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        output_dir = os.path.join(output_dir, time_str)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        if config.getboolean('general', 'save_frames_raw'):
            output_dir_raw = os.path.join(output_dir, 'raw')
            os.makedirs(output_dir_raw, exist_ok=True)
            self.output_dir_raw = output_dir_raw


    def connect(self):

        ip_address = self.config['general']['ip_address']
        tello = Tello(ip_address)
        # tello = Tello('192.168.10.2'); y_ref = +30
        # tello = Tello('192.168.10.4'); y_ref = -30

        if not tello.connect():
            print("Tello not connected")
            exit(1)

        if not tello.set_speed(self.config['pid']['velocity']):
            print("Not set speed to lowest possible")
            exit(1)

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not tello.streamoff():
            print("Could not stop video stream")
            exit(1)

        if not tello.streamon():
            print("Could not start video stream")
            exit(1)

        frame_read = tello.get_frame_read()

        # ok = tello.connect()
        #
        # # In case streaming is on. This happens when we quit this program without the escape key.
        # ok = tello.streamoff()
        #
        # ok = tello.streamon()
        #
        # ok = tello.set_speed(SPEED)
        #
        # frame_read = tello.get_frame_read()

        time.sleep(3)

        ok = tello.takeoff()

        time.sleep(3)

        self.time_received_last_command = time.time()

        self.tello = tello
        self.frame_read = frame_read


    @staticmethod
    def set_rotation_matrices():
        #180 deg rotation matrix around the x axis
        R_flip = np.zeros((3, 3), dtype=np.float32)
        R_flip[0, 0] = 1.0
        R_flip[1, 1] = -1.0
        R_flip[2, 2] = -1.0

        yaw, pitch, roll = mat2euler(R_flip, degrees=True)

        # R_tag_to_drone, R_drone_to_tag = eulerAnglesToRotationMatrix(roll=0, pitch=-90, yaw=90)
        # R_camera_to_tag, R_tag_to_camera = eulerAnglesToRotationMatrix(roll=0, pitch=0, yaw=180)
        #
        # R_camera_to_drone = R_tag_to_drone * R_camera_to_tag

        # roll, pitch, yaw = rotationMatrixToEulerAngles(R_camera_to_tag) # sanity check
        # roll, pitch, yaw = rotationMatrixToEulerAngles(R_camera_to_drone) # sanity check
        R_tag_to_drone = euler2mat(z=-90, y=90, x=0, degrees=True)

        # R_flip = R_tag_to_camera # FIXME!

        return R_flip, R_tag_to_drone

    def main(self):

        # unpack needed variables
        read_images_from_dir = self.config.getboolean('general', 'read_images_from_dir')
        id_to_find = int(self.config['aruco_marker']['id'])
        marker_size = int(self.config['aruco_marker']['marker_size'])
        x_ref = float(self.config['control_reference']['x'])
        y_ref = float(self.config['control_reference']['y'])
        z_ref = float(self.config['control_reference']['z'])
        az_ref = float(self.config['control_reference']['az'])
        momentum = float(self.config['pid']['momentum'])
        time_between_commands = float(self.config['pid']['time_between_commands'])
        output_dir = self.config['general']['output_dir']
        save_frames = self.config.getboolean('general', 'save_frames')
        save_frames_raw = self.config.getboolean('general', 'save_frames_raw')

        if save_frames_raw:
            output_dir_raw = self.self.output_dir_raw

        if not read_images_from_dir:
            frame_read = self.frame_read
            time_received_last_command = self.time_received_last_command

        # -- Font for the text in the image
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 1.2

        notDone = True
        t0 = time.time()
        t_prev = t0
        frame_num = 0

        x_command = 0
        y_command = 0
        z_command = 0
        az_command = 0

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
            corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=self.aruco_dict, parameters=self.aruco_parameters,
                                                         cameraMatrix=self.camera_matrix, distCoeff=self.camera_distortion)

            if ids is not None and ids[0] == id_to_find:
                # -- ret = [rvec, tvec, ?]
                # -- array of rotation and position of each marker in camera frame
                # -- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
                # -- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
                ret = aruco.estimatePoseSingleMarkers(corners, marker_size, self.camera_matrix, self.camera_distortion)

                # -- Unpack the output, get only the first
                # rvec: Rodrigues parameters
                # tvec: translation vector
                rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

                # -- Draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(frame, corners)
                aruco.drawAxis(frame, self.camera_matrix, self.camera_distortion, rvec, tvec, 10)

                # -- Print the tag position in camera frame
                # cv2.putText(frame, str_position, (0, 100), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Obtain the rotation matrix tag->camera
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])  # camera->tag
                R_tc = R_ct.T  # tag->camera

                # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
                # roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

                # -- Print the marker's attitude respect to camera frame
                # str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                # math.degrees(roll_marker), math.degrees(pitch_marker),
                # math.degrees(yaw_marker))
                # cv2.putText(frame, str_attitude, (0, 150), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Now get Position and attitude f the camera respect to the marker
                pos_camera = R_tc * np.matrix(tvec).T

                str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (
                pos_camera[0], pos_camera[1], pos_camera[2])
                cv2.putText(frame, str_position, (0, 200), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Get the attitude of the camera respect to the frame
                # roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
                roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(self.R_flip * R_tc)
                roll_camera, pitch_camera, yaw_camera = mat2euler(self.R_flip * R_tc)
                # roll_camera, pitch_camera, yaw_camera = mat2euler(R_flip * R_tc)

                str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                    math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera))
                cv2.putText(frame, str_attitude, (0, 250), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

                # # drone position
                pos_drone = self.R_tag_to_drone * (1 * np.matrix(tvec).T)
                #
                str_position = "DRONE Position x=%4.0f  y=%4.0f  z=%4.0f" % (pos_drone[0], pos_drone[1], pos_drone[2])
                cv2.putText(frame, str_position, (0, 300), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
                #
                # -- Get the attitude of the drone respect to the frame
                yaw_drone, pitch_drone, roll_drone = mat2euler(self.R_tag_to_drone * R_ct)
                str_attitude = "Drone Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (roll_drone, pitch_drone, yaw_drone)
                cv2.putText(frame, str_attitude, (0, 350), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

                # control

                # use pos_camera
                # x_command = -1 * np.asarray(pid_x(pos_camera[2])).squeeze()
                # y_command = 1 *np.asarray(pid_y(pos_camera[0])).squeeze()
                # z_command = - 1 * np.asarray(pid_z(pos_camera[1])).squeeze()

                # use pos_drone

                x_command = - 1 * np.asarray(self.pid_x(pos_drone[0])).squeeze()
                y_command = 1 * np.asarray(self.pid_y(pos_drone[1])).squeeze()
                z_command = - 1 * np.asarray(self.pid_z(pos_drone[2])).squeeze()
                # x_command = 0
                # y_command = 0
                # z_command = 0

                az = np.degrees(np.arctan2(- pos_drone[1], pos_drone[0]))
                # az_command = 0
                az_command = -1 * np.asarray(self.pid_az(az)).squeeze()  # camera pitch corresponds to azimuth

                str_attitude = "Azimuth: az=%4.0f" % (az)
                cv2.putText(frame, str_attitude, (0, 500), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                # x_command = 0
                # y_command = 0
                # z_command = 0
                # az_command = 0
                x_command = momentum * x_command
                y_command = momentum * y_command
                z_command = momentum * z_command
                az_command = momentum * az_command

            str_attitude = "Control Reference: x=%4.0f  y=%4.0f  z=%4.0f  az=%4.0f" % (
                x_ref, y_ref, z_ref, az_ref)
            cv2.putText(frame, str_attitude, (0, 400), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

            str_attitude = "Control Commands: x=%4.0f  y=%4.0f  z=%4.0f  az=%4.0f" % (
                x_command, y_command, z_command, az_command)
            cv2.putText(frame, str_attitude, (0, 450), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

            if not read_images_from_dir:

                if (time.time() - time_received_last_command) >= time_between_commands:
                    # speed_sign = 1 if x_command >= 0 else -1
                    # x_command = abs(x_command)
                    # tello.send_command_without_return('go {} {} {} {}'.format(x_command, 0, 0, speed_sign * SPEED))

                    left_right = int(y_command)
                    forward_backward = int(x_command)
                    up_down = int(z_command)
                    az = int(az_command)

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
            dt = (t_new - t_prev) * 1000  # [sec] -> [ms]

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
        np.savetxt(filename, data_mat, fmt=['%08.8d', '%06.5f', '%05.2f'], delimiter='  , ',
                   header='frame   , time [sec], dt [ms]')


class SimpleClass(object):
    """ dummy class"""
    def __init__(self):
        pass


if __name__ == '__main__':

    tello = AutoTello()

    if tello.config.getboolean('general', 'read_images_from_dir'):
        input_images_dir = tello.config['general']['input_images_dir']
        images_list = [os.path.join(input_images_dir, image_name) for image_name in os.listdir(input_images_dir) if image_name.endswith('jpg')]
    else:
        tello.connect()

    tello.main()

    print('Done!')