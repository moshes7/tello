from __future__ import print_function, division, unicode_literals, absolute_import

import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math, os
import datetime
from djitellopy import Tello
from simple_pid import PID
from functools import reduce
import configparser
from code.utils import euler2mat, eulerAnglesToRotationMatrix, isRotationMatrix, mat2euler

class AutoTello(object):

    """
    Autonums Tello class
    """

    def __init__(self, config_file='../config/config_default.ini'):

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
        self.camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
        self.camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

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

        if bool(config['general']['save_frames_raw']):
            output_dir_raw = os.path.join(output_dir, 'raw')
            os.makedirs(output_dir_raw, exist_ok=True)
            self.output_dir_raw = output_dir_raw


    def connect(self):
        pass

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


class SimpleClass(object):
    """ dummy class"""
    def __init__(self):
        pass


if __name__ == '__main__':

    at = AutoTello()

    print('Done!')