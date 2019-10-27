
import os
import configparser
import datetime
import numpy as np
from queue import Queue
import threading
import time
import cv2
from djitellopy import Tello
from autonomous_tello.gesture_classifier import GestureClassifier

class TelloGestureControl(object):

    def __init__(self, config_file='./gesture_control/config_default.ini'):

        # read config file
        config = configparser.ConfigParser()
        config.read(config_file)
        self.config = config

        # FIXME: read parameters from config instead of hard-coding
        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 30
        self.momentum = 0.9

        self.send_rc_control = False

        # initialize gesture classifier
        self.model_path = './gesture_classifier/export.pkl' # FIXME: read parameters from config instead of hard-coding
        self.gesture_classifier = GestureClassifier(model_path=self.model_path)

        # takeoff flag
        self.takeoff = config.getboolean('general', 'takeoff')# tello will take off only when True

        # Get the camera calibration path
        calib_path = config['general']['calib_path']
        self.camera_matrix = np.loadtxt(os.path.join(calib_path, 'cameraMatrix.txt'), delimiter=',')
        self.camera_distortion = np.loadtxt(os.path.join(calib_path, 'cameraDistortion.txt'), delimiter=',')

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

        if not tello.set_speed(int(self.config['pid']['velocity'])):
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

        if self.takeoff:
            time.sleep(3)
            ok = tello.takeoff()
            time.sleep(3)

        self.time_received_last_command = time.time()

        self.tello = tello
        self.frame_read = frame_read


    def class2control(self, class_pred):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if class_pred == 'triangle':  # set forward velocity
            self.for_back_velocity = self.speed
        elif class_pred == 'stop':  # set backward velocity
            self.for_back_velocity = -self.speed
        elif class_pred == 'left':  # set left velocity
            self.left_right_velocity = -self.speed
        elif class_pred == 'right':  # set right velocity
            self.left_right_velocity = self.speed
        # elif key == pygame.K_w:  # set up velocity
        #     self.up_down_velocity = S
        # elif key == pygame.K_s:  # set down velocity
        #     self.up_down_velocity = -S
        # elif key == pygame.K_a:  # set yaw clockwise velocity
        #     self.yaw_velocity = -S
        # elif key == pygame.K_d:  # set yaw counter clockwise velocity
        #     self.yaw_velocity = S

    def update_velocity(self):
        self.for_back_velocity *= self.momentum
        self.left_right_velocity *= self.momentum

    def send_control_command(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)



    def main(self):

        # unpack needed variables
        tello = self.tello
        read_images_from_dir = self.config.getboolean('general', 'read_images_from_dir')
        momentum = float(self.config['pid']['momentum'])
        time_between_commands = float(self.config['pid']['time_between_commands'])
        output_dir = self.output_dir
        save_frames = self.config.getboolean('general', 'save_frames')
        save_frames_raw = self.config.getboolean('general', 'save_frames_raw')
        display = self.config.getboolean('general', 'display')

        new_pred = False

        if save_frames_raw:
            output_dir_raw = self.output_dir_raw

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

            if not self.gesture_classifier.image_ready:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.gesture_classifier.lock:
                    self.gesture_classifier.q.put(img)
                    self.gesture_classifier.image_ready = True
                    self.gesture_classifier.prediction_ready = False

            if self.gesture_classifier.prediction_ready:
                with self.gesture_classifier.lock:
                    pred = self.gesture_classifier.q.get()
                    new_pred = True


            if new_pred:
                self.gesture_classifier.image_ready = False
                self.class2control(pred)
                pred_counter = 0

                str_prediction = 'Prediction: {}'.format(pred)
                cv2.putText(frame, str_prediction, (0, 150), font, font_size, (0, 0, 255), 2, cv2.LINE_AA)
                str_prediction_counter = 'Frames from last redictetion: {}'.format(pred_counter)
                cv2.putText(frame, str_prediction_counter, (0, 150), font, font_size, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                self.update_velocity()
                pred_counter += 1
                str_prediction = 'Prediction from : {}'.format(pred)
                cv2.putText(frame, str_prediction, (0, 150), font, font_size, (255, 0, 0), 2, cv2.LINE_AA)
                str_prediction_counter = 'Frames from last redictetion: {}'.format(pred_counter)
                cv2.putText(frame, str_prediction_counter, (0, 150), font, font_size, (255, 0, 0), 2, cv2.LINE_AA)


            str_command = r'Right={}'.format(min(self.left_right_velocity, 0))
            cv2.putText(frame, str_command, (0, 250), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
            str_command = r'Left={}'.format(max(self.left_right_velocity, 0))
            cv2.putText(frame, str_command, (0, 300), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
            str_command = r'Up={}'.format(min(self.up_down_velocity, 0))
            cv2.putText(frame, str_command, (0, 350), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
            str_command = r'Down={}'.format(max(self.up_down_velocity, 0))
            cv2.putText(frame, str_command, (0, 400), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)

            if not read_images_from_dir:

                if (time.time() - time_received_last_command) >= time_between_commands:

                    self.send_control_command()

                    if pred == 'x':
                        time.sleep(0.5)
                        tello.land()
                        time.sleep(5)


            # --- Display the frame
            if display:
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


def run_single_tello_with_gesture_classifer():

    print('run_single_tello_with_gesture_classifier: start')
    config_file = r'./gesture_classifier/config_default.ini' # BE

    tello = TelloGestureControl(config_file=config_file)

    if tello.config.getboolean('general', 'read_images_from_dir'):
        input_images_dir = tello.config['general']['input_images_dir']
        images_list = [os.path.join(input_images_dir, image_name) for image_name in os.listdir(input_images_dir) if image_name.endswith('jpg')]
    else:
        tello.connect()


    # tello main thread
    thread1 = threading.Thread(target=tello.main(), args=(), daemon=True)
    thread1.start()

    # classifier thread
    thread2 = threading.Thread(target=tello.gesture_classifier.main(), args=(), daemon=True)
    thread2.start()


    print('run_single_tello_with_gesture_classifier: end')


if __name__ == '__main__':
    run_single_tello_with_gesture_classifer()
    # run_multiple_tellos()

    print('Done!')