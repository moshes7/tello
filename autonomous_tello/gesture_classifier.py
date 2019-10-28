from __future__ import print_function, division, unicode_literals, absolute_import

from fastai.vision import *
import threading
from queue import Queue

class GestureClassifier(object):

    def __init__(self, model_path='./gesture_control', q=None):

        self.lock = threading.Lock()
        self.image_ready = False
        self.prediction_ready= False
        self.q = Queue()

        # load model
        self.model = load_learner(model_path)

    def predict(self, img):

        t = torch.tensor(np.ascontiguousarray(np.flip(img, 2)).transpose(2,0,1)).float()/255
        t = Image(t) # fastai.vision.Image, not PIL.Image
        pred_class, pred_idx, outputs = self.model.predict(t)

        return pred_class

    def main(self):

        while True:

            if self.image_ready:

                with self.lock:
                    self.image_ready = False
                    self.prediction_ready = False
                    img = self.q.get()

                pred = self.predict(img)

                with self.lock:
                    self.q.put(pred)
                    self.prediction_ready = True

