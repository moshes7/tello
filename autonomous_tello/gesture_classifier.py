from __future__ import print_function, division, unicode_literals, absolute_import

from fastai.vision import *
import threading

class GestureClassifier(object):

    def __init__(self, model_path='./gesture_classifier/export.pkl', q=None):

        self.lock = threading.Lock()
        self.image_ready = False
        self.prediction_ready= False
        self.q = q

        # load model
        self.model = load_learner(model_path)

    def predict(self, img):

        pred_class, pred_idx, outputs = self.model.predict(img)

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

