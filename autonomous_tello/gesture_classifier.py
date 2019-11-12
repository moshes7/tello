from __future__ import print_function, division, unicode_literals, absolute_import

from fastai.vision import *
import threading
from queue import Queue

class GestureClassifier(object):

    def __init__(self, model_path='./gesture_control', q=None):

        self.lock = threading.Lock()
        self.q_images = Queue(maxsize=1)
        self.q_predictions = Queue(maxsize=1)

        # load model
        self.model = load_learner(model_path)

        self.tfms = get_transforms(do_flip=False)
        self.image_size = (360, 480)

    def predict(self, img):

        t = torch.tensor(np.ascontiguousarray(np.flip(img, 2)).transpose(2,0,1)).float()/255
        t = Image(t) # fastai.vision.Image, not PIL.Image
        # t = t.apply_tfms(self.tfms, size=self.image_size)
        pred_class, pred_idx, outputs = self.model.predict(t)

        return pred_class

    def main(self):

        while True:

                img = self.q_images.get()

                pred = self.predict(img)

                if self.q_predictions.qsize() == 0:
                    self.q_predictions.put(pred)

