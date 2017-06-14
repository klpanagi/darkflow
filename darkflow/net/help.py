"""
tfnet secondary (helper) methods
"""
from time import time as timer
import tensorflow as tf
import sys
import cv2
import os


def say(self, *msgs):
    if not self.FLAGS.verbalise:
        return
    msgs = list(msgs)
    for msg in msgs:
        if msg is None:
            continue
        print(msg)


def _get_fps(self, frame):
    elapsed = int()
    start = timer()
    preprocessed = self.framework.preprocess(frame)
    feed_dict = {self.inp: [preprocessed]}
    net_out = self.sess.run(self.out, feed_dict)[0]
    processed = self.framework.postprocess(net_out, frame, False)
    return timer() - start


def camera(self):
    file = self.FLAGS.demo
    SaveVideo = self.FLAGS.saveVideo

    if file == 'camera':
        file = 0
    else:
        assert os.path.isfile(file), 'file {} does not exist'.format(file)

    camera = cv2.VideoCapture(file)
    assert camera.isOpened(), 'Cannot capture source'

    cv2.namedWindow('', 0)
    _, frame = camera.read()
    height, width, _ = frame.shape
    cv2.resizeWindow('', width, height)

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(* 'XVID')
        if file == 0:
            fps = 1 / self._get_fps(frame)
            if fps < 1:
                fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter('video.avi', fourcc, fps, (width,
                                                                 height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()

    elapsed = int()
    start = timer()
    self.say('Press [ESC] to quit demo')
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print('\nEnd of Video')
            break
        preprocessed = self.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)

        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            feed_dict = {self.inp: buffer_pre}
            net_out = self.sess.run(self.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                postprocessed = self.framework.postprocess(
                    single_out, img, False)
                if SaveVideo:
                    videoWriter.write(postprocessed)
                cv2.imshow('', postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                '{0:3.3f} FPS'.format(elapsed / (timer() - start)))
            sys.stdout.flush()
        choice = cv2.waitKey(1)
        if choice == 27:
            break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    cv2.destroyAllWindows()


def to_darknet(self):
    darknet_ckpt = self.darknet

    with self.graph.as_default() as g:
        for var in tf.global_variables():
            name = var.name.split(':')[0]
            var_name = name.split('-')
            l_idx = int(var_name[0])
            w_sig = var_name[1].split('/')[-1]
            l = darknet_ckpt.layers[l_idx]
            l.w[w_sig] = var.eval(self.sess)

    for layer in darknet_ckpt.layers:
        for ph in layer.h:
            layer.h[ph] = None

    return darknet_ckpt
