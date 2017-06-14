import tensorflow as tf
import time
from . import help
from . import flow
from .ops import op_create, identity
from .ops import HEADER, LINE
from .framework import create_framework
from darkflow.dark.darknet import Darknet
from darkflow.utils.loader import create_loader
import json
import os


class TFNet(object):

    OLD_GRAPH_MSG = 'Resolving old graph def {} (no guarantee)'

    _TRAINER = dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'sgd': tf.train.GradientDescentOptimizer
    })

    # imported methods
    _get_fps = help._get_fps
    say = help.say
    train = flow.train
    camera = help.camera
    predict = flow.predict
    return_predict = flow.return_predict
    to_darknet = help.to_darknet

    def __init__(self, FLAGS, darknet=None):
        self.ntrain = 0

        if isinstance(FLAGS, dict):
            from ..defaults import argHandler
            newFLAGS = argHandler()
            newFLAGS.setDefaults()
            newFLAGS.update(FLAGS)
            FLAGS = newFLAGS

        self.FLAGS = FLAGS
        if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
            self.say('\nLoading from .pb and .meta')
            self.graph = tf.Graph()
            device_name = FLAGS.gpuName if FLAGS.gpu > 0.0 else None
            with tf.device(device_name):
                with self.graph.as_default() as g:
                    self.build_from_pb()
            return

        if darknet is None:
            darknet = Darknet(FLAGS)
            self.ntrain = len(darknet.layers)

        self.darknet = darknet
        args = [darknet.meta, FLAGS]
        self.num_layer = len(darknet.layers)
        self.framework = create_framework(*args)

        self.meta = darknet.meta

        self.say('\nBuilding net ...')
        start = time.time()
        self.graph = tf.Graph()
        device_name = FLAGS.gpuName if FLAGS.gpu > 0.0 else None
        with tf.device(device_name):
            with self.graph.as_default() as g:
                # Create the Network
                self.build_forward()
                # Perform meta-operations like summary.
                self.setup_meta_ops()
        self.say('Finished in {}s\n'.format(time.time() - start))

    def build_from_pb(self):
        """TODO documentation"""
        with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name="")
        with open(self.FLAGS.metaLoad, 'r') as fp:
            self.meta = json.load(fp)
        self.framework = create_framework(self.meta, self.FLAGS)

        # Placeholders
        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.feed = dict()  # Tensorflow placeholders
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

        self.setup_meta_ops()

    def build_forward(self):
        """Creates the network"""
        verbalise = self.FLAGS.verbalise

        # Placeholders
        inp_size = [None] + self.meta['inp_size']
        self.inp = tf.placeholder(tf.float32, inp_size, 'input')
        self.feed = dict()  # Tensorflow placeholders

        # Build the forward pass
        flayer = identity(self.inp)
        roof = self.num_layer - self.ntrain
        self.say(HEADER, LINE)
        for i, dlayer in enumerate(self.darknet.layers):
            scope = '{}-{}'.format(str(i), dlayer.type)
            args = [dlayer, flayer, i, roof, self.feed]
            # Create a Tensorflow compatible layer from Darknet Layer
            flayer = op_create(*args)
            mess = flayer.verbalise()
            self.say(mess)
        self.say(LINE)

        # Keep the last layer
        self.top = flayer
        self.out = tf.identity(flayer.out, name='output')

    def setup_meta_ops(self):
        """ TODO Documentation"""
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
        })

        utility = min(self.FLAGS.gpu, 1.)
        if utility > 0.0:
            self.say('GPU mode with {} usage'.format(utility))
            cfg['gpu_options'] = tf.GPUOptions(
                per_process_gpu_memory_fraction=utility)
            cfg['allow_soft_placement'] = True
        else:
            self.say('Running entirely on CPU')
            cfg['device_count'] = {'GPU': 0}

        if self.FLAGS.train:
            self.build_train_op()

        if self.FLAGS.summary is not None:
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')

        self.sess = tf.Session(config=tf.ConfigProto(**cfg))
        # Initialize TF Session
        self.sess.run(tf.global_variables_initializer())

        if not self.ntrain:
            return
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.FLAGS.keep)
        if self.FLAGS.load != 0:
            self.load_from_ckpt()

        if self.FLAGS.summary is not None:
            self.writer.add_graph(self.sess.graph)

    def savepb(self):
        """
        Create a standalone const graph def that
        C++ can load and run.
        """
        darknet_pb = self.to_darknet()
        flags_pb = self.FLAGS
        flags_pb.verbalise = False

        flags_pb.train = False
        # rebuild another tfnet. all const.
        tfnet_pb = TFNet(flags_pb, darknet_pb)
        tfnet_pb.sess = tf.Session(graph=tfnet_pb.graph)
        # tfnet_pb.predict() # uncomment for unit testing
        name = 'built_graph/{}.pb'.format(self.meta['name'])
        os.makedirs(os.path.dirname(name), exist_ok=True)
        # Save dump of everything in meta
        with open('built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
            json.dump(self.meta, fp)
        self.say('Saving const graph def to {}'.format(name))
        graph_def = tfnet_pb.sess.graph_def
        tf.train.write_graph(graph_def, './', name, False)

    def build_train_op(self):
        self.framework.loss(self.out)
        self.say('Building {} train op'.format(self.meta['model']))
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = self._TRAINER[self.FLAGS.trainer](self.learning_rate)
        gradients = optimizer.compute_gradients(self.framework.loss)
        self.train_op = optimizer.apply_gradients(gradients)

    def load_from_ckpt(self):
        if self.FLAGS.load < 0:  # load lastest ckpt
            with open(self.FLAGS.backup + 'checkpoint', 'r') as f:
                last = f.readlines()[-1].strip()
                load_point = last.split(' ')[1]
                load_point = load_point.split('"')[1]
                load_point = load_point.split('-')[-1]
                self.FLAGS.load = int(load_point)

        load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
        load_point = '{}-{}'.format(load_point, self.FLAGS.load)
        self.say('Loading from {}'.format(load_point))
        try:
            self.saver.restore(self.sess, load_point)
        except:
            self.load_old_graph(load_point)

    def load_old_graph(self, ckpt):
        ckpt_loader = create_loader(ckpt)
        self.say(self.OLD_GRAPH_MSG.format(ckpt))

        for var in tf.global_variables():
            name = var.name.split(':')[0]
            args = [name, var.get_shape()]
            val = ckpt_loader(args)
            assert val is not None, 'Cannot find and load {}'.format(var.name)
            shp = val.shape
            plh = tf.placeholder(tf.float32, shp)
            op = tf.assign(var, plh)
            self.sess.run(op, {plh: val})

