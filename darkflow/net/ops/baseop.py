import tensorflow as tf
import numpy as np

FORM = '{:>6} | {:>6} | {:<32} | {}'
FORM_ = '{}+{}+{}+{}'
LINE = FORM_.format('-' * 7, '-' * 8, '-' * 34, '-' * 15)
HEADER = FORM.format('Source', 'Train?', 'Layer description', 'Output size')


def _shape(tensor):  # work for both tf.Tensor & np.ndarray
    if type(tensor) in [tf.Variable, tf.Tensor]:
        return tensor.get_shape()
    else:
        return tensor.shape


def _name(tensor):
    return tensor.name.split(':')[0]


class BaseOp(object):
    """
    BaseOp objects initialise with a darknet's `layer` object
    and input tensor of that layer `inp`, it calculates the
    output of this layer and place the result in self.out
    """

    # let slim take care of the following vars
    _SLIM = ['gamma', 'moving_mean', 'moving_variance']

    def __init__(self, layer, inp, num, roof, feed):
        self.inp = inp  # BaseOp
        self.num = num  # int
        self.out = None  # tf.Tensor
        self.lay = layer

        self.scope = '{}-{}'.format(str(self.num), self.lay.type)
        self.gap = roof - self.num
        self.var = not self.gap > 0
        self.act = 'Load '
        self.convert(feed)
        if self.var:
            self.train_msg = 'Yep! '
        else:
            self.train_msg = 'Nope '
        self.forward()

    def convert(self, feed):
        """ Convert self.lay to variables & placeholders

        @type feed
        @param feed
        """
        for idx in self.lay.wshape:
            self._wrap_variable(idx)
        for ph in self.lay.h:
            self._wrap_pholder(ph, feed)

    def _wrap_variable(self, varidx):
        """Wrap Darknet layer weights into a Tensorflow Variable.

        For more information on Tensorflow Variables:
            https://www.tensorflow.org/programmers_guide/variables

        @type var
        @param var
        """
        val = self.lay.w.get(varidx, None)
        if val is None:
            shape = self.lay.wshape[varidx]
            args = [0., 1e-2, shape]
            if 'moving_mean' in varidx:
                val = np.zeros(shape)
            elif 'moving_variance' in varidx:
                val = np.ones(shape)
            else:
                val = np.random.normal(*args)
            self.lay.w[varidx] = val.astype(np.float32)
            self.act = 'Init '
        if not self.var:
            return

        val = self.lay.w[varidx]
        self.lay.w[varidx] = tf.constant_initializer(val)
        if varidx in self._SLIM:
            return
        with tf.variable_scope(self.scope):
            self.lay.w[varidx] = tf.get_variable(
                varidx,
                shape=self.lay.wshape[varidx],
                dtype=tf.float32,
                initializer=self.lay.w[varidx])

    def _wrap_pholder(self, ph, feed):
        """wrap Darknet layer.h into Tensorflow placeholders"""
        phtype = type(self.lay.h[ph])
        if phtype is not dict:
            return

        sig = '{}/{}'.format(self.scope, ph)
        val = self.lay.h[ph]

        self.lay.h[ph] = tf.placeholder_with_default(
            val['dfault'], val['shape'], name=sig)
        feed[self.lay.h[ph]] = val['feed']

    def verbalise(self):  # console speaker
        msg = str()
        inp = _name(self.inp.out)
        if inp == 'input':
            msg = FORM.format('', '', 'input', _shape(self.inp.out)) + '\n'
        if not self.act:
            return msg
        return msg + FORM.format(self.act, self.train_msg,
                                 self.speak(), _shape(self.out))

    def speak(self):
        pass
