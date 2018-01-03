from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from keras.layers import Activation


class ReLUs(Activation):
    def __init__(self, activation, **kwargs):
        super(ReLUs, self).__init__(activation, **kwargs)
        self.__name__ = 'relus'

    @staticmethod
    def config():
        get_custom_objects().update({'relus': ReLUs(ReLUs.relus)})

    @staticmethod
    def relus(Z):
        e_param = 1.1
        pi = K.variable((3.14))
        m = e_param + (K.sigmoid(K.sin(Z)) - K.sigmoid(K.cos(Z)) * K.exp(K.sqrt(pi)))
        A = K.maximum(m, Z)
        return A