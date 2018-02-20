from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from keras.layers import Activation


class ReLUs(Activation):
    def __init__(self, activation, **kwargs):
        super(ReLUs, self).__init__(activation, **kwargs)
        self.__name__ = 'relus'

    @staticmethod
    def config(e_param):
        ReLUs.e_param = e_param
        get_custom_objects().update({'relus': ReLUs(ReLUs.relus)})

    @staticmethod
    def relus(Z):
        e = ReLUs.e_param
        pi = K.variable((3.14))
        m = e * (K.sigmoid(K.sin(Z)) - K.sigmoid(K.cos(Z)) * K.exp(K.sqrt(pi)))
        A = K.maximum(m, Z)
        return A