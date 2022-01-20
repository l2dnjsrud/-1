from collections import OrderedDict
from multiprocessing import pool
from random import random
import re
from ch07.pooling import Pooling
from common.layers import Affine, Convolution, Relu, SoftmaxWithLoss
from common.util import conv_output_size


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                conv_param={'filter_num':30, 'filter_size':5, 
                           'pad':0, 'stride':1}, 
                hidden_size = 100, ouput_size = 10, weight_init_std = 0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        #가중치 매게변수 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, ouput_size)
        self.params['b3'] = np.zeros(ouput_size)

        # CNN을 구성하는 계층들 초기화
        self.layers = OrderedDict()
        self.layers['conv1'] = Convolution(self.params['W1'], self.params['b1'], self.params['stride'], self.params['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layers = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """        
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradeint(self, x, t):
        #순전파
        self.loss(x, t)

        #역전파
        dout = 1
        dout = self.last_layers.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #결과저장
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['w2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads