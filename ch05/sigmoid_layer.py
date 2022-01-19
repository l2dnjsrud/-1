from turtle import forward


# coding: utf-8
import numpy as np


class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out                #순전파의 출력을 인스턴스 변수 out에 보관

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx