# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)   #가중치 매개변수

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

np.argmax(p)  #최대값 인덱스

t = np.array([0,0,1])  # 정답 레이블
net.loss(x, t)

def f(w):
    return net.loss(x, t)  ## f = lambda w : net.loss(x, t) 람다기법으로 간략하게 쓸 수 있다 이쪽이 더 좋음

dw = numerical_gradient(f, net.W)
print(dw)