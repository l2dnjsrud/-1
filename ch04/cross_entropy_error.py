# coding: utf-8
import numpy as np
def cross_entropy_error(y, t):
    delta = 1e - 7
    return - np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]      #정답이 2인 경우
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]     # 예1 : '2'일 확률이 가장 높다고 추청(0.6)

cross_entropy_error(np.array(y), np.array(t))