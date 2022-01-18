# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

#데이터 준비
x = np.arange(0, 6, 0.1) #0에서 6까지 0.1 간격으로 생성
y = np.sin(x)

#그래프 그리기
plt.plot(x,y)  #그래프 그리기
plt.show()     #위에서 그린 그래프 출력
plt.savefig('simple_graph.png') #리눅스는 그래프 출력 GUI 사용 불가, 따라서 따로 저장해야함