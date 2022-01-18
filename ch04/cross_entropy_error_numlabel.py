def cross_entropy_error(y, t):
    if y.ndim == 1:     #y가 1차원, 즉 데이터 하나
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size   #각 데이터의 정답 레이블에 해당하는 신경망 출력을 추출