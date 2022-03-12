import collections
import time
import extractor
import numpy as np


def extract_local_shape(state):
    feature = collections.Counter()
    feature_raw = {}
    for p in range(1, 4):
        size = 6 - p
        for i in range(size):
            for j in range(size):
                # location independent
                li_feature = encode_shape(state[i:i + p, j:j + p], size=p)
                feature[li_feature] += 1
                feature_raw[li_feature] = state[i:i + p, j:j + p]
                # location dependent
                ld_local = np.zeros((5, 5), dtype=np.int_)
                ld_local[i:i + p, j:j + p] = state[i:i + p, j:j + p]
                ld_feature = encode_shape(ld_local, size=5)
                feature[ld_feature] += 1
                feature_raw[ld_feature] = ld_local

    return dict(feature), feature_raw


def encode_shape(p, size):
    return ''.join([str(p[i][j]) for i in range(size) for j in range(size)])


if __name__ == '__main__':
    state = np.array([[2, 0, 0, 1, 0], [0, 0, 1, 0, 1], [2, 1, 1, 1, 1], [2, 2, 1, 2, 1], [0, 2, 2, 0, 2]])

    # py
    start = time.time()
    npy = {}
    for _ in range(10000):
        npy, fpy = extract_local_shape(state)
    end = time.time()
    print(end - start)

    start = time.time()
    ncy = {}
    # cy
    for _ in range(10000):
        ncy, fcy = extractor.extract(state)
    end = time.time()
    print(end - start)
