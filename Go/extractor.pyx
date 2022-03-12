cimport cython
import numpy as np
cimport numpy as cnp

ctypedef cnp.int_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object extract_c(cnp.ndarray[DTYPE_t, ndim=2] state):
    feature_num = dict()
    feature_raw = dict()
    for p in range(1, 4):
        size = 6 - p
        for i in range(size):
            for j in range(size):
                # location independent
                li_key = encode_c(state[i:i + p, j:j + p], p)
                li_num = feature_num.get(li_key, 0) + 1
                feature_num[li_key] = li_num
                feature_raw[li_key] = state[i:i + p, j:j + p]
                # location dependent
                ld_local = np.zeros((5, 5), dtype=np.int_)
                ld_local[i:i + p, j:j + p] = state[i:i + p, j:j + p]
                ld_key = encode_c(ld_local, 5)
                ld_num = feature_num.get(ld_key, 0) + 1
                feature_num[ld_key] = ld_num
                feature_raw[ld_key] = ld_local

    return feature_num, feature_raw

@cython.boundscheck(False)
@cython.wraparound(False)
cdef str encode_c(cnp.ndarray[DTYPE_t, ndim=2] state, int size):
    cdef str encoded = ""
    cdef int i = 0
    cdef int j = 0
    for i in range(size):
        for j in range(size):
            encoded += str(state[i, j])
    return encoded

def extract(state):
    return extract_c(state)

def encode(state, size):
    return encode_c(state, size)