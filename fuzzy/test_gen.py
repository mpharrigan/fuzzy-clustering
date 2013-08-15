from __future__ import division

import numpy as np

eps = 1.0e-10

def outer(v1, v2):
    return np.outer(v1, v2)

def putter(v1, v2):
    return np.array([v1, v2]).transpose()

def outernorm(v1, v2, which='rows'):
#     import pdb; pdb.set_trace()
    mat = np.outer(v1, v2)
    
    if which == 'columns':
        axis = 0
    elif which == 'rows':
        axis = 1
    else:
        raise ValueError()
    
    sums = np.sum(mat, axis=axis)
    for i in xrange(mat.shape[axis]):
        if sums[i] > eps:
            if which == 'columns':
                mat[:, i] = mat[:, i] / sums[i]
            elif which == 'rows':
                mat[i] = mat[i] / sums[i]
                
    return mat



test = [
        [[1, 0], [1, 0]],
        [[1, 0], [0, 1]],
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.9, 0.1], [0.8, 0.2]]
        ]

def main(func=outernorm):
    for case in test:
        print("\nCase:")
        v1 = np.array(case[0], dtype='float32')
        v2 = np.array(case[1], dtype='float32')
        result = func(v1, v2)

        print("Result")
        print(result)
        
        expectedv2 = np.dot(v1, result)
        expectedv1 = np.dot(result, v2)
        
        print("v1 vs expected v1")
        print(v1)
        print(expectedv1)
        
        print("v2 vs expected v2")
        print(v2)
        print(expectedv2)
        
        
#         diff = np.abs(result - expected)
#         max_diff = np.max(diff)
        # print(diff)
        # print(max_diff)
        
if __name__ == "__main__":
    main()
        
