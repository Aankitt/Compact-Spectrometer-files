import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy.linalg import svd


def SVD(matrix, tolerance):
    T = np.array(matrix)
    S, V, DT = svd(matrix, full_matrices=True, compute_uv=True)

    for i in np.arange(len(V)):
        if V[i] < tolerance:
            V[i] = 0
        else:
            V[i] = V[i]

    for j in np.arange(len(V)):
        if V[j] == 0:
            V[j] = V[j]
        else:
            V[j] = np.reciprocal(V[j])

    Diag_matrix = np.zeros((T.shape[0], T.shape[0]))
    Diag_matrix[:T.shape[1], :T.shape[1]] = np.diag(V)

    # Inverse procedure #
    D = np.transpose(DT)
    ST = np.transpose(S)
    MM = Diag_matrix @ ST
    T_inv = [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*MM)] for A_row in D]
    T_inv = pd.DataFrame(T_inv)
    return V, T_inv
